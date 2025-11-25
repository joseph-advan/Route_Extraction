# analysis/meeting_analyzer.py

import pandas as pd
from analysis.advanced_stay_detector import find_advanced_stay_points, haversine_distance
# 【新增匯入】需要用到分群功能來產生 LocationAreaID
from analysis.camera_clusterer import cluster_cameras_by_distance

def check_time_overlap(start1, end1, start2, end2):
    """檢查兩個時間區段是否有重疊"""
    overlap_start = max(start1, start2)
    overlap_end = min(end1, end2)
    return overlap_start < overlap_end

def run_dual_vehicle_meeting_analysis(df_a: pd.DataFrame, df_b: pd.DataFrame, 
                                      plate_a: str, plate_b: str):
    """
    執行雙車碰面分析的主流程
    """
    print(f"\n--- 開始分析 {plate_a} 與 {plate_b} 的碰面紀錄 ---")

    # ==============================================================================
    # 步驟 0: 預處理 - 產生 LocationAreaID (這是修正 KeyError 的關鍵)
    # ==============================================================================
    print("正在進行地點分群與預處理...")
    
    # 1. 合併兩車資料以建立統一的地點分群 (避免兩車的 Area ID 定義不同)
    combined_df = pd.concat([df_a, df_b], ignore_index=True)
    
    # 2. 提取不重複攝影機
    unique_cameras = combined_df[['攝影機', '攝影機名稱', '經度', '緯度']].drop_duplicates(subset=['攝影機']).reset_index(drop=True)
    
    # 3. 執行分群 (產生 LocationAreaID)
    cameras_with_area = cluster_cameras_by_distance(unique_cameras, radius_meters=200)
    
    # 4. 將 LocationAreaID 合併回原始資料
    # 注意：需確保欄位名稱一致
    df_a = pd.merge(df_a, cameras_with_area[['攝影機', 'LocationAreaID']], on='攝影機', how='left')
    df_b = pd.merge(df_b, cameras_with_area[['攝影機', 'LocationAreaID']], on='攝影機', how='left')
    
    # ==============================================================================
    # 步驟 1: 分別計算兩台車的停留點 (使用進階混合邏輯)
    # ==============================================================================
    print(f"正在計算 {plate_a} 的停留點 (含隱性停留)...")
    stays_a = find_advanced_stay_points(df_a)
    
    print(f"正在計算 {plate_b} 的停留點 (含隱性停留)...")
    stays_b = find_advanced_stay_points(df_b)
    
    print(f"-> {plate_a} 共有 {len(stays_a)} 個停留點")
    print(f"-> {plate_b} 共有 {len(stays_b)} 個停留點")
    
    meetings = []
    
    # 2. 雙重迴圈比對 (Matching)
    # 閾值設定：距離 80 公尺內視為碰面 (無視 Area ID，只看物理距離)
    MEETING_DISTANCE_THRESHOLD = 80 
    
    for s_a in stays_a:
        for s_b in stays_b:
            
            # [檢查 1] 時間是否有重疊
            if check_time_overlap(s_a['start_time'], s_a['end_time'],
                                  s_b['start_time'], s_b['end_time']):
                
                # [檢查 2] 物理距離是否夠近 (使用平均經緯度)
                dist_meters = haversine_distance(
                    s_a['center_lon'], s_a['center_lat'],
                    s_b['center_lon'], s_b['center_lat']
                )
                
                if dist_meters <= MEETING_DISTANCE_THRESHOLD:
                    # 賓果！抓到碰面
                    
                    # 計算重疊時間長度
                    overlap_start = max(s_a['start_time'], s_b['start_time'])
                    overlap_end = min(s_a['end_time'], s_b['end_time'])
                    duration = (overlap_end - overlap_start).total_seconds() / 60
                    
                    # 判斷是否為跨區碰面 (供報告參考)
                    is_cross_area = False
                    location_hint = s_a['location_desc']
                    
                    # 安全存取 area_id_hint，避免有些 Gap Stay 可能沒有這個欄位
                    aid_a = s_a.get('area_id_hint')
                    aid_b = s_b.get('area_id_hint')
                    
                    if aid_a and aid_b:
                        if aid_a != aid_b:
                            is_cross_area = True
                            location_hint = f"{aid_a} 與 {aid_b} 交界"

                    meetings.append({
                        'start_time': overlap_start,
                        'end_time': overlap_end,
                        'duration_mins': round(duration, 1),
                        'distance_meters': round(dist_meters, 1),
                        'location_desc': location_hint,
                        'type_a': s_a['type'],
                        'type_b': s_b['type'],
                        'is_cross_area': is_cross_area
                    })
    
    # 3. 輸出結果報告
    if not meetings:
        print("\n[分析結果]：未發現兩車有任何碰面或共同停留的跡象。")
    else:
        print(f"\n[分析結果]：共發現 {len(meetings)} 次碰面事件！")
        print("="*60)
        meetings.sort(key=lambda x: x['start_time'])
        
        for idx, m in enumerate(meetings):
            time_str = m['start_time'].strftime('%Y-%m-%d %H:%M')
            print(f"{idx+1}. [{time_str}] 持續 {m['duration_mins']} 分鐘")
            print(f"   - 地點：{m['location_desc']} (距離僅 {m['distance_meters']} 公尺)")
            print(f"   - 類型：A車({m['type_a']}) + B車({m['type_b']})")
            if m['is_cross_area']:
                print(f"   - 備註：⚠️ 這是跨區域的邊界碰面 (Area ID 不同但距離近)")
            print("-" * 30)