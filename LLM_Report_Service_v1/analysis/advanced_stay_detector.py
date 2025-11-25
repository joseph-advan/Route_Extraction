# analysis/advanced_stay_detector.py

import pandas as pd
import numpy as np

# ==========================================
# 1. 核心距離公式
# ==========================================
def haversine_distance(lon1, lat1, lon2, lat2):
    """計算地球上兩點的距離 (單位: 公尺)"""
    R = 6371000  # 地球半徑
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c

# ==========================================
# 2. 進階停留點偵測邏輯 (Hybrid)
# ==========================================
def find_advanced_stay_points(vehicle_df: pd.DataFrame, 
                              time_threshold_mins: int = 20, 
                              gap_speed_threshold_kph: float = 10.0) -> list:
    """
    偵測車輛的停留點，包含「顯性連續停留」與「隱性區間停留」。
    
    邏輯流程：
    1. 資料前處理：排序。
    2. 兩階段判定：
       - 階段一 (Time Filter): 檢查相鄰兩點的時間差。
       - 階段二 (Space Validation): 若時間差大，檢查移動速度。
    
    Args:
        vehicle_df: 單一車輛的軌跡 DataFrame (需包含 'datetime', '經度', '緯度', 'LocationAreaID')
        time_threshold_mins: 定義停留的最短時間 (預設 20 分鐘)
        gap_speed_threshold_kph: 定義隱性停留的最大移動速度 (預設 10 km/h)
                                 (若兩點間隔很久，但換算時速極低，視為停留)

    Returns:
        list of dict: 停留點列表
    """
    stays = []
    
    # 1. 資料清洗與排序
    if vehicle_df.empty:
        return []
    
    # 確保是副本以免影響原始資料
    df = vehicle_df.copy()
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.sort_values('datetime').reset_index(drop=True)
    
    # 強制轉型經緯度，避免字串運算錯誤
    df['經度'] = pd.to_numeric(df['經度'], errors='coerce')
    df['緯度'] = pd.to_numeric(df['緯度'], errors='coerce')
    df = df.dropna(subset=['經度', '緯度'])
    
    if len(df) < 2:
        return []

    # 2. 迭代檢查 (Scanning)
    # 我們採用「分段 (Segment)」的概念：
    # - 如果兩點很近 (時間 < 閾值)，視為同一個「活動區段」。
    # - 如果兩點很遠 (時間 >= 閾值)，則檢查這段空窗期是否為「隱性停留」。
    
    # 初始化第一個區段
    current_segment = [df.iloc[0]]
    
    for i in range(len(df) - 1):
        curr_rec = df.iloc[i]
        next_rec = df.iloc[i+1]
        
        # 計算時間差 (分鐘)
        time_diff = (next_rec['datetime'] - curr_rec['datetime']).total_seconds() / 60
        
        # -----------------------------------------------------------
        # 狀況 A: 連續活動 (時間差 < 閾值) -> 歸類為「顯性停留」的候選
        # -----------------------------------------------------------
        if time_diff < time_threshold_mins:
            current_segment.append(next_rec)
        
        # -----------------------------------------------------------
        # 狀況 B: 時間斷層 (時間差 >= 閾值) -> 觸發檢查機制
        # -----------------------------------------------------------
        else:
            # [1] 先結算上一個區段 (Explicit Stay Check)
            # 如果上一個區段累積的時間夠長，且都在附近，那就是「顯性停留」(例如路邊停車)
            if len(current_segment) > 1:
                seg_start = current_segment[0]['datetime']
                seg_end = current_segment[-1]['datetime']
                seg_duration = (seg_end - seg_start).total_seconds() / 60
                
                # 這裡簡單判斷：只要區段頭尾時間夠長，就算顯性停留
                # (因為時間差小代表連續被拍，通常是在同一區)
                if seg_duration >= time_threshold_mins:
                    avg_lat = np.mean([r['緯度'] for r in current_segment])
                    avg_lon = np.mean([r['經度'] for r in current_segment])
                    stays.append({
                        'type': 'Explicit Stay (顯性連續)',
                        'start_time': seg_start,
                        'end_time': seg_end,
                        'duration_minutes': round(seg_duration, 2),
                        'location_desc': f"{current_segment[0]['LocationAreaID']} (連續活動)",
                        'center_lat': avg_lat,
                        'center_lon': avg_lon,
                        'area_id_hint': current_segment[0]['LocationAreaID']
                    })

            # [2] 檢查這個斷層是否為「隱性停留」 (Gap Stay Check)
            # 這就是您設計的邏輯：時間久 + 距離短
            
            # 計算物理距離 (公里)
            dist_km = haversine_distance(
                curr_rec['經度'], curr_rec['緯度'],
                next_rec['經度'], next_rec['緯度']
            ) / 1000.0
            
            implied_speed = dist_km / (time_diff / 60.0)
            
            if implied_speed < gap_speed_threshold_kph:
                # 判定為停留！
                
                # 地點描述處理
                start_area = curr_rec['LocationAreaID'] if pd.notna(curr_rec.get('LocationAreaID')) else "未知"
                end_area = next_rec['LocationAreaID'] if pd.notna(next_rec.get('LocationAreaID')) else "未知"
                
                if start_area == end_area:
                    loc_desc = f"{start_area} (長時間靜止)"
                else:
                    loc_desc = f"{start_area} -> {end_area} (區間停留)"

                stays.append({
                    'type': 'Gap Stay (隱性區間)',
                    'start_time': curr_rec['datetime'],
                    'end_time': next_rec['datetime'],
                    'duration_minutes': round(time_diff, 2),
                    'location_desc': loc_desc,
                    'center_lat': (curr_rec['緯度'] + next_rec['緯度']) / 2, # 取中點
                    'center_lon': (curr_rec['經度'] + next_rec['經度']) / 2,
                    'area_id_hint': start_area, # 標記起點供參考
                    'avg_speed_kph': round(implied_speed, 2)
                })
            
            # [3] 重置區段，準備開始下一輪
            current_segment = [next_rec]

    # 迴圈結束後，別忘了檢查最後一段 Segment
    if len(current_segment) > 1:
        seg_start = current_segment[0]['datetime']
        seg_end = current_segment[-1]['datetime']
        seg_duration = (seg_end - seg_start).total_seconds() / 60
        
        if seg_duration >= time_threshold_mins:
            avg_lat = np.mean([r['緯度'] for r in current_segment])
            avg_lon = np.mean([r['經度'] for r in current_segment])
            stays.append({
                'type': 'Explicit Stay (顯性連續)',
                'start_time': seg_start,
                'end_time': seg_end,
                'duration_minutes': round(seg_duration, 2),
                'location_desc': f"{current_segment[0]['LocationAreaID']} (連續活動)",
                'center_lat': avg_lat,
                'center_lon': avg_lon,
                'area_id_hint': current_segment[0]['LocationAreaID']
            })
            
    return stays