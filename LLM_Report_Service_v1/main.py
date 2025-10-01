# main.py (最終完整版)

import pandas as pd
from pathlib import Path

# 匯入我們所有的分析模組
# 每個模組都負責一項獨立、清晰的任務
from analysis.camera_clusterer import cluster_cameras_by_distance
from analysis.stay_point_detector import find_stay_points_v2
from analysis.trip_segmenter import segment_trips_v2
from analysis.pattern_clusterer import find_regular_patterns_v10
from analysis.anomaly_detector import find_anomalies_v2

def run_full_analysis(full_df: pd.DataFrame, target_plate: str):
    """
    執行從數據讀取到最終摘要生成的完整分析流程。
    """
    # --- 步驟 A: 攝影機地理分群 ---
    # 這是所有分析的基礎，將地理位置相近的攝影機劃為同一個區域。
    print("\n--- 步驟 A: 攝影機地理分群 ---")
    unique_cameras = full_df[['攝影機', '攝影機名稱', '經度', '緯度', '單位']].drop_duplicates(subset=['攝影機']).reset_index(drop=True)
    cameras_with_area_id = cluster_cameras_by_distance(unique_cameras, radius_meters=200)
    print(f"分群完成，共 {cameras_with_area_id['LocationAreaID'].nunique()} 個獨立位置區域。")

    # --- 步驟 B: 為目標車輛尋找停留點 ---
    # 找出所有停留超過 20 分鐘的「停留事件」。
    print(f"\n--- 步驟 B: 為車輛 {target_plate} 尋找停留點 ---")
    vehicle_data = full_df[full_df['車牌'] == target_plate].copy()
    vehicle_data_with_area = pd.merge(vehicle_data, cameras_with_area_id[['攝影機', 'LocationAreaID']], on='攝影機', how='left')
    stay_points_result = find_stay_points_v2(vehicle_data_with_area, time_threshold_minutes=20)
    
    if not stay_points_result:
        print("在指定時間範圍內未找到任何停留點，分析中止。")
        return
    print(f"找到 {len(stay_points_result)} 次停留事件。")
    
    # --- 步驟 C: 根據停留點進行行程分段 ---
    # 將兩個停留點之間的移動定義為一次「行程」。
    print(f"\n--- 步驟 C: 為車輛 {target_plate} 進行行程分段 ---")
    trips_result = segment_trips_v2(vehicle_data_with_area, stay_points_result)
        
    if not trips_result:
        print("未切割出任何行程，分析中止。")
        return
    print(f"切割出 {len(trips_result)} 段行程。")

    # --- 步驟 D: 識別「規律性」資訊 ---
    # 專注於尋找重複出現的行為模式、主要據點和停留點統計。
    print(f"\n--- 步驟 D: 識別規律模式與主要據點 ---")
    pattern_result = find_regular_patterns_v10(trips_result, stay_points_result)
    regular_summary = pattern_result["summary"]
    area_map = pattern_result["area_map"]
    trips_df = pattern_result["trips_df"] # 取得帶有特徵的 trips_df 以供下一步使用

    # --- 步驟 E: 偵測「異常」資訊 ---
    # 接收所有行程和已確認的規律，找出路徑異常和時間異常。
    print(f"\n--- 步驟 E: 偵測異常事件 ---")
    anomalies = find_anomalies_v2(trips_df, regular_summary["regular_patterns"])

    # --- 步驟 F: 組合最終報告 ---
    # 將規律性摘要和異常摘要合併成一份完整的報告原料。
    final_summary = {**regular_summary, **anomalies}

    # --- 步驟 G: 呈現最終分析摘要 ---
    print("\n" + "="*50)
    print("          最終分析摘要 (準備提交給 LLM)")
    print("="*50)
    
    print(f"\n[主要據點資訊]")
    if final_summary['base_info']['primary']:
        hb = final_summary['base_info']['primary']
        print(f"  [主要基地] {hb['name']} ({hb['area_id']})")
        print(f"    - 停留次數: {hb['visit_count']} 次")
        print(f"    - 通常停留時段: {hb['avg_arrival_time']} ~ {hb['avg_departure_time']}")
    
    if final_summary['base_info']['secondary']:
        for i, sb in enumerate(final_summary['base_info']['secondary']):
            print(f"  [次要基地 {i+1}] {sb['name']} ({sb['area_id']})")
            print(f"    - 停留次數: {sb['visit_count']} 次")
            print(f"    - 平均停留時段: {sb['avg_arrival_time']} ~ {sb['avg_departure_time']}")

    print(f"\n[所有停留點統計 (按總時長排序)]")
    if final_summary['all_stay_points_stats']:
        for i, sp in enumerate(final_summary['all_stay_points_stats'], 1):
            print(f"  {i}. {sp['name']} ({sp['area_id']})")
            print(f"     - 來訪: {sp['visit_count']} 次, 總計: {sp['total_duration_hours']} 小時, 平均: {sp['avg_duration_hours']} 小時")
            print(f"     - 通常時段: {sp['avg_arrival_time']} ~ {sp['avg_departure_time']}")
    
    print(f"\n[已確認的規律模式 (發生 >= 4 次)]")
    if final_summary['regular_patterns']:
        for p in final_summary['regular_patterns']:
            start_name = area_map.get(p['start_area_id'], "未知")
            end_name = area_map.get(p['end_area_id'], "未知")
            print(f"  - 從 [{start_name}] 到 [{end_name}]")
            print(f"    - 發生 {p['occurrence_count']} 次，通常時段為 {p['avg_start_time']} ~ {p['avg_end_time']}。")

    print(f"\n[路徑異常或待觀察的模式 (發生 < 4 次)]")
    if final_summary['infrequent_patterns']:
        for p in final_summary['infrequent_patterns']:
            start_name = area_map.get(p['start_area_id'], "未知")
            end_name = area_map.get(p['end_area_id'], "未知")
            print(f"  - 從 [{start_name}] 到 [{end_name}] (發生 {p['occurrence_count']} 次)")

    print(f"\n[時間異常事件 (耗時與平時差異顯著的規律行程)]")
    if final_summary['duration_anomalies']:
        for anomaly in final_summary['duration_anomalies']:
            pattern_info = next((p for p in final_summary['regular_patterns'] if p['signature'] == anomaly['pattern_signature']), None)
            if pattern_info:
                start_name = area_map.get(pattern_info['start_area_id'], "未知")
                end_name = area_map.get(pattern_info['end_area_id'], "未知")
                print(f"  - 日期: {anomaly['date']}")
                print(f"    - 路線: 從 [{start_name}] 到 [{end_name}]")
                print(f"    - 異常耗時: {anomaly['actual_duration_minutes']} 分鐘 (屬於此路線耗時的第 {anomaly['percentile']}% 等級)")
                print(f"    - 判斷依據: 正常耗時上限約 {anomaly['normal_upper_bound_minutes']} 分鐘 (基於 IQR 法計算)")
                print(f"    - 路線統計: 平均耗時 {anomaly['mean_duration_for_pattern']} 分鐘，標準差 {anomaly['std_dev_for_pattern']} 分鐘")
    else:
        print("  無")


if __name__ == '__main__':
    try:
        # --- 檔案路徑設定 ---
        current_file_path = Path(__file__)
        project_root_path = current_file_path.parent
        DATA_FILE_PATH = project_root_path / 'data' / 'generated_car_data_month.csv'

        # --- 讀取與基礎預處理 ---
        full_data = pd.read_csv(DATA_FILE_PATH)
        full_data['datetime'] = pd.to_datetime(full_data['日期'] + ' ' + full_data['時間'])
        full_data = full_data.sort_values(by='datetime').reset_index(drop=True)
        print(f"成功讀取完整資料，共 {len(full_data)} 筆紀錄。")
        
        # --- 為了測試，手動製造一次時間異常事件 ---
        # 找到 8/20 的通勤紀錄並將其行程時間人為地拉長 40 分鐘來模擬塞車
        mask = (full_data['車牌'] == 'BKE-6831') & (full_data['日期'] == '2025-08-20') & (full_data['時間'].str.startswith('08:'))
        full_data.loc[mask, 'datetime'] = full_data.loc[mask, 'datetime'] + pd.Timedelta(minutes=40)
        print(" (已手動注入一次時間異常用於測試) ")
        
        # --- 執行主分析流程 ---
        run_full_analysis(full_data, target_plate='BKE-6831')

    except FileNotFoundError:
        print(f"錯誤：找不到檔案 {DATA_FILE_PATH}。請確認檔案路徑。")
    except Exception as e:
        print(f"執行過程中發生未預期的錯誤：{e}")