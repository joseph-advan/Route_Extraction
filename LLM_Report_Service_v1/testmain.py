# main.py (最終完整版 - 整合所有功能)

import pandas as pd
from pathlib import Path
import json

# 匯入所有最新版本的分析模組
from analysis.camera_clusterer import cluster_cameras_by_distance
from analysis.stay_point_detector import find_stay_points_v2
from analysis.trip_segmenter import segment_trips_v2
from analysis.pattern_clusterer import find_regular_patterns_v12
from analysis.anomaly_detector import find_anomalies_v2
from security.anonymizer import anonymize_data
from security.deanonymizer import deanonymize_report
from llm_clients.cloud_client import generate_report_from_summary

def run_llm_reporting_flow(full_df: pd.DataFrame, target_plate: str, debug_mode: bool = False):
    """
    執行從數據分析到 LLM 報告生成的完整流程。
    """
    # --- 步驟 1-4: 執行本地數據分析引擎 ---
    if debug_mode:
        print("\n" + "="*70)
        print("## 步驟 1-4: 執行本地數據分析引擎...")
        print("="*70)
    
    unique_cameras = full_df[['攝影機', '攝影機名稱', '經度', '緯度', '單位']].drop_duplicates(subset=['攝影機']).reset_index(drop=True)
    cameras_with_area_id = cluster_cameras_by_distance(unique_cameras, radius_meters=200)
    if debug_mode: print(f"- 地理分群完成，共 {cameras_with_area_id['LocationAreaID'].nunique()} 個獨立位置區域。")

    vehicle_data = full_df[full_df['車牌'] == target_plate].copy()
    if vehicle_data.empty:
        print(f"錯誤：在資料集中找不到車牌 {target_plate} 的任何紀錄。")
        return

    vehicle_data_with_area = pd.merge(vehicle_data, cameras_with_area_id[['攝影機', 'LocationAreaID']], on='攝影機', how='left')
    stay_points_result = find_stay_points_v2(vehicle_data_with_area, time_threshold_minutes=20)
    if not stay_points_result: print(f"- 未找到 {target_plate} 的任何停留點，分析中止。"); return
    if debug_mode: print(f"- 找到 {len(stay_points_result)} 次停留事件。")
    
    trips_result = segment_trips_v2(vehicle_data_with_area, stay_points_result)
    if not trips_result: print(f"- 未切割出 {target_plate} 的任何行程，分析中止。"); return
    if debug_mode: print(f"- 切割出 {len(trips_result)} 段行程。")
    
    # 呼叫最新版的規律性分析模組
    pattern_result = find_regular_patterns_v12(trips_result, stay_points_result)
    regular_summary = pattern_result["summary"]
    area_map = pattern_result["area_map"]
    trips_df = pattern_result["trips_df"]
    
    # 呼叫最新版的異常偵測模組
    anomalies = find_anomalies_v2(trips_df, regular_summary["regular_patterns"])
    
    final_summary = {**regular_summary, **anomalies}
    if debug_mode: print("- 本地數據分析完成！")

    if debug_mode:
        print("\n" + "#"*70)
        print("## 輸出 1.1: 位置區域 ID 映射表 (Area Map)")
        print("#"*70)
        print(json.dumps(area_map, indent=2, ensure_ascii=False))

        print("\n" + "#"*70)
        print("## 輸出 1.2: 分析引擎產出的完整結構化摘要 (Final Summary)")
        print("#"*70)
        print(json.dumps(final_summary, indent=2, ensure_ascii=False, default=str))

    # --- 步驟 5: 去識別化 ---
    if debug_mode:
        print("\n" + "="*70)
        print("## 步驟 5: 對分析摘要進行去識別化...")
        print("="*70)
    anonymized_prompt, reversal_map = anonymize_data(final_summary, area_map, target_plate)
    
    if debug_mode:
        print("\n" + "#"*70)
        print("## 輸出 2: 去識別化後，準備發送給 LLM 的完整 Prompt 文本")
        print("#"*70)
        print(anonymized_prompt)

    # --- 步驟 6: 呼叫雲端 LLM ---
    if debug_mode:
        print("\n" + "="*70)
        print("## 步驟 6: 呼叫雲端 LLM API 生成報告草稿...")
        print("="*70)
    draft_report = generate_report_from_summary(anonymized_prompt)
    
    if debug_mode:
        print("\n" + "#"*70)
        print("## 輸出 3: LLM 回傳的、包含代碼的報告草稿 (Draft Report)")
        print("#"*70)
        print(draft_report)

    # --- 步驟 7: 還原報告 ---
    if debug_mode:
        print("\n" + "="*70)
        print("## 步驟 7: 將報告草稿還原為最終報告...")
        print("="*70)
    final_report = deanonymize_report(draft_report, reversal_map)
    if debug_mode: print("報告還原完成！")

    # --- 輸出 4: 最終呈現的報告 ---
    print("\n" + "#"*70)
    print("## 最終分析報告 (Final Report)")
    print("#"*70)
    print(final_report)


if __name__ == '__main__':
    # --- 除錯模式開關 ---
    # 將這裡改為 False，即可關閉所有詳細的中間步驟輸出，只顯示最終報告
    DEBUG_MODE = True

    try:
        current_file_path = Path(__file__)
        project_root_path = current_file_path.parent
        DATA_FILE_PATH = project_root_path / 'data' / 'vehicle_behavior_dataset_month_v4.csv' # 確保讀取的是最新的資料集

        full_data = pd.read_csv(DATA_FILE_PATH)
        full_data['datetime'] = pd.to_datetime(full_data['日期'] + ' ' + full_data['時間'])
        full_data = full_data.sort_values(by='datetime').reset_index(drop=True)
        print("成功讀取完整資料...")
        
        # --- 數字選項互動介面 ---
        available_plates = full_data['車牌'].unique()
        print("\n資料集中可用的車牌號碼：")
        for i, plate in enumerate(available_plates):
            print(f"  [{i+1}] {plate}")
        
        choice = -1
        while choice < 1 or choice > len(available_plates):
            try:
                choice_input = input(f"請輸入您想分析的車牌選項 [1-{len(available_plates)}] (預設 1): ") or "1"
                choice = int(choice_input)
                if not (1 <= choice <= len(available_plates)):
                    print("錯誤：輸入無效，請重新輸入。")
            except ValueError:
                print("錯誤：請輸入數字。")

        target_plate_input = available_plates[choice - 1]
        
        print(f"\n您已選擇分析車牌：{target_plate_input}")
        
        # --- 執行主分析流程 ---
        print("\n--- 開始進行分析 ---")
        run_llm_reporting_flow(full_data, target_plate_input, debug_mode=DEBUG_MODE)

    except FileNotFoundError:
        print(f"錯誤：找不到檔案 {DATA_FILE_PATH}。請確認檔案路徑。")
    except Exception as e:
        print(f"執行過程中發生未預期的錯誤：{e}")