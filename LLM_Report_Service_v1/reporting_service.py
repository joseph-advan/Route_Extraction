# reporting_service.py (Corrected Version)

import pandas as pd
import json

# Import all the latest versions of our analysis modules
from analysis.camera_clusterer import cluster_cameras_by_distance
from analysis.stay_point_detector import find_stay_points_v2
from analysis.trip_segmenter import segment_trips_v3
from analysis.pattern_clusterer import find_regular_patterns_v13 # Using the correct v15
from analysis.anomaly_detector import find_anomalies_v3
from security.anonymizer import anonymize_data
from security.deanonymizer import deanonymize_report
from llm_clients.cloud_client import generate_report_from_summary

def run_llm_reporting_flow(full_df: pd.DataFrame, target_plate: str, debug_mode: bool = False):
    """
    Executes the complete analysis and report generation workflow for a single vehicle.
    """
    if debug_mode:
        print("\n" + "="*70); print("## Step 1-4: Executing Local Data Analysis Engine..."); print("="*70)
    
    unique_cameras = full_df[['攝影機', '攝影機名稱', '經度', '緯度', '單位']].drop_duplicates(subset=['攝影機']).reset_index(drop=True)
    cameras_with_area_id = cluster_cameras_by_distance(unique_cameras, radius_meters=200)
    if debug_mode: print(f"- Geographic clustering complete. Found {cameras_with_area_id['LocationAreaID'].nunique()} unique location areas.")

    vehicle_data = full_df[full_df['車牌'] == target_plate].copy()
    if vehicle_data.empty:
        print(f"Error: No records found for license plate {target_plate} in the dataset."); return

    vehicle_data_with_area = pd.merge(vehicle_data, cameras_with_area_id[['攝影機', 'LocationAreaID']], on='攝影機', how='left')
    stay_points_result = find_stay_points_v2(vehicle_data_with_area, time_threshold_minutes=20)


    if not stay_points_result:
        # 產生並印出你指定的詳細說明
        print("\n" + "="*70)
        print("## 分析中止：無法定位主要棲息地")
        print("="*70)
        print(f"原因：無法從現有軌跡中找到任何 {target_plate} 有效的長時停留點 (棲息地)。")

        print("\n建議：")
        print("- 確認查詢的時間範圍是否正確。")
        print("- 嘗試查找或整合更多元的監視器數據來源，以獲得更完整的車輛軌跡。")
        return
    
    if debug_mode: print(f"- Found {len(stay_points_result)} stay point events.")


    
    trips_result = segment_trips_v3(vehicle_data_with_area)
    if not trips_result: print(f"- Could not segment any trips for {target_plate}. Analysis stopped."); return
    if debug_mode: print(f"- Segmented into {len(trips_result)} trips.")
    
    # ========================= CODE CORRECTION =========================
    # Pass the complete camera map (cameras_with_area_id) into the function
    pattern_result = find_regular_patterns_v13(trips_result, stay_points_result, all_cameras_with_area=cameras_with_area_id)
    # ========================= CORRECTION END ==========================
    
    regular_summary = pattern_result["summary"]
    area_map = pattern_result["area_map"]
    trips_df = pattern_result["trips_df"]
    
    anomalies = find_anomalies_v3(trips_df, regular_summary["regular_patterns"])
    
    final_summary = {**regular_summary, **anomalies}
    if debug_mode: print("- Local data analysis complete!")

    if debug_mode:
        print("\n" + "#"*70); print("## Output 1.1: Location Area ID Map (Area Map)"); print("#"*70)
        print(json.dumps(area_map, indent=2, ensure_ascii=False))
        print("\n" + "#"*70); print("## Output 1.2: Complete Structured Summary from Analysis Engine (Final Summary)"); print("#"*70)
        print(json.dumps(final_summary, indent=2, ensure_ascii=False, default=str))

    if debug_mode:
        print("\n" + "="*70); print("## Step 5: Anonymizing the analysis summary..."); print("="*70)
    anonymized_prompt, reversal_map = anonymize_data(final_summary, area_map, target_plate)
    
    if debug_mode:
        print("\n" + "#"*70); print("## Output 2: The full, anonymized prompt text being sent to the LLM"); print("#"*70)
        print(anonymized_prompt)

    if debug_mode:
        print("\n" + "="*70); print("## Step 6: Calling Cloud LLM API to generate draft report..."); print("="*70)
    draft_report = generate_report_from_summary(anonymized_prompt)
    
    if debug_mode:
        print("\n" + "#"*70); print("## Output 3: The draft report with codes returned from the LLM"); print("#"*70)
        print(draft_report)

    if debug_mode:
        print("\n" + "="*70); print("## Step 7: Deanonymizing the draft report..."); print("="*70)
    final_report = deanonymize_report(draft_report, reversal_map)
    if debug_mode: print("Report deanonymization complete!")

    print("\n" + "#"*70); print("## Final Analysis Report"); print("#"*70)
    print(final_report)