# reporting_service.py (最終版 - Markdown 格式化輸出)

import pandas as pd
import json

# 匯入所有需要的分析與安全模組
from analysis.camera_clusterer import cluster_cameras_by_distance
from analysis.stay_point_detector import find_stay_points_v2
from analysis.trip_segmenter import segment_trips_v3
from analysis.pattern_clusterer import find_regular_patterns_v13
from analysis.anomaly_detector import find_anomalies_v3
from security.anonymizer import anonymize_data
from security.deanonymizer import deanonymize_report
from llm_clients.cloud_client import generate_report_from_summary

# ==============================================================================
# 【【【 修改處: 將輸出格式改為 Markdown 】】】
# ==============================================================================
def format_details_to_string(summary_data: dict, area_map: dict) -> str:
    """將分析完成的 summary dictionary 轉換為人類可讀的 Markdown 格式化字串。"""
    output = []

    # 1. 主要據點資訊
    if summary_data.get("base_info"):
        output.append("### [主要據點資訊]\n")
        primary = summary_data["base_info"].get("primary")
        if primary:
            output.append(f"- **主要基地**: {primary.get('area_id', 'N/A')} ({primary.get('name', 'N/A')})")
            output.append(f"  - **長時停留次數**: {primary.get('long_stay_count', 'N/A')} 次")
            if primary.get('avg_arrival_time'):
                 output.append(f"  - **平均停留時段**: {primary.get('avg_arrival_time', 'N/A')} ~ {primary.get('avg_departure_time', 'N/A')}")
        
        secondary = summary_data["base_info"].get("secondary", [])
        for i, base in enumerate(secondary):
            output.append(f"- **次要基地 {i+1}**: {base.get('area_id', 'N/A')} ({base.get('name', 'N/A')})")
            output.append(f"  - **長時停留次數**: {base.get('long_stay_count', 'N/A')} 次")
            if base.get('stay_pattern_type') == '長期駐留':
                output.append(f"  - **模式**：{base.get('stay_pattern_type', 'N/A')} (平均每次停留約 {base.get('avg_duration_days', 'N/A')} 天)")
        output.append("\n---\n")

    # 2. 所有停留點統計
    if summary_data.get("all_stay_points_stats"):
        output.append("### [所有停留點統計]\n")
        for sp in summary_data["all_stay_points_stats"]:
            line = f"- **{sp.get('area_id', 'N/A')} ({sp.get('name', 'N/A')})**: 來訪 {sp.get('visit_count', 'N/A')} 次, 總計 {sp.get('total_duration_hours', 'N/A')} 小時"
            if sp.get('stay_pattern_type') == '長期駐留':
                line += f", **模式**：{sp.get('stay_pattern_type', 'N/A')} (平均每次停留約 {sp.get('avg_duration_days', 'N/A')} 天)"
            elif sp.get('avg_arrival_time'):
                 line += f", **通常時段**: {sp.get('avg_arrival_time', 'N/A')} ~ {sp.get('avg_departure_time', 'N/A')}"
            output.append(line)
        output.append("\n---\n")

    # 3. 已確認的規律模式
    if summary_data.get("regular_patterns"):
        output.append("### [已確認的規律模式]\n")
        pattern_labels = [f"模式 {chr(65 + i)}" for i in range(len(summary_data["regular_patterns"]))]
        for label, p in zip(pattern_labels, summary_data["regular_patterns"]):
            start_name = area_map.get(p.get('start_area_id'), "未知地點")
            end_name = area_map.get(p.get('end_area_id'), "未知地點")
            output.append(f"- **{label}** (從 *{start_name}* 到 *{end_name}*, {p.get('day_type', 'N/A')}-{p.get('time_slot', 'N/A')}): "
                          f"發生 **{p.get('occurrence_count', 'N/A')} 次**, "
                          f"平均時段 `{p.get('avg_start_time', 'N/A')}~{p.get('avg_end_time', 'N/A')}`, "
                          f"平均耗時 **{p.get('avg_duration_minutes', 'N/A')} 分鐘**")
        output.append("\n---\n")

    # 4. 路徑異常事件
    output.append("### [路徑異常事件 (次數較少的行程)]\n")
    infrequent = summary_data.get("infrequent_patterns", [])
    if not infrequent:
        output.append("- 無\n")
    else:
        for anomaly in infrequent:
            start_name = area_map.get(anomaly.get('start_area_id'), "未知地點")
            end_name = area_map.get(anomaly.get('end_area_id'), "未知地點")
            start_time_str = anomaly.get('start_time', pd.Timestamp.min).strftime('%Y-%m-%d %H:%M')
            output.append(f"- `{start_time_str}` 從 *{start_name}* 到 *{end_name}*, "
                          f"耗時 **{anomaly.get('duration_minutes', 'N/A')} 分鐘**")
    output.append("\n---\n")

    # 5. 時間異常事件
    output.append("### [時間異常事件]\n")
    duration_anomalies = summary_data.get("duration_anomalies", [])
    if not duration_anomalies:
        output.append("- 無\n")
    else:
        for anomaly in duration_anomalies:
            actual_duration = anomaly.get('actual_duration_minutes', 0)
            median_duration = anomaly.get('median_duration_for_pattern', 0)
            diff = round(actual_duration - median_duration, 2)
            start_time_str = anomaly.get('start_time', pd.Timestamp.min).strftime('%Y-%m-%d %H:%M')
            end_time_str = anomaly.get('end_time', pd.Timestamp.min).strftime('%Y-%m-%d %H:%M')
            output.append(f"- `{start_time_str}` 到 `{end_time_str}`, "
                          f"在「*{anomaly.get('pattern_signature', 'N/A')}*」路徑上, "
                          f"耗時 **{actual_duration} 分鐘**, "
                          f"超出中位數時間 **{diff} 分鐘** (中位數: {median_duration} 分鐘)。")

    return "\n".join(output)

# ==============================================================================
# 【【【 主要流程函式 (維持不變) 】】】
# ==============================================================================
def run_llm_reporting_flow(full_df: pd.DataFrame, target_plate: str, debug_mode: bool = False):
    """
    執行從數據分析到 LLM 報告生成的完整流程。
    """
    print("\n--- 正在執行本地數據分析引擎... ---")
    
    unique_cameras = full_df[['攝影機', '攝影機名稱', '經度', '緯度', '單位']].drop_duplicates(subset=['攝影機']).reset_index(drop=True)
    cameras_with_area_id = cluster_cameras_by_distance(unique_cameras, radius_meters=200)
    
    vehicle_data = full_df[full_df['車牌'] == target_plate].copy()
    if vehicle_data.empty:
        print(f"錯誤：在資料集中找不到車牌 {target_plate} 的任何紀錄。")
        return

    vehicle_data_with_area = pd.merge(vehicle_data, cameras_with_area_id[['攝影機', 'LocationAreaID']], on='攝影機', how='left')
    
    stay_points_result = find_stay_points_v2(vehicle_data_with_area, time_threshold_minutes=20)
    if not stay_points_result:
        print(f"- 未找到 {target_plate} 的任何停留點，分析中止。")
        return
    
    trips_result = segment_trips_v3(vehicle_data_with_area)
    if not trips_result:
        print(f"- 未切割出 {target_plate} 的任何行程，分析中止。")
        return
    
    pattern_result = find_regular_patterns_v13(trips_result, stay_points_result, cameras_with_area_id)
    regular_summary = pattern_result["summary"]
    area_map = pattern_result["area_map"]
    trips_df = pattern_result["trips_df"]
    
    anomalies = find_anomalies_v3(trips_df, regular_summary["regular_patterns"])
    
    final_summary = {**regular_summary, **anomalies}
    print("--- 本地數據分析完成 ---")

    if debug_mode:
        print("\n" + "#"*70)
        print("## DEBUG: Area ID -> 地點名稱映射表 (Area Map)")
        print("#"*70)
        print(json.dumps(area_map, indent=2, ensure_ascii=False))
        
        print("\n" + "#"*70)
        print("## DEBUG: 完整分析結果原始 JSON (Final Summary)")
        print("#"*70)
        print(json.dumps(final_summary, indent=2, ensure_ascii=False, default=str))

    anonymized_prompt, reversal_map = anonymize_data(final_summary, area_map, target_plate)
    
    print("\n--- 正在呼叫雲端 LLM 生成智慧摘要... ---")
    draft_summary = generate_report_from_summary(anonymized_prompt)
    
    final_summary_text = deanonymize_report(draft_summary, reversal_map)

    details_str = format_details_to_string(final_summary, area_map)

    # --- 最終輸出 ---
    print("\n\n" + "#"*70)
    print("## 最終分析報告")
    print("#"*70)

    print("\n【 智慧摘要 】\n")
    print(final_summary_text)

    print("\n" + "="*70 + "\n")
    print("【 詳細數據 】\n")
    print(details_str)