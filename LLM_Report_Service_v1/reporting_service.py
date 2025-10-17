# reporting_service.py (修正版 - 正確處理 OpenAI 回應物件)

import pandas as pd
import json
import re

# (上方的 import 和 format_details_to_string 函式維持不變)
from analysis.camera_clusterer import cluster_cameras_by_distance
from analysis.stay_point_detector import find_stay_points_v2
from analysis.trip_segmenter import segment_trips_v3
from analysis.pattern_clusterer import find_regular_patterns_v13
from analysis.anomaly_detector import find_anomalies_v3
from security.anonymizer import anonymize_data
from security.deanonymizer import deanonymize_report
from llm_clients.cloud_client import generate_report_from_summary

def format_details_to_string(summary_data: dict, area_map: dict) -> str:
    """
    (最終架構版) 將分析完成的 summary dictionary 轉換成人類可讀的 Markdown 格式化字串。
    - 為「單次停留」的點，顯示其具體的開始與結束時間。
    """
    output = []

    # --- 區塊 1: 停留點統計 ---
    output.append("### [停留點統計 (依總停留時間排序)]\n")
    stay_points = summary_data.get("all_stay_points_stats", [])
    if stay_points:
        for i, sp in enumerate(stay_points):
            label = f" (主要活動/停留點 {i+1})" if i < 3 else ""
            line = (f"- **{sp.get('area_id', 'N/A')}{label}**: "
                    f"來訪 {sp.get('visit_count', 'N/A')} 次, "
                    f"總計 {sp.get('total_duration_hours', 'N/A')} 小時")
            
            if sp.get('stay_pattern_type') == '長期駐留':
                line += f", **模式**：長期駐留 (平均每次停留約 {sp.get('avg_duration_days', 'N/A')} 天)"
            
            # 對於多次停留的點，顯示平均時段
            if sp.get('avg_arrival_time'):
                 line += f", **通常時段**：{sp.get('avg_arrival_time', 'N/A')} ~ {sp.get('avg_departure_time', 'N/A')}"
            
            # 【【核心修改】】對於單次停留的點，顯示精確時段
            if sp.get('stay_pattern_type') == '單次停留' and 'start_time' in sp:
                start_str = sp['start_time'].strftime('%Y-%m-%d %H:%M')
                end_str = sp['end_time'].strftime('%Y-%m-%d %H:%M')
                line += f", **停留時間**：{start_str} ~ {end_str}"

            output.append(line)
    else:
        output.append("- 在此期間未偵測到任何明顯的停留點。\n")
    output.append("\n---\n")

    # ... (函式的其餘部分維持不變，此處省略以保持簡潔) ...
    # --- 區塊 2: 已確認的規律模式 ---
    output.append("### [已確認的規律模式]\n")
    regular_patterns = summary_data.get("regular_patterns", [])
    if regular_patterns:
        pattern_labels = [f"模式 {chr(65 + i)}" for i in range(len(regular_patterns))]
        for label, p in zip(pattern_labels, regular_patterns):
            start_name = area_map.get(p.get('start_area_id'), "未知地點")
            end_name = area_map.get(p.get('end_area_id'), "未知地點")
            output.append(f"- **{label}** (從 *{start_name}* 到 *{end_name}*, {p.get('day_type', 'N/A')}-{p.get('time_slot', 'N/A')}): "
                          f"發生 **{p.get('occurrence_count', 'N/A')} 次**, "
                          f"平均時段 `{p.get('avg_start_time', 'N/A')}~{p.get('avg_end_time', 'N/A')}`, "
                          f"平均耗時 **{p.get('avg_duration_minutes', 'N/A')} 分鐘**")
    else:
        output.append("- 在此期間未發現任何重複出現的規律性行程。\n")
    output.append("\n---\n")

    # --- 區塊 3: 路徑異常事件 ---
    output.append("### [路徑異常事件 (次數較少的行程)]\n")
    infrequent_patterns = summary_data.get("infrequent_patterns", [])
    if infrequent_patterns:
        for anomaly in infrequent_patterns:
            start_name = area_map.get(anomaly.get('start_area_id'), "未知地點")
            end_name = area_map.get(anomaly.get('end_area_id'), "未知地點")
            start_time_str = anomaly.get('start_time', pd.Timestamp.min).strftime('%Y-%m-%d %H:%M')
            output.append(f"- `{start_time_str}` 從 *{start_name}* 到 *{end_name}*, "
                          f"耗時 **{anomaly.get('duration_minutes', 'N/A')} 分鐘**")
    else:
        output.append("- 在此期間未發現任何偶發性之路徑。\n")
    output.append("\n---\n")

    # --- 區塊 4: 時間異常事件 ---
    output.append("### [時間異常事件]\n")
    duration_anomalies = summary_data.get("duration_anomalies", [])
    if not duration_anomalies:
        output.append("- 在此期間，車輛於規律路徑上的行駛時間皆在正常範圍內，未發現時間異常。\n")
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

def run_llm_reporting_flow(full_df: pd.DataFrame, target_plate: str, debug_mode: bool = False):

    
    # ==============================================================================
    # 步驟 1: 執行本地數據分析引擎 (此區塊不變)
    # ==============================================================================
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
    # ... (debug 模式程式碼不變) ...

    # ==============================================================================
    # 步驟 2: 去識別化並呼叫 LLM (此區塊不變)
    # ==============================================================================
    anonymized_prompt, reversal_map = anonymize_data(final_summary, area_map, target_plate)
    
    print("\n--- 正在呼叫雲端 LLM 生成智慧摘要... ---")
    summary_from_llm = generate_report_from_summary(anonymized_prompt)
    
    # ==============================================================================
    # 步驟 3: 組合並輸出最終報告
    # ==============================================================================
    
    # 【【【【 核心修正處：將 details_str 的定義加回這裡 】】】】
    details_str = format_details_to_string(final_summary, area_map)
    
    print("\n\n" + "#"*70)
    print("## 最終分析報告")
    print("#"*70)

    print("\n【 智慧摘要 】\n")
    print(summary_from_llm)

    print("\n" + "-"*35)
    print("  地點說明:")
    
    mentioned_areas = sorted(list(set(re.findall(r'Area-\d+', summary_from_llm))))
            
    if mentioned_areas:
        main_points = []
        other_points = []
        
        for area_id in mentioned_areas:
            if area_id in reversal_map:
                info = reversal_map[area_id]
                if info.get("label"):
                    main_points.append((area_id, info))
                else:
                    other_points.append((area_id, info))
        
        main_points.sort(key=lambda item: int(item[1]['label'].replace("主要活動/停留點", "")))

        for area_id, info in main_points:
            print(f'  * {area_id} ({info["label"]}): {info["name"]}')

        for area_id, info in other_points:
            print(f'  * {area_id}: {info["name"]}')
    else:
        print("  - 摘要中未提及具體地點。")
        
    print("-" * 35)
    
    print("\n" + "="*70 + "\n")
    print("【 詳細數據 】\n")
    
    final_details_str = details_str
    sorted_real_names = sorted(area_map.values(), key=len, reverse=True)
    for name in sorted_real_names:
        area_id = next((aid for aid, n in area_map.items() if n == name), None)
        if area_id:
            final_details_str = final_details_str.replace(name, area_id)
            
    print(final_details_str)