# security/anonymizer.py (最終完整版)
import pandas as pd
def format_summary_for_prompt(summary: dict, area_map: dict) -> str:
    """
    將結構化的 summary 轉換為一個對 LLM 更友善、資訊更豐富的純文字格式。
    """
    id_to_name = area_map
    prompt_text = "請根據以下車輛活動分析摘要，生成一份專業的情報分析報告。\n\n"
    prompt_text += "--- 分析摘要 ---\n"

    # --- 主要據點資訊 ---
    prompt_text += "\n[主要據點資訊]\n"
    if summary['base_info']['primary']:
        hb = summary['base_info']['primary']
        prompt_text += f"- 主要基地: {hb['area_id']} ({hb['name']})\n"
        prompt_text += f"  - 長時停留次數: {hb.get('long_stay_count', hb['visit_count'])} 次\n"
        arrival_time = hb.get('avg_arrival_time')
        departure_time = hb.get('avg_departure_time')
        if arrival_time and departure_time:
            prompt_text += f"  - 平均停留時段: {arrival_time} ~ {departure_time}\n"
        else:
            prompt_text += f"  - 停留時長: {hb['total_duration_hours']} 小時\n"
    else:
        prompt_text += "- 無法確定主要基地。\n"
        
    if summary['base_info']['secondary']:
        for i, sb in enumerate(summary['base_info']['secondary']):
            prompt_text += f"- 次要基地 {i+1}: {sb['area_id']} ({sb['name']})\n"
            prompt_text += f"  - 長時停留次數: {sb.get('long_stay_count', sb['visit_count'])} 次\n"
            arrival_time = sb.get('avg_arrival_time')
            departure_time = sb.get('avg_departure_time')
            if arrival_time and departure_time:
                prompt_text += f"  - 平均停留時段: {arrival_time} ~ {departure_time}\n"
            else:
                prompt_text += f"  - 停留時長: {sb['total_duration_hours']} 小時\n"
    
    # --- 所有停留點統計 ---
    prompt_text += "\n[所有停留點統計]\n"
    for sp in summary['all_stay_points_stats']:
        prompt_text += f"- {sp['area_id']} ({sp['name']}): "
        prompt_text += f"來訪 {sp['visit_count']} 次, 總計 {sp['total_duration_hours']} 小時"
        if sp.get('avg_arrival_time'):
             prompt_text += f", 通常時段 {sp['avg_arrival_time']} ~ {sp['avg_departure_time']}\n"
        else:
             prompt_text += "\n"

    # --- 已確認的規律模式 ---
    prompt_text += "\n[已確認的規律模式]\n"
    if summary['regular_patterns']:
        for i, p in enumerate(summary['regular_patterns']):
            start_name = id_to_name.get(p['start_area_id'], "未知")
            end_name = id_to_name.get(p['end_area_id'], "未知")
            prompt_text += f"- 模式 {chr(65+i)} (從 {start_name} 到 {end_name}, {p['day_type']}-{p['time_slot']}): 發生 {p['occurrence_count']} 次, 平均時段 {p['avg_start_time']}~{p['avg_end_time']}, 平均耗時 {p['avg_duration_minutes']:.2f} 分鐘\n"
    else:
        prompt_text += "- 無\n"


    # --- 路徑異常事件 ---
    prompt_text += "\n[路徑異常事件 (次數較少的行程)]\n"
    if summary['infrequent_patterns']:
        # Convert to DataFrame to easily group by signature
        infrequent_df = pd.DataFrame(summary['infrequent_patterns'])
        
        # Group by the signature and iterate through each group
        for signature, group in infrequent_df.groupby('signature'):
            start_name = id_to_name.get(group['start_area_id'].iloc[0], "未知")
            end_name = id_to_name.get(group['end_area_id'].iloc[0], "未知")
            count = len(group)
            
            # Print a header for the group
            prompt_text += f"- 模式「從 {start_name} 到 {end_name}」 (共 {count} 次):\n"
            
            # List each specific occurrence time for that group
            for _, row in group.iterrows():
                prompt_text += f"  - {row['start_time'].strftime('%Y-%m-%d %H:%M')} 到 {row['end_time'].strftime('%Y-%m-%d %H:%M')}\n"
    else:
        prompt_text += "- 無\n"
        
    # --- 時間異常事件 ---
    prompt_text += "\n[時間異常事件]\n"
    if summary['duration_anomalies']:
        for a in summary['duration_anomalies']:
            pattern_index = next((i for i, p in enumerate(summary['regular_patterns']) if p['signature'] == a['pattern_signature']), -1)
            pattern_label = f"模式 {chr(65+pattern_index)}" if pattern_index != -1 else "一個規律模式"
            
            p_info = next((p for p in summary['regular_patterns'] if p['signature'] == a['pattern_signature']), None)
            start_name = id_to_name.get(p_info['start_area_id'], "未知") if p_info else "未知"
            end_name = id_to_name.get(p_info['end_area_id'], "未知") if p_info else "未知"

            exceeded_time = round(a['actual_duration_minutes'] - a['median_duration_for_pattern'], 2)
            prompt_text += f"- {a['start_time'].strftime('%Y-%m-%d %H:%M')} 到 {a['end_time'].strftime('%Y-%m-%d %H:%M')}, 在「{pattern_label} (從 {start_name} 到 {end_name})」路徑上, "
            prompt_text += f"耗時 {a['actual_duration_minutes']} 分鐘, 超出中位數時間 {exceeded_time} 分鐘 (中位數: {a['median_duration_for_pattern']} 分鐘)。\n"
    else:
        prompt_text += "- 無\n"
        
    prompt_text += "\n--- 摘要結束 ---\n"
    return prompt_text


def anonymize_data(summary: dict, area_map: dict, plate_number: str):
    """
    將分析摘要和映射表去識別化，生成安全的 Prompt 和還原用的映射表。
    """
    reversal_map = {}
    
    plate_code = "目標車輛A"
    reversal_map[plate_code] = plate_number
    
    name_to_code_map = {}
    # 根據地點在 all_stay_points_stats 中的排序來生成 A, B, C... 代碼
    if 'all_stay_points_stats' in summary:
        for i, sp in enumerate(summary['all_stay_points_stats']):
            name = sp['name']
            if name not in name_to_code_map:
                location_code = f"地點-{chr(65+i)}"
                name_to_code_map[name] = location_code
                reversal_map[location_code] = name
    
    # 為 area_map 中有，但不在 all_stay_points_stats 的地點提供備用代碼
    for area_id, name in area_map.items():
        if name not in name_to_code_map:
            location_code = f"地點-{area_id.replace('Area-','')}"
            name_to_code_map[name] = location_code
            reversal_map[location_code] = name

    full_prompt_text = format_summary_for_prompt(summary, area_map)

    anonymized_prompt_text = full_prompt_text
    
    # 優先替換較長的名稱，避免部分匹配錯誤
    sorted_names = sorted(name_to_code_map.keys(), key=len, reverse=True)
    
    for name in sorted_names:
        code = name_to_code_map[name]
        anonymized_prompt_text = anonymized_prompt_text.replace(name, code)
    
    # 最後替換車牌
    anonymized_prompt_text = anonymized_prompt_text.replace(plate_number, plate_code)
        
    return anonymized_prompt_text, reversal_map