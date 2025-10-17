# security/anonymizer.py (修正後)
import pandas as pd

def format_summary_for_prompt(summary: dict, area_map: dict) -> str:
    """
    將結構化的 summary 轉換為一個對 LLM 更友善、資訊更豐富的純文字格式。
    """
    id_to_name = area_map
    prompt_text = "請根據以下車輛活動分析摘要，生成一份專業的情報分析報告。\n\n"
    prompt_text += "--- 分析摘要 ---\n"

    # --- 主要據點資訊 ---
    prompt_text += "\n[主要活動/停留點分析]\n"
# 【簡化】不再需要區分 primary/secondary，因為標籤1, 2, 3已經隱含了重要性
    if not summary['all_stay_points_stats']:
        prompt_text += "- 未發現明顯的長時間停留點。\n"
    else:
        for sp in summary['all_stay_points_stats']:
            prompt_text += f"- {sp['name']} ({sp['area_id']}): " # (這裡的 sp['name'] 會被後續步驟替換掉)
            prompt_text += f"來訪 {sp['visit_count']} 次, 總計 {sp['total_duration_hours']} 小時"
            
            if sp.get('stay_pattern_type') == '長期駐留':
                prompt_text += f", 模式：長期駐留 (平均每次停留約 {sp['avg_duration_days']} 天)\n"
            elif sp.get('avg_arrival_time'):
                 prompt_text += f", 通常時段 {sp['avg_arrival_time']} ~ {sp['avg_departure_time']}\n"
            else:
                 prompt_text += "\n"
    

    # --- 後續的程式碼 (規律模式、異常事件) 維持不變 ---
    # (此處省略後續未變動的程式碼，你的版本是正確的)
    # ...
    prompt_text += "\n[已確認的規律模式]\n"
    if summary['regular_patterns']:
        for i, p in enumerate(summary['regular_patterns']):
            start_name = id_to_name.get(p['start_area_id'], "未知")
            end_name = id_to_name.get(p['end_area_id'], "未知")
            prompt_text += f"- 模式 {chr(65+i)} (從 {start_name} 到 {end_name}, {p['day_type']}-{p['time_slot']}): 發生 {p['occurrence_count']} 次, 平均時段 {p['avg_start_time']}~{p['avg_end_time']}, 平均耗時 {p['avg_duration_minutes']:.2f} 分鐘\n"
    else:
        prompt_text += "- 無\n"


    prompt_text += "\n[路徑異常事件 (次數較少的行程)]\n"
    if summary['infrequent_patterns']:
        infrequent_df = pd.DataFrame(summary['infrequent_patterns'])
        for signature, group in infrequent_df.groupby('signature'):
            start_name = id_to_name.get(group['start_area_id'].iloc[0], "未知")
            end_name = id_to_name.get(group['end_area_id'].iloc[0], "未知")
            count = len(group)
            avg_duration = group['duration_minutes'].mean()
            prompt_text += f"- 模式「從 {start_name} 到 {end_name}」 (共 {count} 次, 平均耗時 {avg_duration:.1f} 分鐘):\n"
            for _, row in group.iterrows():
                prompt_text += f"  - {row['start_time'].strftime('%Y-%m-%d %H:%M')} 到 {row['end_time'].strftime('%Y-%m-%d %H:%M')}\n"
    else:
        prompt_text += "- 無\n"
        
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

# security/anonymizer.py

# ... (format_summary_for_prompt 函式先不用動) ...

# security/anonymizer.py (新架構版)
import pandas as pd

# format_summary_for_prompt 函式維持不變，因為我們希望原始資料盡可能完整
# (此處省略該函式，請保留您檔案中原有的版本)
# ...

def anonymize_data(summary: dict, area_map: dict, plate_number: str):
    """
    (新架構版) 將分析摘要去識別化。
    - reversal_map 現在儲存更豐富的資訊：{"Area-ID": {"name": "...", "label": "..."}}
    - Prompt 中的地點名稱會被統一替換成 Area-ID。
    """
    reversal_map = {}
    
    # 步驟 1: 建立基礎的 reversal_map，包含所有地點的名稱和預設空標籤
    for area_id, name in area_map.items():
        reversal_map[area_id] = {"name": name, "label": None}
        
    # 步驟 2: 遍歷排序後的主要停留點，為它們在 map 中“貼上”標籤
    label_counter = 1
    if 'all_stay_points_stats' in summary:
        for sp in summary['all_stay_points_stats']:
            area_id = sp['area_id']
            if area_id in reversal_map:
                reversal_map[area_id]["label"] = f"主要活動/停留點{label_counter}"
                label_counter += 1

    # 處理車牌
    reversal_map["目標車輛A"] = {"name": plate_number, "label": "目標車輛"}

    # 步驟 3: 準備發送給 LLM 的文本，將所有地點全名替換成 Area-ID
    full_prompt_text = format_summary_for_prompt(summary, area_map)
    anonymized_prompt_text = full_prompt_text

    # 優先替換較長的攝影機名稱，避免部分匹配錯誤
    sorted_names = sorted(area_map.values(), key=len, reverse=True)
    for name in sorted_names:
        # 找到這個名稱對應的 Area-ID
        area_id = next((aid for aid, n in area_map.items() if n == name), None)
        if area_id:
            anonymized_prompt_text = anonymized_prompt_text.replace(name, area_id)

    # 替換車牌
    anonymized_prompt_text = anonymized_prompt_text.replace(plate_number, "目標車輛A")
        
    return anonymized_prompt_text, reversal_map