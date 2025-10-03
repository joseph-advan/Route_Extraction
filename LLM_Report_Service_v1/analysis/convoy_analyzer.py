# analysis/convoy_analyzer.py (V7 - 排序優化版)
import pandas as pd
import numpy as np
from itertools import groupby

# 從現有的模組中，匯入我們需要的行程切分工具
from .trip_segmenter import segment_trips_v3

# --- 核心演算法函式 ---

def _find_continuous_segments(events_df: pd.DataFrame, max_gap_minutes: int = 10) -> list:
    """從一系列共現事件中，找出所有連續的同行片段。"""
    if events_df.empty:
        return []
    
    segments = []
    # 僅根據目標車的時間差來切分路段
    events_df['time_gap'] = events_df['datetime_x'].diff() > pd.Timedelta(minutes=max_gap_minutes)
    group_ids = events_df['time_gap'].cumsum()
    
    for _, group in events_df.groupby(group_ids):
        segments.append(group.drop(columns=['time_gap']))
    return segments

def _get_following_pattern(target_trip_df: pd.DataFrame, convoy_segment_df: pd.DataFrame) -> str:
    """根據同行路段在完整行程中的位置，產生跟隨模式標籤。"""
    target_len = len(target_trip_df)
    if target_len == 0: return "模式未知"
    convoy_len = len(convoy_segment_df)
    
    if convoy_len / target_len > 0.9:
        return "全程跟隨"

    first_convoy_point_time = convoy_segment_df.iloc[0]['datetime_x']
    last_convoy_point_time = convoy_segment_df.iloc[-1]['datetime_x']

    start_indices = target_trip_df.index[target_trip_df['datetime'] == first_convoy_point_time]
    if len(start_indices) == 0: return "模式未知"
    start_idx = start_indices[0]

    end_indices = target_trip_df.index[target_trip_df['datetime'] == last_convoy_point_time]
    if len(end_indices) == 0: return "模式未知"
    end_idx = end_indices[0]

    is_start = start_idx < target_len / 3
    is_end = end_idx > target_len * 2 / 3

    if is_start and not is_end:
        return "開頭到中間"
    if not is_start and is_end:
        return "中間到結尾"
    if not is_start and not is_end:
        return "中間到中間"
    return "全程跟隨"

# --- 主流程函式 ---

def run_trip_oriented_convoy_analysis(full_data: pd.DataFrame):
    """執行「目標行程導向的隨行分析」的主函式。"""
    
    available_plates = sorted(full_data['車牌'].unique())
    print("\n" + "="*50); print("== 目標行程導向隨行分析 =="); print("="*50)
    for i, plate in enumerate(available_plates): print(f"  [{i+1}] {plate}")
    
    try:
        choice_input = input(f"\n請選擇要作為基準的目標車牌 [1-{len(available_plates)}]: ")
        target_plate = available_plates[int(choice_input) - 1]
    except (ValueError, IndexError):
        print("錯誤：無效的選擇，返回主菜單。")
        return

    print("\n--- 正在分析目標車輛的所有行程並尋找同行者... ---")
    
    target_df = full_data[full_data['車牌'] == target_plate].copy()
    
    if 'LocationAreaID' not in target_df.columns:
         target_df['LocationAreaID'] = target_df['LocationID']
    
    all_target_trips = segment_trips_v3(target_df, gap_threshold_minutes=20)
    
    if not all_target_trips:
        print(f"錯誤：無法為車輛 {target_plate} 切分出任何有效行程。")
        return

    analyzed_trips = []
    cam_name_map = full_data.drop_duplicates(subset=['LocationID']).set_index('LocationID')['攝影機名稱'].to_dict()

    for trip_index, trip_info in enumerate(all_target_trips):
        trip_start_time = trip_info['start_time']
        trip_end_time = trip_info['end_time']
        
        target_trip_df = target_df[
            (target_df['datetime'] >= trip_start_time) & (target_df['datetime'] <= trip_end_time)
        ].sort_values('datetime').reset_index(drop=True)

        convoy_partners_found = []
        max_convoy_length_in_trip = 0

        for partner_plate in available_plates:
            if partner_plate == target_plate:
                continue
            
            partner_df = full_data[full_data['車牌'] == partner_plate].copy()
            partner_df_sorted = partner_df.sort_values('datetime')
            
            co_occurrence_events_list = []
            time_tolerance = pd.Timedelta(minutes=1)

            for _, target_row in target_trip_df.iterrows():
                target_time = target_row['datetime']
                target_loc = target_row['LocationID']
                
                time_min = target_time - time_tolerance
                time_max = target_time + time_tolerance
                
                possible_matches = partner_df_sorted[
                    (partner_df_sorted['LocationID'] == target_loc) &
                    (partner_df_sorted['datetime'].between(time_min, time_max))
                ]
                
                if not possible_matches.empty:
                    best_match = possible_matches.loc[
                        (possible_matches['datetime'] - target_time).abs().idxmin()
                    ]
                    
                    event = {
                        'datetime_x': target_time,
                        'datetime_y': best_match['datetime'],
                        'LocationID': target_loc,
                    }
                    co_occurrence_events_list.append(event)

            if not co_occurrence_events_list:
                continue

            co_occurrence_events_df = pd.DataFrame(co_occurrence_events_list)
            
            continuous_segments = _find_continuous_segments(co_occurrence_events_df)
            
            for segment_df in continuous_segments:
                if len(segment_df) >= 20:
                    partner_info = {
                        'plate': partner_plate,
                        'segment_length': len(segment_df),
                        'start_time': segment_df.iloc[0]['datetime_y'],
                        'end_time': segment_df.iloc[-1]['datetime_y'],
                        'start_loc_id': segment_df.iloc[0]['LocationID'],
                        'end_loc_id': segment_df.iloc[-1]['LocationID'],
                        'time_lags': (segment_df['datetime_y'] - segment_df['datetime_x']).dt.total_seconds().tolist(),
                        'convoy_segment_df': segment_df
                    }
                    convoy_partners_found.append(partner_info)
                    
                    if len(segment_df) > max_convoy_length_in_trip:
                        max_convoy_length_in_trip = len(segment_df)
        
        if convoy_partners_found:
            analyzed_trips.append({
                'trip_info': trip_info,
                'target_trip_df': target_trip_df,
                'convoy_partners': convoy_partners_found,
                'max_convoy_length': max_convoy_length_in_trip
            })

    if not analyzed_trips:
        print(f"\n分析完成：未找到車輛 {target_plate} 有任何被跟隨超過 20 個地點的行程。")
        return

    sorted_trips = sorted(analyzed_trips, key=lambda x: x['max_convoy_length'], reverse=True)
    
    print(f"\n--- {target_plate} 被跟隨路段最長的前 {min(3, len(sorted_trips))} 大行程報告 ---")

    for i, trip_data in enumerate(sorted_trips[:3]):
        target_info = trip_data['trip_info']
        target_df = trip_data['target_trip_df']
        partners = trip_data['convoy_partners']
        
        start_loc_name = cam_name_map.get(target_info['start_area_id'], "未知")
        end_loc_name = cam_name_map.get(target_info['end_area_id'], "未知")

        print("\n" + "="*70)
        print(f"## 報告 {i+1}: 目標車行程 (共 {len(target_df)} 個地點)")
        print("="*70)
        print(f"  - 行程日期: {target_info['start_time'].strftime('%Y-%m-%d')}")
        print(f"  - 行程時間: {target_info['start_time'].strftime('%H:%M')} -> {target_info['end_time'].strftime('%H:%M')} (耗時 {target_info['duration_minutes']:.1f} 分鐘)")
        print(f"  - 起點: {start_loc_name} ({target_info['start_area_id']})")
        print(f"  - 終點: {end_loc_name} ({target_info['end_area_id']})")
        print(f"  - 行程路徑: {' -> '.join(target_df['LocationID'].tolist())}")

        print("\n  --- 同行車資訊 ---")
        
        # 【【【 核心修正處：在列印前，先根據同行比例對夥伴車輛進行排序 】】】
        # 使用 lambda 函式來計算每個夥伴的同行比例作為排序的 key
        # reverse=True 確保比例越高的排在越前面
        sorted_partners = sorted(
            partners,
            key=lambda p: p['segment_length'] / len(target_df) if len(target_df) > 0 else 0,
            reverse=True
        )

        # 使用排序後的 sorted_partners 列表來產生報告
        for j, partner_info in enumerate(sorted_partners):
            p_start_loc_name = cam_name_map.get(partner_info['start_loc_id'], "未知")
            p_end_loc_name = cam_name_map.get(partner_info['end_loc_id'], "未知")
            avg_lag = np.mean(partner_info['time_lags'])
            lag_str = f"晚 {avg_lag:.1f} 秒" if avg_lag > 0 else f"早 {abs(avg_lag):.1f} 秒"
            
            pattern_tag = _get_following_pattern(target_df, partner_info['convoy_segment_df'])

            print(f"\n    [同行車 #{j+1}]")
            print(f"    - 車牌: {partner_info['plate']}")
            
            convoy_points = partner_info['segment_length']
            total_points = len(target_df)
            ratio_percentage = (convoy_points / total_points * 100) if total_points > 0 else 0
            print(f"    - 同行比例: {ratio_percentage:.1f}% ({convoy_points}/{total_points} 個地點)")
            
            print(f"    - 平均時間差: {lag_str}")
            print(f"    - 跟隨模式標籤: {pattern_tag}")
            print(f"    - 同行路段: {' -> '.join(partner_info['convoy_segment_df']['LocationID'].tolist())}")
            print(f"    - 同行時間: {partner_info['start_time'].strftime('%H:%M:%S')} -> {partner_info['end_time'].strftime('%H:%M:%S')} (耗時 {(partner_info['end_time'] - partner_info['start_time']).total_seconds() / 60:.1f} 分鐘)")
            print(f"    - 同行起點: {p_start_loc_name} ({partner_info['start_loc_id']})")
            print(f"    - 同行終點: {p_end_loc_name} ({partner_info['end_loc_id']})")

