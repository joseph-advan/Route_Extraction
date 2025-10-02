# analysis/similarity_analyzer.py (V12.0 - 分析報告版)

import pandas as pd
from collections import Counter, defaultdict
import numpy as np

# --- 核心資料處理函式 (與前版相同) ---
def find_all_co_occurrence_events(df1: pd.DataFrame, df2: pd.DataFrame, time_tolerance_minutes: int = 15) -> pd.DataFrame:
    df1['time_key'] = df1['datetime'].dt.round(f'{time_tolerance_minutes}min')
    df2['time_key'] = df2['datetime'].dt.round(f'{time_tolerance_minutes}min')
    merged_df = pd.merge(df1, df2, on=['LocationID', 'time_key'])
    time_diff = (merged_df['datetime_x'] - merged_df['datetime_y']).abs()
    return merged_df[time_diff <= pd.Timedelta(minutes=time_tolerance_minutes)].copy()

def stitch_events_into_routes(events_df: pd.DataFrame, max_gap_minutes: int = 60) -> list:
    if events_df.empty: return []
    events_df = events_df.sort_values(by='datetime_x').reset_index(drop=True)
    routes, current_route = [], []
    for _, row in events_df.iterrows():
        if not current_route or (row['datetime_x'] - current_route[-1]['datetime_x']) > pd.Timedelta(minutes=max_gap_minutes):
            if len(current_route) > 1: routes.append(current_route)
            current_route = [row.to_dict()]
        else:
            current_route.append(row.to_dict())
    if len(current_route) > 1: routes.append(current_route)
    return routes

# ========================= 全新：報告生成模組 =========================
def analyze_route_summary(instances: list, target_plate: str) -> dict:
    """
    【全新】分析一條頻繁路徑的所有發生實例，並產生摘要。
    """
    target_travel_times = []
    partner_travel_times = []
    start_hours = []

    for instance in instances:
        details = instance['details']
        target_start = details[0]['datetime_x']
        target_end = details[-1]['datetime_x']
        partner_start = details[0]['datetime_y']
        partner_end = details[-1]['datetime_y']
        
        target_travel_times.append((target_end - target_start).total_seconds())
        partner_travel_times.append((partner_end - partner_start).total_seconds())
        start_hours.append(target_start.hour)

    # 分析同行時段是否集中
    hour_std_dev = np.std(start_hours)
    if hour_std_dev > 3: # 如果開始時間的標準差大於3小時，認為時間段分散
        time_period_summary = "同行時段分散 (橫跨早/中/晚)，請參考下方詳細案例。"
    else:
        avg_hour = int(np.mean(start_hours))
        time_period_summary = f"主要集中在 {avg_hour:02d}:00 ~ {avg_hour+1:02d}:00 之間。"

    return {
        'avg_target_time_min': np.mean(target_travel_times) / 60,
        'avg_partner_time_min': np.mean(partner_travel_times) / 60,
        'time_period_summary': time_period_summary
    }

def run_event_driven_analysis(full_data: pd.DataFrame, min_route_len: int = 2):
    """主流程函式：產生詳細的分析報告。"""
    time_tolerance_minutes = 15
    if 'LocationID' not in full_data.columns:
        print("錯誤：資料中缺少 'LocationID' 欄位。"); return
        
    available_plates = sorted(full_data['車牌'].unique())
    print("\n" + "="*50); print("== 事件驅動同行路徑分析報告 =="); print("="*50)
    print(f"** 目前設定：只統計長度 >= {min_route_len} 的同行路徑 **")
    print(f"** 時間容忍度：兩車出現在同地點的時間差在 {time_tolerance_minutes} 分鐘內 **")
    for i, plate in enumerate(available_plates): print(f"  [{i+1}] {plate}")

    try:
        choice_input = input(f"\n請選擇要作為基準的目標車牌 [1-{len(available_plates)}]: ")
        target_plate = available_plates[int(choice_input) - 1]
        target_df = full_data[full_data['車牌'] == target_plate].copy()
        
        print("\n--- 正在掃描所有同行事件並組合路徑... ---")
        all_common_routes = []
        for partner_plate, partner_df in full_data.groupby('車牌'):
            if partner_plate == target_plate: continue
            co_events = find_all_co_occurrence_events(target_df, partner_df, time_tolerance_minutes)
            if not co_events.empty:
                routes = stitch_events_into_routes(co_events)
                for route in routes:
                    if len(route) >= min_route_len:
                        all_common_routes.append({
                            'partner': partner_plate,
                            'route_tuple': tuple(item['LocationID'] for item in route),
                            'details': route
                        })

        if not all_common_routes:
            print(f"\n分析完成：未找到與 {target_plate} 任何長度超過 {min_route_len} 的同行路徑。"); return

        route_counter = Counter(item['route_tuple'] for item in all_common_routes)
        cam_name_map = full_data.groupby('LocationID')['攝影機名稱'].unique().apply(list).to_dict()
        
        print(f"\n--- 與 {target_plate} 的最頻繁同行路徑 Top 3 分析報告 ---")
        
        for i, (route_tuple, count) in enumerate(route_counter.most_common(3)):
            path_str = " -> ".join(route_tuple)
            instances = [item for item in all_common_routes if item['route_tuple'] == route_tuple]
            
            # 產生並印出摘要
            summary = analyze_route_summary(instances, target_plate)
            start_loc_id, end_loc_id = route_tuple[0], route_tuple[-1]
            start_cam_names = cam_name_map.get(start_loc_id, ["未知"])
            end_cam_names = cam_name_map.get(end_loc_id, ["未知"])

            print("\n" + "="*70)
            print(f"## 報告 {i+1}: 最頻繁同行路徑")
            print("="*70)
            print(f"  - 路線 ({len(route_tuple)}個地點): {path_str}")
            print(f"  - 總計發生: {count} 次")
            print(f"  - 起點名稱: {start_cam_names[0]}")
            print(f"  - 終點名稱: {end_cam_names[0]}")
            print("\n  【路線摘要】")
            print(f"  - 同行時段分析: {summary['time_period_summary']}")
            print(f"  - 平均旅行時間: {target_plate} 約 {summary['avg_target_time_min']:.1f} 分鐘 | 夥伴車輛約 {summary['avg_partner_time_min']:.1f} 分鐘")
            
            # 印出詳細案例
            print("\n  【詳細案例列表】")
            for j, instance in enumerate(instances):
                partner = instance['partner']
                details = instance['details']
                
                target_start_time = details[0]['datetime_x']
                target_end_time = details[-1]['datetime_x']
                partner_start_time = details[0]['datetime_y']
                partner_end_time = details[-1]['datetime_y']

                time_diff_seconds = (partner_start_time - target_start_time).total_seconds()
                time_diff_min = time_diff_seconds / 60
                time_corr_str = f"晚 {abs(time_diff_min):.1f} 分鐘" if time_diff_min > 0 else f"早 {abs(time_diff_min):.1f} 分鐘"
                
                print(f"    - [案例 #{j+1}] 與 {partner} 在 {target_start_time.strftime('%Y-%m-%d')}")
                print(f"      - {target_plate.ljust(9)}: {target_start_time.strftime('%H:%M:%S')} -> {target_end_time.strftime('%H:%M:%S')}")
                print(f"      - {partner.ljust(9)}: {partner_start_time.strftime('%H:%M:%S')} -> {partner_end_time.strftime('%H:%M:%S')} (在起點比目標{time_corr_str})")

    except (ValueError, IndexError):
        print("錯誤：無效的選擇，返回主菜單。")
    except Exception as e:
        print(f"分析過程中發生未預期的錯誤: {e}, {e.__traceback__.tb_lineno}")