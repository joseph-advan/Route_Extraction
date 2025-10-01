# analysis/similarity_analyzer.py (V5 - 整合互動流程)

import pandas as pd
from collections import Counter, defaultdict

# --- 分析核心函式 (第一階段) ---
def find_top_companions(full_df: pd.DataFrame, target_plate: str, 
                        time_window_minutes: int = 1, 
                        min_consecutive_points: int = 10) -> dict:
    """
    (V4 - 豐富資訊版) 找出同行夥伴，並回傳詳細的時空維度資訊。
    """
    min_co_occurrences = min_consecutive_points
    if 'LocationID' not in full_df.columns:
        return {"status": "error", "message": "錯誤：資料中缺少 'LocationID' 欄位。", "similar_vehicles": []}
    
    df_sorted = full_df.sort_values(by=['LocationID', 'datetime']).reset_index(drop=True)
    
    co_occurrence_events = []
    for loc_id, group in df_sorted.groupby('LocationID'):
        if len(group) < 2: continue
        time_diffs = group['datetime'].diff() > pd.Timedelta(minutes=time_window_minutes)
        event_ids = time_diffs.cumsum()
        for _, event_group in group.groupby(event_ids):
            if len(event_group) > 1:
                unique_plates = set(event_group['車牌'])
                if target_plate in unique_plates:
                    co_occurrence_events.append({
                        'location': loc_id,
                        'plates': unique_plates,
                        'datetime': event_group['datetime'].mean()
                    })

    if not co_occurrence_events:
        return {"status": "success", "message": f"分析完成，但未找到任何與 {target_plate} 的同行事件。", "similar_vehicles": []}

    companion_stats = defaultdict(lambda: {"datetimes": [], "locations": []})
    for event in co_occurrence_events:
        plates = event['plates']
        plates.remove(target_plate)
        for companion_plate in plates:
            companion_stats[companion_plate]["datetimes"].append(event['datetime'])
            companion_stats[companion_plate]["locations"].append(event['location'])
            
    similar_vehicles = []
    sorted_companions = sorted(companion_stats.items(), key=lambda item: len(item[1]['datetimes']), reverse=True)

    for plate, stats in sorted_companions:
        count = len(stats['datetimes'])
        if count >= min_co_occurrences:
            datetimes = stats['datetimes']
            locations = stats['locations']
            
            first_sighting = min(datetimes).strftime('%Y-%m-%d %H:%M:%S')
            last_sighting = max(datetimes).strftime('%Y-%m-%d %H:%M:%S')
            unique_days = pd.to_datetime(datetimes).normalize().nunique()
            breadth = len(set(locations))
            hotspots = [f"{loc} ({count}次)" for loc, count in Counter(locations).most_common(3)]

            similar_vehicles.append({
                "plate": plate, "score": count, "first_sighting": first_sighting,
                "last_sighting": last_sighting, "unique_days": unique_days,
                "breadth": breadth, "hotspots": hotspots
            })

    if not similar_vehicles:
        return {"status": "success", "message": f"分析完成，但沒有車輛的同行次數達到 {min_co_occurrences} 次門檻。", "similar_vehicles": []}
        
    return {"status": "success", "message": "成功找到同行車輛。", "similar_vehicles": similar_vehicles}

# --- 分析核心函式 (第二階段) ---
def find_hot_routes(full_df: pd.DataFrame, plate1: str, plate2: str, 
                    time_window_minutes: int = 5,
                    min_route_occurrences: int = 3) -> dict:
    """
    分析兩輛指定車牌最常共同行駛的路線 (LocationID_A -> LocationID_B)。
    """
    pair_df = full_df[full_df['車牌'].isin([plate1, plate2])].copy()
    pair_df = pair_df.sort_values(by=['LocationID', 'datetime'])
    co_occurrence_events = []
    for loc_id, group in pair_df.groupby('LocationID'):
        if len(group) < 2: continue
        time_diffs = group['datetime'].diff() > pd.Timedelta(minutes=time_window_minutes)
        event_ids = time_diffs.cumsum()
        for _, event_group in group.groupby(event_ids):
            plates_in_event = set(event_group['車牌'])
            if plate1 in plates_in_event and plate2 in plates_in_event:
                co_occurrence_events.append({'datetime': event_group['datetime'].mean(), 'LocationID': loc_id})
    if len(co_occurrence_events) < 2:
        return {"status": "success", "message": "兩車同行事件過少，無法分析路線。", "hot_routes": []}
    events_df = pd.DataFrame(co_occurrence_events).sort_values(by='datetime')
    events_df['NextLocationID'] = events_df['LocationID'].shift(-1)
    events_df.dropna(subset=['NextLocationID'], inplace=True)
    route_counter = Counter(zip(events_df['LocationID'], events_df['NextLocationID']))
    hot_routes = []
    for (loc_a, loc_b), count in route_counter.most_common():
        if count >= min_route_occurrences:
            hot_routes.append({"route": f"{loc_a} -> {loc_b}", "count": count})
    if not hot_routes:
        return {"status": "success", "message": f"分析完成，但未找到出現超過 {min_route_occurrences} 次的固定同行路線。", "hot_routes": []}
    return {"status": "success", "message": "成功找到熱門同行路線。", "hot_routes": hot_routes[:5]}

# --- 全新：整合性互動流程函式 ---
def run_full_analysis_flow(full_data: pd.DataFrame):
    """
    處理「尋找同行車輛」及後續的「熱門路線分析」的完整互動流程。
    """
    available_plates = sorted(full_data['車牌'].unique())
    print("\n--- 尋找同行車輛 ---")
    for i, plate in enumerate(available_plates): print(f"  [{i+1}] {plate}")

    try:
        choice_input = input(f"請選擇要作為基準的目標車牌 [1-{len(available_plates)}]: ")
        choice = int(choice_input)
        target_plate = available_plates[choice - 1]

        time_window = 1
        min_occur = 10
        print(f"\n--- 正在為 {target_plate} 尋找同行夥伴 (時間窗 ±{time_window} 分鐘，最少 {min_occur} 次同行)... ---")
        
        result = find_top_companions(full_data, target_plate, 
                                     time_window_minutes=time_window, 
                                     min_consecutive_points=min_occur)
        
        print("\n--- 分析結果 ---")
        print(f"狀態: {result['status']}")
        print(f"訊息: {result['message']}")
        
        if result['similar_vehicles']:
            companions = result['similar_vehicles']
            print("\n推薦的同行夥伴:")
            
            for i, vehicle in enumerate(companions):
                print("\n" + "-"*40)
                print(f" {i+1}. 車牌: {vehicle['plate']}")
                print(f"    - 同行地點次數: {vehicle['score']}")
                print(f"    - 首次同行時間: {vehicle['first_sighting']}")
                print(f"    - 末次同行時間: {vehicle['last_sighting']}")
                print(f"    - 同行總天數: {vehicle['unique_days']} 天")
                print(f"    - 同行廣度 (不同地點數): {vehicle['breadth']}")
                print(f"    - Top 3 熱點區域: {', '.join(vehicle['hotspots'])}")
            print("-" * 40)
            
            hot_route_choice = input("\n是否要對其中一輛車進行深入的「熱門同行路線」分析? (y/N): ").lower()
            if hot_route_choice == 'y':
                partner_choice_input = input(f"請選擇要分析的夥伴車牌 [1-{len(companions)}]: ")
                partner_choice = int(partner_choice_input)
                if 1 <= partner_choice <= len(companions):
                    partner_plate = companions[partner_choice - 1]['plate']
                    
                    # 在此直接呼叫並處理熱門路線分析
                    print(f"\n--- 正在分析 {target_plate} 與 {partner_plate} 的熱門同行路線 (最少 3 次)... ---")
                    route_result = find_hot_routes(full_data, target_plate, partner_plate, min_route_occurrences=3)
                    print("\n--- 熱門路線分析結果 ---")
                    print(f"狀態: {route_result['status']}")
                    print(f"訊息: {route_result['message']}")
                    if route_result['hot_routes']:
                        print("\n最常見的同行路線 Top 5:")
                        for route in route_result['hot_routes']:
                            print(f"  - 路線: {route['route']}, 共同行駛次數: {route['count']}")
                else:
                    print("錯誤：無效的選擇。")

    except (ValueError, IndexError):
        print("錯誤：無效的選擇，返回主菜單。")
