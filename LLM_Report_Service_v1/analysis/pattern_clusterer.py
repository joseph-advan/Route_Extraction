import pandas as pd
import numpy as np

def get_time_slot(hour: int) -> str:
    """Gets the time slot (e.g., morning, afternoon) based on the hour."""
    if 5 <= hour < 8: return "清晨"
    elif 8 <= hour < 12: return "上午"
    elif 12 <= hour < 18: return "下午"
    elif 18 <= hour < 23: return "晚上"
    else: return "深夜"

def calculate_circular_avg_hour(hours: pd.Series) -> float:
    """Calculates the circular average of hours to correctly handle times crossing midnight."""
    # Convert hours (0-24) to radians on a circle
    radians = 2 * np.pi * hours / 24
    # Calculate the average of the x and y components of each time vector
    sin_avg = np.mean(np.sin(radians))
    cos_avg = np.mean(np.cos(radians))
    # Convert the average vector back to a radian angle
    avg_radian = np.arctan2(sin_avg, cos_avg)
    # Convert the radian angle back to an hour (0-24)
    avg_hour = avg_radian * 24 / (2 * np.pi)
    # Ensure the result is positive (e.g., -1 becomes 23)
    return avg_hour if avg_hour >= 0 else avg_hour + 24

def get_day_type(day_of_week: int) -> str:
    """Determines if the day is a weekday or weekend."""
    if day_of_week < 5:  # Monday is 0, Sunday is 6
        return "工作日"
    else:
        return "週末"

def find_regular_patterns_v13(trips: list, stay_points: list, all_cameras_with_area: pd.DataFrame,
                              confirmed_threshold: int = 4,
                              secondary_base_threshold: int = 3,
                              long_stay_duration_hours: float = 4.0) -> dict:
    """
    (V13) Adds intelligent stay point statistics, robust mapping, and complete pattern data.
    - Differentiates between "single stay" and "multiple stays".
    - Provides a duration range (min/max) for "multiple stays".
    - Builds a complete location map to prevent unknown locations.
    - Includes all necessary keys in the final pattern summary.
    """
    analysis_summary = {
        "base_info": { "primary": None, "secondary": [] },
        "all_stay_points_stats": [],
        "regular_patterns": []
    }

    # Create a complete map from ALL cameras first to prevent "未知" locations.
    # We group by the Area ID and take the first camera name as its representative name.
    temp_map_df = all_cameras_with_area.drop_duplicates(subset=['LocationAreaID'])
    area_to_name_map = pd.Series(
        temp_map_df['攝影機名稱'].values,
        index=temp_map_df['LocationAreaID']
    ).to_dict()

    if not trips or not stay_points:
        return { "summary": analysis_summary, "area_map": area_to_name_map, "trips_df": pd.DataFrame(trips) }

    trips_df = pd.DataFrame(trips)
    stay_points_df = pd.DataFrame(stay_points)

    # --- Calculate Statistics for All Stay Points ---
    if not stay_points_df.empty:
        stay_points_df['arrival_hour_float'] = stay_points_df['start_time'].dt.hour + stay_points_df['start_time'].dt.minute / 60
        stay_points_df['departure_hour_float'] = stay_points_df['end_time'].dt.hour + stay_points_df['end_time'].dt.minute / 60

        # Step A: Aggregate all non-time-averaging stats first
        location_stats = stay_points_df.groupby('location_area_id').agg(
            visit_count=('location_area_id', 'count'),
            total_duration_minutes=('duration_minutes', 'sum'),
            avg_duration_minutes=('duration_minutes', 'mean'),
            min_duration_minutes=('duration_minutes', 'min'),
            max_duration_minutes=('duration_minutes', 'max')
        ).reset_index()

        # Step B: Calculate circular averages for time and merge them in
        time_avg_stats = stay_points_df.groupby('location_area_id').agg(
            avg_arrival_hour=('arrival_hour_float', calculate_circular_avg_hour),
            avg_departure_hour=('departure_hour_float', calculate_circular_avg_hour)
        ).reset_index()

        location_stats = pd.merge(location_stats, time_avg_stats, on='location_area_id')
        # --- END OF REPLACEMENT BLOCK ---

        sorted_locations = location_stats.sort_values(by='total_duration_minutes', ascending=False)

        for _, row in sorted_locations.iterrows():
            avg_arrival_h, avg_arrival_m = divmod(row['avg_arrival_hour'] * 60, 60)
            avg_departure_h, avg_departure_m = divmod(row['avg_departure_hour'] * 60, 60)

            stats_dict = {
                "area_id": row['location_area_id'],
                "name": area_to_name_map.get(row['location_area_id'], "地點未知"),
                "visit_count": int(row['visit_count']),
                "total_duration_hours": round(row['total_duration_minutes'] / 60, 1)
            }
            if int(row['visit_count']) > 1:
                stats_dict["stay_pattern_type"] = "多次停留"
                stats_dict["avg_duration_hours"] = round(row['avg_duration_minutes'] / 60, 1)
                stats_dict["duration_range_hours"] = [
                    round(row['min_duration_minutes'] / 60, 1),
                    round(row['max_duration_minutes'] / 60, 1)
                ]
                stats_dict["avg_arrival_time"] = f"{int(avg_arrival_h):02d}:{int(avg_arrival_m):02d}"
                stats_dict["avg_departure_time"] = f"{int(avg_departure_h):02d}:{int(avg_departure_m):02d}"
            else:
                stats_dict["stay_pattern_type"] = "單次長時停留"

            analysis_summary["all_stay_points_stats"].append(stats_dict)

    # --- Find Primary and Secondary Bases ---
    long_stay_threshold_minutes = long_stay_duration_hours * 60
    long_stays_df = stay_points_df[stay_points_df['duration_minutes'] > long_stay_threshold_minutes]
    if not long_stays_df.empty:
        long_stay_counts = long_stays_df.groupby('location_area_id').size().sort_values(ascending=False)
        for area_id, count in long_stay_counts.items():
            stats = next((sp for sp in analysis_summary["all_stay_points_stats"] if sp["area_id"] == area_id), None)
            if not stats: continue
            stats['long_stay_count'] = count
            if analysis_summary["base_info"]["primary"] is None:
                analysis_summary["base_info"]["primary"] = stats
            elif count >= secondary_base_threshold:
                analysis_summary["base_info"]["secondary"].append(stats)

    # --- Build Trip Features ---
    trips_df['start_hour_float'] = trips_df['start_time'].dt.hour + trips_df['start_time'].dt.minute / 60
    trips_df['end_hour_float'] = trips_df['end_time'].dt.hour + trips_df['end_time'].dt.minute / 60
    trips_df['day_of_week'] = trips_df['start_time'].dt.dayofweek
    trips_df['day_type'] = trips_df['day_of_week'].apply(get_day_type)
    trips_df['time_slot'] = trips_df['start_time'].dt.hour.apply(get_time_slot)
    trips_df['signature'] = (
        trips_df['start_area_id'].astype(str) + '->' +
        trips_df['end_area_id'].astype(str) + '_' +
        trips_df['day_type'].astype(str) + '_' +
        trips_df['time_slot'].astype(str)
    )
    pattern_groups = trips_df.groupby('signature').filter(lambda x: len(x) >= confirmed_threshold)
    if not pattern_groups.empty:
        for signature, group in pattern_groups.groupby('signature'):
            avg_start_h, avg_start_m = divmod(group['start_hour_float'].mean() * 60, 60)
            avg_end_h, avg_end_m = divmod(group['end_hour_float'].mean() * 60, 60)
            
            try:
                # The signature is like "Area-1->Area-5_工作日_下午". Split it to get the parts.
                _, day_type, time_slot = signature.split('_')
            except ValueError:
                # Fallback for unexpected signature format
                day_type = "未知"
                time_slot = "未知"
            
            analysis_summary["regular_patterns"].append({
                'signature': signature, 
                'start_area_id': group['start_area_id'].iloc[0],
                'end_area_id': group['end_area_id'].iloc[0], 
                'occurrence_count': len(group),
                'occurrence_days': group['start_time'].dt.date.nunique(),
                'avg_duration_minutes': round(group['duration_minutes'].mean(), 2),
                'avg_start_time': f"{int(avg_start_h):02d}:{int(avg_start_m):02d}",
                'avg_end_time': f"{int(avg_end_h):02d}:{int(avg_end_m):02d}",
                'day_type': day_type,
                'time_slot': time_slot
            })
            
    analysis_summary["regular_patterns"] = sorted(analysis_summary["regular_patterns"], key=lambda p: p['occurrence_count'], reverse=True)
            
    return { "summary": analysis_summary, "area_map": area_to_name_map, "trips_df": trips_df }