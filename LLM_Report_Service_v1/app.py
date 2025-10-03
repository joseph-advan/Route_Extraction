# app.py (V9 - 精簡版)

import pandas as pd
from pathlib import Path

# 匯入需要的分析函式
from reporting_service import run_llm_reporting_flow
# 【已移除】similarity_analyzer 的匯入
from analysis.convoy_analyzer import run_trip_oriented_convoy_analysis

def main_console():
    """
    應用主控台，提供使用者選擇不同的分析功能。
    """
    # --- 步驟 1: 載入並預處理資料 ---
    try:
        current_file_path = Path(__file__)
        project_root_path = current_file_path.parent
        
        DATA_FILE_PATH = project_root_path / 'data' / 'realistic_vehicle_dataset1.csv'
        full_data = pd.read_csv(DATA_FILE_PATH)
        full_data['datetime'] = pd.to_datetime(full_data['日期'] + ' ' + full_data['時間'])
        full_data = full_data.sort_values(by='datetime').reset_index(drop=True)
        
        if 'LocationID' not in full_data.columns:
            print(f"錯誤：資料檔案 {DATA_FILE_PATH} 中缺少 'LocationID' 欄位。")
            return
        
        full_data['LocationID'] = full_data['LocationID'].astype(str)
            
        print("--- 成功讀取並預處理軌跡資料 ---")

    except FileNotFoundError:
        print(f"錯誤：找不到軌跡資料檔案 {DATA_FILE_PATH}。請確認檔案是否存在於 data 資料夾中。")
        return
    except Exception as e:
        print(f"讀取資料時發生錯誤: {e}")
        return

    # --- 步驟 2: 顯示功能菜單並接收使用者選擇 ---
    while True:
        print("\n" + "="*50)
        print("== 車輛軌跡智慧分析系統 ==")
        print("="*50)
        # 【【【 核心修改處：簡化選單 】】】
        print("  [1] 生成單一車輛深度分析報告 (LLM)")
        print("  [2] 分析目標行程的隨行車輛 (行程導向)")
        print("  [q] 結束程式")
        
        choice = input("請輸入您的選擇: ")
        
        if choice == '1':
            run_single_vehicle_analysis(full_data)
        # 【【【 核心修改處：移除選項 2，並將選項 3 改為 2 】】】
        elif choice == '2':
            run_trip_oriented_convoy_analysis(full_data)
        elif choice.lower() == 'q':
            print("感謝使用，程式結束。")
            break
        else:
            print("無效的選擇，請重新輸入。")

# (run_single_vehicle_analysis 函式維持不變)
def run_single_vehicle_analysis(full_data):
    """處理「單一車輛報告生成」的使用者互動與呼叫"""
    available_plates = sorted(full_data['車牌'].unique())
    print("\n--- 生成單一車輛深度分析報告 ---")
    print("資料集中可用的車牌號碼：")
    for i, plate in enumerate(available_plates): print(f"  [{i+1}] {plate}")
    
    try:
        choice_input = input(f"請選擇要分析的車牌 [1-{len(available_plates)}]: ")
        choice = int(choice_input)
        target_plate = available_plates[choice - 1]

        debug_choice = input("是否啟用除錯模式 (y/N)? ").lower()
        debug_mode = True if debug_choice == 'y' else False

        run_llm_reporting_flow(full_data, target_plate, debug_mode=debug_mode)
    except (ValueError, IndexError):
        print("錯誤：無效的選擇，返回主菜單。")

if __name__ == '__main__':
    main_console()