import sqlite3
import pandas as pd
import numpy as np
import os

DB_PATH = 'data/market_data.db'
OUTPUT_PATH = 'reports/all_banks_financial_metrics_2021_2025.csv'

def extract_all_metrics():
    if not os.path.exists(DB_PATH):
        print(f"Error: Database not found at {DB_PATH}")
        return

    conn = sqlite3.connect(DB_PATH)
    
    query = """
    SELECT 
        i.ticker,
        i.name as [short name],
        i.country,
        i.region,
        i.size_category as [size],
        mfy.fy,
        mfy.net_income,
        mfy.avg_market_cap,
        mfy.dividend_amt,
        mfy.buyback_amt,
        mfy.eps_fy,
        mfy.dps_fy,
        mfy.dividend_yield_fy,
        mfy.buyback_yield_fy,
        mfy.total_yield_fy,
        mfy.dividend_share_pct,
        mfy.buyback_share_pct,
        mfy.shares_outstanding
    FROM institutions i
    JOIN market_financial_years mfy ON i.lei = mfy.lei
    WHERE mfy.fy BETWEEN 2021 AND 2025
    ORDER BY i.name, mfy.fy
    """
    
    df = pd.read_sql_query(query, conn)
    
    # Add calculated metrics
    # 1. Payout Ratio
    df['payout_ratio_fy'] = (df['dividend_amt'] + df['buyback_amt']) / df['net_income'].replace(0, np.nan)
    
    # 2. BB EPS Rise (Est.)
    # EPS Rise = 1/(1-yield) - 1
    def calc_accretion(y):
        if pd.notna(y) and y > 0 and y < 1:
            return (1 / (1 - y)) - 1
        return 0
    
    df['bb_eps_rise_est'] = df['buyback_yield_fy'].apply(calc_accretion)
    
    # Reorder columns for logical flow
    cols = [
        'ticker', 'short name', 'country', 'region', 'size', 'fy',
        'net_income', 'avg_market_cap', 'dividend_amt', 'buyback_amt',
        'payout_ratio_fy', 'dividend_yield_fy', 'buyback_yield_fy', 'total_yield_fy',
        'bb_eps_rise_est', 'eps_fy', 'dps_fy', 'dividend_share_pct', 'buyback_share_pct'
    ]
    
    df = df[cols]
    
    # Mark 2025 as TTM
    df['fy_status'] = df['fy'].apply(lambda x: 'TTM' if x == 2025 else 'FY')
    
    # Save to CSV
    os.makedirs('reports', exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Successfully extracted all metrics to: {OUTPUT_PATH}")
    
    conn.close()

if __name__ == "__main__":
    extract_all_metrics()
