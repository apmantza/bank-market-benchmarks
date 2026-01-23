import sqlite3
import pandas as pd
import sys
import os
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

DB_PATH = 'data/market_data.db'

def generate_cumulative_report():
    if not os.path.exists(DB_PATH):
        print(f"Error: Database not found at {DB_PATH}")
        return

    conn = sqlite3.connect(DB_PATH)
    
    # Get base data
    query = """
    SELECT 
        i.ticker,
        i.name as [short name],
        i.country,
        i.region,
        i.size_category as size,
        mfy.fy,
        mfy.net_income,
        mfy.dividend_amt,
        mfy.buyback_amt,
        mfy.dividend_yield_fy,
        mfy.buyback_yield_fy,
        mfy.total_yield_fy,
        mfy.eps_fy,
        mfy.dps_fy,
        mfy.dividend_share_pct,
        mfy.buyback_share_pct
    FROM institutions i
    JOIN market_financial_years mfy ON i.lei = mfy.lei
    WHERE mfy.fy BETWEEN 2021 AND 2025
    """
    
    df = pd.read_sql_query(query, conn)
    
    # Group by bank
    results = []
    for ticker, group in df.groupby('ticker'):
        # Metadata
        meta = group.iloc[0][['ticker', 'short name', 'country', 'region', 'size']].to_dict()
        
        # Cumulative
        cum_ni = group['net_income'].sum()
        cum_div = group['dividend_amt'].sum()
        cum_bb = group['buyback_amt'].sum()
        cum_payout = cum_div + cum_bb
        
        # Averages
        avg_div_yield = group['dividend_yield_fy'].mean()
        avg_bb_yield = group['buyback_yield_fy'].mean()
        avg_total_yield = group['total_yield_fy'].mean()
        avg_eps = group['eps_fy'].mean()
        avg_dps = group['dps_fy'].mean()
        
        # Annual Payout Ratio (per year, then averaged)
        # Avoid division by zero
        group['ann_payout_ratio'] = (group['dividend_amt'] + group['buyback_amt']) / group['net_income'].replace(0, float('nan'))
        avg_ann_payout_ratio = group['ann_payout_ratio'].mean()
        
        # Metrics
        meta['cum net income'] = cum_ni
        meta['cum dividend amount'] = cum_div
        meta['cum buyback amount'] = cum_bb
        meta['cum payout amount'] = cum_payout
        meta['cum payout % of net income'] = cum_payout / cum_ni if cum_ni > 0 else 0
        meta['avg annual payout % of net income'] = avg_ann_payout_ratio
        meta['cum dividend % of cum payouts'] = cum_div / cum_payout if cum_payout > 0 else 0
        meta['cum buyback % of cum payouts'] = cum_bb / cum_payout if cum_payout > 0 else 0
        meta['avg annual dividend yield'] = avg_div_yield
        meta['avg annual buyback yield'] = avg_bb_yield
        meta['avg payout yield'] = avg_total_yield
        meta['avg annual EPS'] = avg_eps
        meta['avg annual DPS'] = avg_dps
        
        results.append(meta)
        
    final_df = pd.DataFrame(results)
    
    # Reorder columns to match image exactly
    cols_order = [
        'ticker', 'short name', 'country', 'region', 'size',
        'cum net income', 'cum dividend amount', 'cum buyback amount', 'cum payout amount',
        'cum payout % of net income', 'avg annual payout % of net income',
        'cum dividend % of cum payouts', 'cum buyback % of cum payouts',
        'avg annual dividend yield', 'avg annual buyback yield', 'avg payout yield',
        'avg annual EPS', 'avg annual DPS'
    ]
    
    # Check if any requested column is missing
    final_df = final_df[[c for c in cols_order if c in final_df.columns]]
    
    # Save to CSV
    output_path = 'reports/cumulative_remuneration_2021_2025.csv'
    final_df.to_csv(output_path, index=False)
    print(f"Report generated: {output_path}")
    
    conn.close()

if __name__ == "__main__":
    generate_cumulative_report()
