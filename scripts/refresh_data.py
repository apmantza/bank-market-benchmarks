import os
import sys
import sqlite3
import pandas as pd

# Add src to path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(BASE_DIR, 'src'))

from market_bench.config import DB_NAME
from market_bench.data import fetch_yahoo_data

def refresh_all_market_data():
    if not os.path.exists(DB_NAME):
        print(f"Database not found at {DB_NAME}")
        return

    conn = sqlite3.connect(DB_NAME)
    cur = conn.cursor()
    
    # Get Public Banks
    cur.execute("SELECT lei, ticker, name, country FROM institutions WHERE ticker IS NOT NULL")
    banks = cur.fetchall()
    
    print(f"Refreshing market data for {len(banks)} banks...")
    
    for lei, ticker, name, country in banks:
        safe_name = name[:30].encode('ascii', 'replace').decode('ascii')
        print(f"  {ticker:12} | {safe_name}...", end=" ")
        
        # Pass LEI and country for valuation logic
        data = fetch_yahoo_data(ticker, lei=lei, country=country)
        
        if data:
            # We assume table exists from init_db, but we should UPSERT
            # SQLite ON CONFLICT
            columns = ['lei', 'ticker'] + list(data.keys())
            placeholders = ','.join(['?'] * len(columns))
            cols_str = ','.join(columns)
            
            # Construct UPDATE clause
            update_clause = ', '.join([f"{k}=Excluded.{k}" for k in data.keys()])
            
            sql = f"""
                INSERT INTO market_data ({cols_str})
                VALUES ({placeholders})
                ON CONFLICT(lei) DO UPDATE SET
                {update_clause}
            """
            
            values = [lei, ticker] + list(data.values())
            cur.execute(sql, values)
            print("OK")
        else:
            print("FAIL")
            
    conn.commit()
    conn.close()
    print("Done!")

if __name__ == "__main__":
    refresh_all_market_data()
