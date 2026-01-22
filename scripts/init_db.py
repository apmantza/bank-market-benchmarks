import sqlite3
import os
import shutil

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SOURCE_DB = os.path.abspath(os.path.join(BASE_DIR, "..", "eba-benchmarking", "data", "eba_data.db"))
TARGET_DB = os.path.join(BASE_DIR, "data", "market_data.db")

def init_db():
    if os.path.exists(TARGET_DB):
        os.remove(TARGET_DB)
        
    conn = sqlite3.connect(TARGET_DB)
    cur = conn.cursor()
    
    print(f"Initializing {TARGET_DB}...")
    
    # 1. Create Tables
    # Simplified Institutions
    cur.execute("""
        CREATE TABLE institutions (
            lei TEXT PRIMARY KEY,
            name TEXT,
            ticker TEXT,
            country TEXT,
            region TEXT,
            size_category TEXT
        )
    """)
    
    # Market Data (Snapshot)
    # We'll copy the schema structure from the source by reading one row or just duplicating the CREATE statement if we knew it perfectly.
    # To be robust, let's ATTACH the source DB and copy.
    
    
    if not os.path.exists(SOURCE_DB):
        print(f"Error: Source DB not found at {SOURCE_DB}")
        return

    # Copy to avoid locks
    TEMP_SOURCE = os.path.join(BASE_DIR, "data", "temp_source.db")
    print(f"Copying source to {TEMP_SOURCE}...")
    shutil.copy2(SOURCE_DB, TEMP_SOURCE)

    print(f"Attaching source: {TEMP_SOURCE}")
    cur.execute(f"ATTACH DATABASE '{TEMP_SOURCE}' AS source_db")
    
    print("Migrating Institutions...")
    # Map (short_name or commercial_name) -> name, country_iso -> country
    cur.execute("""
        INSERT INTO institutions (lei, name, ticker, country, region, size_category)
        SELECT lei, COALESCE(short_name, commercial_name), ticker, country_iso, region, size_category
        FROM source_db.institutions
        WHERE ticker IS NOT NULL AND trading_status = 'Public'
    """)
    
    print("Migrating Market Data...")
    # Copy structure and data for market_data
    # We can use "CREATE TABLE ... AS SELECT *" but we want to define PKs if possible.
    # Let's just copy for now.
    cur.execute("CREATE TABLE market_data AS SELECT * FROM source_db.market_data")
    
    print("Migrating Market History...")
    cur.execute("CREATE TABLE market_history AS SELECT * FROM source_db.market_history")
    
    print("Migrating Financial Years...")
    cur.execute("CREATE TABLE market_financial_years AS SELECT * FROM source_db.market_financial_years")
    
    conn.commit()

    # Cleanup
    cur.execute("DETACH DATABASE source_db")
    if os.path.exists(TEMP_SOURCE):
        os.remove(TEMP_SOURCE)
    
    # Add Indices/PKs if needed (SQLite 'CREATE TABLE AS' doesn't copy PKs/Indices usually)
    # However for a read-heavy benchmark app, we can add indices now.
    print("Creating Indices...")
    cur.execute("CREATE INDEX idx_hist_lei ON market_history(lei)")
    cur.execute("CREATE INDEX idx_hist_date ON market_history(date)")
    cur.execute("CREATE INDEX idx_fy_lei ON market_financial_years(lei)")
    
    conn.commit()
    conn.close()
    print("Done!")

if __name__ == "__main__":
    # Ensure we run from the repo root
    if not os.path.exists("data"):
        print("Please run from the bank-market-benchmarks root directory.")
    else:
        init_db()
