import sqlite3

DB_PATH = r'../bank-market-benchmarks/data/market_data.db'

def add_intrinsic_columns():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    
    # Add columns to market_data
    columns_to_add = [
        ('intrinsic_value', 'REAL'),
        ('upside', 'REAL')
    ]
    
    # Check current columns
    cur.execute("PRAGMA table_info(market_data)")
    existing_cols = [r[1] for r in cur.fetchall()]
    
    for col, ctype in columns_to_add:
        if col not in existing_cols:
            print(f"Adding column {col}...")
            cur.execute(f"ALTER TABLE market_data ADD COLUMN {col} {ctype}")
            
    conn.commit()
    conn.close()
    print("Done.")

if __name__ == "__main__":
    add_intrinsic_columns()
