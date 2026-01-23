import pandas as pd
import sqlite3
import yfinance as yf
import os
from datetime import datetime, timedelta
import numpy as np
from .config import DB_NAME

# Conditional streamlit import for caching
try:
    import streamlit as st
    cache_decorator = st.cache_data(ttl=3600)
except:
    # Fallback when running as script (no streamlit)
    def cache_decorator(func):
        return func

# Currency conversion utilities
CURRENCY_FX_PAIRS = {
    'EUR': None,
    'GBp': 'GBPEUR=X',
    'SEK': 'EURSEK=X',
    'NOK': 'EURNOK=X',
    'DKK': 'EURDKK=X',
    'PLN': 'EURPLN=X',
    'HUF': 'EURHUF=X',
    'CZK': 'EURCZK=X',
    'RON': 'EURRON=X',
    'ISK': 'EURISK=X',
    'CHF': 'EURCHF=X',
    'USD': 'EURUSD=X',
    'GBP': 'GBPEUR=X',
}


def get_fx_rate(currency):
    if currency == 'EUR': return 1.0
    # Handle pence sterling by lookup GBP
    lookup_currency = 'GBP' if currency == 'GBp' else currency
    fx_pair = CURRENCY_FX_PAIRS.get(lookup_currency)
    if not fx_pair: return None
    try:
        fx = yf.Ticker(fx_pair)
        info = fx.info
        rate = info.get('regularMarketPrice') or info.get('previousClose')
        if rate:
            # Calculate base FX rate to EUR
            if 'GBP' in fx_pair and fx_pair.startswith('GBP'): 
                return rate
            else: 
                return 1.0 / rate
        return None
    except: return None

# Ticker overrides for historical data
HISTORY_TICKER_MAP = {}

@cache_decorator
def get_market_data(lei_list=None):
    if not os.path.exists(DB_NAME): return pd.DataFrame()
    conn = sqlite3.connect(DB_NAME)
    
    # Simplified query on new schema
    query = """
        SELECT m.*, i.name, i.country, i.region, i.size_category
        FROM market_data m
        JOIN institutions i ON m.lei = i.lei
    """
    if lei_list:
        leis_str = "'" + "','".join([str(l) for l in lei_list]) + "'"
        query += f" WHERE m.lei IN ({leis_str})"
    
    try:
        df = pd.read_sql(query, conn)
        conn.close()
        return df
    except Exception as e:
        print(f"Error in get_market_data: {e}")
        conn.close()
        return pd.DataFrame()

@cache_decorator
def get_market_history(lei_list=None, period="5y"):
    if not os.path.exists(DB_NAME): return pd.DataFrame()
    conn = sqlite3.connect(DB_NAME)
    query = """
        SELECT h.*, i.name, i.country
        FROM market_history h
        JOIN institutions i ON h.lei = i.lei
    """
    if lei_list:
        leis_str = "'" + "','".join([str(l) for l in lei_list]) + "'"
        query += f" WHERE h.lei IN ({leis_str})"
    query += " ORDER BY h.date"
    
    try:
        df = pd.read_sql(query, conn)
        conn.close()
        return df
    except:
        conn.close()
        return pd.DataFrame()

@cache_decorator
def get_market_financial_years(lei_list=None):
    if not os.path.exists(DB_NAME): return pd.DataFrame()
    conn = sqlite3.connect(DB_NAME)
    query = """
        SELECT f.*, i.name
        FROM market_financial_years f
        JOIN institutions i ON f.lei = i.lei
    """
    if lei_list:
        leis_str = "'" + "','".join([str(l) for l in lei_list]) + "'"
        query += f" WHERE f.lei IN ({leis_str})"
    query += " ORDER BY f.fy DESC"
    
    try:
        df = pd.read_sql(query, conn)
        # Strategic Metrics Calculation
        total_payout = df['dividend_amt'] + df['buyback_amt']
        df['payout_ratio_fy'] = total_payout / df['net_income']
        df['dividend_payout_ratio_fy'] = df['dividend_amt'] / df['net_income']
        
        # Avoid negative earnings distortion
        df.loc[df['net_income'] <= 0, 'payout_ratio_fy'] = np.nan
        df.loc[df['net_income'] <= 0, 'dividend_payout_ratio_fy'] = np.nan
        
        df['earnings_yield_fy'] = df['net_income'] / df['avg_market_cap']
        conn.close()
        return df
    except Exception as e:
        conn.close()
        return pd.DataFrame()

@cache_decorator
def get_structured_benchmark_data(base_lei):
    """
    Returns a unified DataFrame sorted specifically for the new Benchmarking Tab:
    1. Base Bank
    2. Domestic Peers + Domestic Avg
    3. Regional Peers (excl domestic/small) + Regional Avg
    4. Core Peers (excl small) + Core Avg
    5. CEE Peers (all) + CEE Avg
    6. EU Avg
    """
    df_all = get_market_data()
    if df_all.empty: return pd.DataFrame()
    
    # Identify Base Bank
    base = df_all[df_all['lei'] == base_lei]
    if base.empty: return pd.DataFrame()
    base_row = base.iloc[0]
    base_country = base_row['country']
    base_region = base_row['region']
    
    # Constants
    ATHEX_PEER_LEIS = ['635400L14KNHZXPUZM19'] # Bank of Cyprus
    CORE_REGIONS = ['Western Europe', 'Northern Europe']
    SMALL_SIZE = 'Small (<50bn)'
    
    # --- GROUPS DEFINITION ---
    # 1. Domestic
    domestic_mask = (df_all['country'] == base_country) | (df_all['lei'].isin(ATHEX_PEER_LEIS) if base_country == 'GR' else False)
    domestic_peers = df_all[domestic_mask & (df_all['lei'] != base_lei)].sort_values('market_cap', ascending=False)
    
    # 2. Regional (Same Region, Excl Domestic, Excl Small)
    regional_mask = (df_all['region'] == base_region) & (~domestic_mask) & (df_all['size_category'] != SMALL_SIZE)
    regional_peers = df_all[regional_mask].sort_values('market_cap', ascending=False)
    
    # 3. Core (Western/Northern, Excl Small)
    core_mask = (df_all['region'].isin(CORE_REGIONS)) & (df_all['size_category'] != SMALL_SIZE)
    core_peers = df_all[core_mask].sort_values('market_cap', ascending=False)
    
    # 4. CEE (All, if not already captured in regional, but request says "CEE peers")
    # If base is CEE, this overlaps with Regional. The request implies a specific CEE section.
    # Let's define CEE peers as ALL CEE banks (excluding those already shown? no, "CEE peers")
    # To avoid duplicates if Base/Regional *is* CEE:
    #   - If Base is CEE, "Regional" is CEE. "CEE Peers" would be redundant.
    #   - Let's follow the requested structure strictly, but maybe filter duplicates in display?
    #   - Request: "Regional peers... followed by CEE peers".
    #   - Let's assume CEE bucket is explicitly for CEE banks.
    cee_mask = (df_all['region'] == 'CEE')
    # Exclude banks already in Domestic? Usually yes.
    cee_peers = df_all[cee_mask & (~domestic_mask)].sort_values('market_cap', ascending=False)
    
    # If Base is in CEE, Regional and CEE might be identical.
    # Let's skip "Regional" block if Region == CEE to avoid complete duplication, or just show distinct subsets.
    # Actually, "Regional Peers" logic above uses (df_all['region'] == base_region).
    # If Base Region is CEE, "Regional Peers" = CEE Peers.
    # We will include distinct groups as requested.
    
    numeric_cols = ['current_price', 'market_cap', 'pe_trailing', 'price_to_book', 'dividend_yield', 
                    'buyback_yield', 'payout_yield', 'beta', 'return_1y', 'return_3y', 'return_5y',
                    'eps_trailing', 'dps_trailing', 'payout_ratio', 'intrinsic_value', 'upside']

    def create_avg_row(df_group, name):
        if df_group.empty: return None
        avg = df_group[numeric_cols].mean().to_dict()
        avg['name'] = name
        return avg

    structured_rows = []
    
    # 1. Base Bank
    structured_rows.append(base_row.to_dict())
    
    # 2. Domestic
    if not domestic_peers.empty:
        structured_rows.extend(domestic_peers.to_dict('records'))
        avg = create_avg_row(pd.concat([base, domestic_peers]), f"Domestic Avg ({base_country})")
        if avg: structured_rows.append(avg)
    
    # 3. Regional
    # If Base is CEE, "Regional" is redundant with "CEE". 
    # Logic: Show Regional ONLY if Region is NOT CEE, OR if we want to separate "Large CEE" vs "All CEE".
    # User said: "Regional peers (excl. domestics and small banks)"
    if not regional_peers.empty:
        # Check if we should skip to avoid dupes with CEE section?
        # If base region is CEE, let's keep it but label it "Regional (Large/Med)"
        structured_rows.extend(regional_peers.to_dict('records'))
        avg = create_avg_row(regional_peers, f"Regional Avg ({base_region}, >Small)")
        if avg: structured_rows.append(avg)

    # 4. Core
    if not core_peers.empty:
        structured_rows.extend(core_peers.to_dict('records'))
        avg = create_avg_row(core_peers, "Core Europe Avg (>Small)")
        if avg: structured_rows.append(avg)

    # 5. CEE
    if not cee_peers.empty:
        # Determine overlap with Regional
        # We want to show CEE peers. If they were already shown in Regional, we might duplicate.
        # Filter out banks already in (Regional Peers) to avoid showing same bank twice in one chart?
        # Bar charts with duplicate x-axis labels are confusing.
        # Let's Filter CEE peers to exclude those already added in Regional.
        regional_ids = regional_peers['lei'].tolist() if not regional_peers.empty else []
        domestic_ids = domestic_peers['lei'].tolist() if not domestic_peers.empty else []
        
        cee_unique = cee_peers[~cee_peers['lei'].isin(regional_ids)]
        # Actually user might WANT to see them again grouped as CEE.
        # But for 'clean' charts, unique is better.
        # However, "CEE Avg" should include ALL CEE.
        
        if not cee_unique.empty:
            structured_rows.extend(cee_unique.to_dict('records'))
        
        # Calculate CEE Avg using ALL CEE banks (not just unique rest)
        all_cee = df_all[cee_mask]
        avg = create_avg_row(all_cee, "CEE Avg (All)")
        if avg: structured_rows.append(avg)

    # 6. EU Avg
    eu_avg = create_avg_row(df_all, "EU Wide Avg")
    if eu_avg: structured_rows.append(eu_avg)
    
    return pd.DataFrame(structured_rows)

@cache_decorator
def get_ranking_data():
    """Returns all market data for Ranking Tab."""
    return get_market_data()

@cache_decorator
def get_market_fy_averages(base_country, base_region, base_size):
    df_fy_all = get_market_financial_years()
    if df_fy_all.empty: return pd.DataFrame()
        
    conn = sqlite3.connect(DB_NAME)
    df_meta = pd.read_sql("SELECT lei, country, region, size_category FROM institutions", conn)
    conn.close()
    
    df_fy_all = pd.merge(df_fy_all, df_meta, on='lei', how='left')
    
    ATHEX_PEER_LEIS = ['635400L14KNHZXPUZM19']
    CORE_REGIONS = ['Western Europe', 'Northern Europe']
    SMALL_SIZE = 'Small (<50bn)'
    
    group_filters = {
        "Domestic Avg": (df_fy_all['country'] == base_country) | (df_fy_all['lei'].isin(ATHEX_PEER_LEIS) if base_country == 'GR' else False),
        "Regional (Same Size)": (df_fy_all['region'] == base_region) & (df_fy_all['size_category'] == base_size),
        "Regional (All but Small)": (df_fy_all['region'] == base_region) & (df_fy_all['size_category'] != SMALL_SIZE),
        "Core (Same Size)": (df_fy_all['region'].isin(CORE_REGIONS)) & (df_fy_all['size_category'] == base_size),
        "Core (All but Small)": (df_fy_all['region'].isin(CORE_REGIONS)) & (df_fy_all['size_category'] != SMALL_SIZE),
        "CEE (All)": (df_fy_all['region'] == 'CEE')
    }
    
    fy_stats = []
    fy_numeric_cols = ['dividend_yield_fy', 'buyback_yield_fy', 'total_yield_fy', 
                       'payout_ratio_fy', 'dividend_payout_ratio_fy', 'earnings_yield_fy',
                       'eps_fy', 'dps_fy']
    
    for label, mask in group_filters.items():
        group_df = df_fy_all[mask]
        if not group_df.empty:
            group_avgs = group_df.groupby('fy')[fy_numeric_cols].mean().reset_index()
            group_avgs['name'] = label
            fy_stats.append(group_avgs)
            
    if not fy_stats: return pd.DataFrame()
    return pd.concat(fy_stats, ignore_index=True)

# Fetching Logic (Simplified for refresh_data)
def fetch_yahoo_data(ticker, lei=None, country=None):
    try:
        stock = yf.Ticker(ticker)
        # Fetch 5 years
        hist = stock.history(period="5y") 
        info = stock.info
        
        # Manual Overrides (NLB and others)
        if lei == '5493001BABFV7P27OW30': # NLB
            if not info.get('sharesOutstanding'): info['sharesOutstanding'] = 20_000_000
            if not info.get('bookValue'): info['bookValue'] = 111.0
            
        shares = info.get('sharesOutstanding') or 0
        currency = info.get('currency', 'EUR')
        fx_rate = get_fx_rate(currency) or 1.0
        
        # GBp (Pence) tickers have Price in pence, but secondary financial metrics (Book Value, EPS, etc.) usually in GBP (Pounds).
        # We need a price-specific divisor.
        price_divisor = 100.0 if currency == 'GBp' else 1.0
        
        def to_eur_price(v): return (float(v) * fx_rate / price_divisor) if v else None
        def to_eur_metric(v): return (float(v) * fx_rate) if v else None # Used for Cap, Book Value, EPS, DPS, Buybacks

        current = to_eur_price(info.get('currentPrice') or info.get('regularMarketPrice'))
        book_val = to_eur_metric(info.get('bookValue'))
        eps = to_eur_metric(info.get('trailingEps'))
        dps = to_eur_metric(info.get('dividendRate'))

        # Market Cap Sanity Check
        yf_cap = to_eur_metric(info.get('marketCap'))
        calc_cap = None
        market_cap_final = yf_cap
        
        if current and shares > 0:
            calc_cap = current * shares
            
        # Use calculated if YF is missing or deviates > 20% (e.g. double counting error)
        if calc_cap and (not yf_cap or abs(yf_cap - calc_cap) / calc_cap > 0.2):
            market_cap_final = calc_cap
        
        # Standardize Dividend Yield (ensure it is 0.05 rather than 5.0)
        dy = info.get('dividendYield')
        if dy and dy > 1.0: # e.g. 6.46 meaning 6.46%
            dy = dy / 100.0

        data = {
            'current_price': current,
            'fetch_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'currency': 'EUR', # We normalize to EUR
            'market_cap': market_cap_final,
            'price_to_book': None, # Calc below
            'pe_trailing': info.get('trailingPE'),
            'dividend_yield': dy,
            'beta': info.get('beta'),
            'return_1y': None, 'return_3y': None, 'return_5y': None,
            'buyback_yield': None, 'payout_yield': None,
            'eps_trailing': eps,
            'dps_trailing': dps
        }

        # P/B calc
        if current and book_val and book_val > 0:
            data['price_to_book'] = current / book_val
            
        # Returns calculation
        if not hist.empty:
            hist['CloseEUR'] = hist['Close'] * fx_rate
            curr = hist['CloseEUR'].iloc[-1]
            def get_ret(days):
                start = datetime.now() - timedelta(days=days)
                # Ensure start is timezone-naive if hist index is, or aware if it is
                hist_idx = hist.index
                if hist_idx.tz is not None:
                     start = start.astimezone(hist_idx.tz)
                
                # Filter
                hist_slice = hist[hist_idx <= start]
                start_p = hist_slice['CloseEUR'].iloc[-1] if not hist_slice.empty else None
                return (curr - start_p) / start_p if start_p else None
                
            data['return_1y'] = get_ret(365)
            data['return_3y'] = get_ret(3*365)
            data['return_5y'] = get_ret(5*365)

        # Buyback Yield Calculation (TTM)
        try:
            buyback_amt_local = 0
            
            # 1. Try Quarterly (TTM sum)
            qcf = stock.quarterly_cashflow
            if qcf is not None and not qcf.empty:
                bb_rows = [i for i in qcf.index if 'Repurchase' in i or 'Buyback' in i or 'Common Stock Payments' in i]
                if bb_rows:
                    best_row = next((r for r in bb_rows if 'Repurchase Of Capital Stock' in r), bb_rows[0])
                    row_data = qcf.loc[best_row].iloc[:4]
                    buyback_q_sum = abs(row_data[row_data < 0].sum())
                    buyback_amt_local = max(buyback_amt_local, buyback_q_sum)

            # 2. Try Annual (Last Year) - often more reliable for EU banks
            acf = stock.cashflow
            if acf is not None and not acf.empty:
                 bb_rows_a = [i for i in acf.index if 'Repurchase' in i or 'Buyback' in i or 'Common Stock Payments' in i]
                 for row_name in bb_rows_a:
                     val = acf.loc[row_name].iloc[0]
                     val_abs = abs(val) if val < 0 else 0 # Only count outflows
                     buyback_amt_local = max(buyback_amt_local, val_abs)
            
            if buyback_amt_local > 0:
                buyback_amt_eur = to_eur_metric(buyback_amt_local)
                if data['market_cap'] and data['market_cap'] > 0:
                    data['buyback_yield'] = buyback_amt_eur / data['market_cap']

        except Exception:
            pass
            
        # Payout Yield
        if data['buyback_yield'] is not None or data['dividend_yield'] is not None:
            dy_val = data['dividend_yield'] or 0
            by_val = data['buyback_yield'] or 0
            data['payout_yield'] = dy_val + by_val
            
        # --- Intrinsic Value Calculation (Residual Income Model / GGM) ---
        # IV = BV * (ROE - g) / (COE - g)
        try:
            # Assumptions
            G = 0.02 # Long term growth
            RF = 0.035 # Base Risk free rate (EUR benchmark)
            ERP = 0.055 # Equity Risk Premium
            
            bv = book_val # Already in EUR
            eps = data.get('eps_trailing')
            beta = data.get('beta') or 1.0 # Default to 1.0 if missing
            price = data.get('current_price')
            
            if bv and bv > 0 and eps and price:
                roe = eps / bv
                # Unified COE across all EU banks
                coe = RF + abs(beta) * ERP
                data['coe'] = coe
                
                # Formula: Price = BV * (ROE - g) / (COE - g)
                if coe > G:
                    iv = bv * (roe - G) / (coe - G)
                    data['intrinsic_value'] = iv
                    data['upside'] = (iv - price) / price if price > 0 else None
                else:
                    data['coe'] = coe
                    data['intrinsic_value'] = None
                    data['upside'] = None
        except Exception:
            data['intrinsic_value'] = None
            data['upside'] = None

        return data
        
    except Exception as e:
        print(f"Error fetching {ticker}: {e}")
        import traceback
        traceback.print_exc()
        return None

@cache_decorator
def get_all_banks():
    if not os.path.exists(DB_NAME): return pd.DataFrame()
    conn = sqlite3.connect(DB_NAME)
    try:
        df = pd.read_sql("SELECT lei, name, ticker, country, region, size_category FROM institutions ORDER BY name", conn)
        conn.close()
        return df
    except:
        conn.close()
        return pd.DataFrame()


# =============================================================================
# SPRINT 3: RANKING ENHANCEMENTS - PER-YEAR AND CUMULATIVE METRICS
# =============================================================================

@cache_decorator
def get_ranking_data_by_year(fy: int):
    """
    Get ranking data for a specific fiscal year.

    Args:
        fy: Fiscal year (e.g., 2024, 2023)

    Returns:
        DataFrame with per-year metrics from market_financial_years
    """
    if not os.path.exists(DB_NAME):
        return pd.DataFrame()

    conn = sqlite3.connect(DB_NAME)

    query = """
        SELECT
            i.lei,
            i.name,
            i.ticker,
            i.country,
            i.region,
            i.size_category,
            f.fy,
            f.dividend_amt,
            f.buyback_amt,
            f.net_income,
            f.avg_market_cap,
            f.dividend_yield_fy,
            f.buyback_yield_fy,
            f.total_yield_fy,
            f.eps_fy,
            f.dps_fy,
            f.dividend_share_pct,
            f.buyback_share_pct
        FROM institutions i
        JOIN market_financial_years f ON i.lei = f.lei
        WHERE f.fy = ?
        ORDER BY i.name
    """

    try:
        df = pd.read_sql_query(query, conn, params=(fy,))
        conn.close()
        
        if not df.empty:
            # Calculate metrics not stored in DB
            total_payout = df['dividend_amt'].fillna(0) + df['buyback_amt'].fillna(0)
            
            df['payout_ratio_fy'] = total_payout / df['net_income']
            df['dividend_payout_ratio_fy'] = df['dividend_amt'] / df['net_income']
            df['earnings_yield_fy'] = df['net_income'] / df['avg_market_cap']
            
            # Identify columns to mask (where net_income <= 0)
            mask = df['net_income'] <= 0
            df.loc[mask, 'payout_ratio_fy'] = None
            df.loc[mask, 'dividend_payout_ratio_fy'] = None
            
        return df
    except Exception as e:
        print(f"Error getting ranking data for FY {fy}: {e}")
        conn.close()
        return pd.DataFrame()


@cache_decorator
def calculate_cumulative_metrics():
    """
    Calculate 5-year cumulative metrics for all banks.

    Returns:
        DataFrame with:
            - total_dividends_5y: Sum of dividends (EUR)
            - total_buybacks_5y: Sum of buybacks (EUR)
            - total_payout_5y: Combined total (EUR)
            - avg_dividend_yield_5y: Average annual dividend yield
            - avg_buyback_yield_5y: Average annual buyback yield
            - avg_payout_yield_5y: Average annual total yield
            - avg_payout_ratio_5y: Average annual payout ratio
            - dividend_cagr_5y: 5-year dividend CAGR
            - buyback_years: Number of years with buybacks
            - consistency_score: Buyback consistency (0-1)
    """
    if not os.path.exists(DB_NAME):
        return pd.DataFrame()

    conn = sqlite3.connect(DB_NAME)

    query = """
        SELECT
            i.lei,
            i.name,
            i.ticker,
            i.country,
            i.region,
            i.size_category,
            SUM(f.dividend_amt) as total_dividends_5y,
            SUM(f.buyback_amt) as total_buybacks_5y,
            SUM(f.dividend_amt + COALESCE(f.buyback_amt, 0)) as total_payout_5y,
            AVG(f.dividend_yield_fy) as avg_dividend_yield_5y,
            AVG(f.buyback_yield_fy) as avg_buyback_yield_5y,
            AVG(f.total_yield_fy) as avg_payout_yield_5y,
            AVG(f.payout_ratio_fy) as avg_payout_ratio_5y,
            SUM(CASE WHEN f.buyback_amt > 0 THEN 1 ELSE 0 END) as buyback_years,
            COUNT(f.fy) as total_years
        FROM institutions i
        JOIN market_financial_years f ON i.lei = f.lei
        WHERE f.fy >= (SELECT MAX(fy) - 4 FROM market_financial_years)
        GROUP BY i.lei
        ORDER BY i.name
    """

    try:
        df = pd.read_sql_query(query, conn)

        # Calculate consistency score (% of years with buybacks)
        df['consistency_score'] = df['buyback_years'] / df['total_years']

        # Calculate dividend CAGR
        df['dividend_cagr_5y'] = df.apply(
            lambda row: calculate_dividend_cagr_for_lei(conn, row['lei']),
            axis=1
        )

        conn.close()
        return df
    except Exception as e:
        print(f"Error calculating cumulative metrics: {e}")
        conn.close()
        return pd.DataFrame()


def calculate_dividend_cagr_for_lei(conn, lei: str) -> float:
    """
    Calculate 5-year dividend CAGR for a specific bank.

    Args:
        conn: Database connection
        lei: Legal Entity Identifier

    Returns:
        CAGR as decimal (e.g., 0.15 = 15% growth)
    """
    try:
        query = """
            SELECT fy, dps_fy
            FROM market_financial_years
            WHERE lei = ?
            AND fy >= (SELECT MAX(fy) - 4 FROM market_financial_years WHERE lei = ?)
            ORDER BY fy
        """

        df = pd.read_sql_query(query, conn, params=(lei, lei))

        if len(df) < 2:
            return None

        # Get first and last dividend per share
        first_dps = df.iloc[0]['dps_fy']
        last_dps = df.iloc[-1]['dps_fy']
        years = len(df) - 1

        if first_dps is None or last_dps is None or first_dps <= 0:
            return None

        # CAGR formula: (Ending Value / Beginning Value)^(1/years) - 1
        cagr = (last_dps / first_dps) ** (1 / years) - 1

        return cagr
    except Exception as e:
        return None


@cache_decorator
def calculate_total_shareholder_return(years: int = 5):
    """
    Calculate Total Shareholder Return (price appreciation + dividends).

    Args:
        years: Number of years to look back (1, 3, or 5)

    Returns:
        DataFrame with TSR metrics
    """
    if not os.path.exists(DB_NAME):
        return pd.DataFrame()

    conn = sqlite3.connect(DB_NAME)

    # Get current prices and historical prices
    query = f"""
        SELECT
            i.lei,
            i.name,
            i.ticker,
            md.current_price,
            md.market_cap
        FROM institutions i
        JOIN market_data md ON i.lei = md.lei
        WHERE i.ticker IS NOT NULL
    """

    try:
        df = pd.read_sql_query(query, conn)

        # Calculate price return for each bank
        df[f'price_return_{years}y'] = df.apply(
            lambda row: calculate_price_return(conn, row['lei'], years),
            axis=1
        )

        # Get dividend sum for the period
        df[f'total_dividends_{years}y_eur'] = df.apply(
            lambda row: get_total_dividends_period(conn, row['lei'], years),
            axis=1
        )

        # Calculate dividend return (dividends / initial market cap)
        df[f'dividend_return_{years}y'] = (
            df[f'total_dividends_{years}y_eur'] / df['market_cap']
        )

        # Total Shareholder Return = Price Return + Dividend Return
        df[f'tsr_{years}y'] = (
            df[f'price_return_{years}y'].fillna(0) +
            df[f'dividend_return_{years}y'].fillna(0)
        )

        conn.close()
        return df
    except Exception as e:
        print(f"Error calculating TSR: {e}")
        conn.close()
        return pd.DataFrame()


def calculate_price_return(conn, lei: str, years: int) -> float:
    """Calculate price return over specified period."""
    try:
        days = years * 365

        query = """
            SELECT close
            FROM market_history
            WHERE lei = ?
            AND date <= date('now', '-' || ? || ' days')
            ORDER BY date DESC
            LIMIT 1
        """

        result = conn.execute(query, (lei, days)).fetchone()

        if not result:
            return None

        historical_price = result[0]

        # Get current price
        current_query = "SELECT current_price FROM market_data WHERE lei = ?"
        current_result = conn.execute(current_query, (lei,)).fetchone()

        if not current_result or not historical_price:
            return None

        current_price = current_result[0]

        # Calculate return
        price_return = (current_price - historical_price) / historical_price

        return price_return
    except Exception as e:
        return None


def get_total_dividends_period(conn, lei: str, years: int) -> float:
    """Get total dividends paid over specified period in EUR."""
    try:
        query = """
            SELECT SUM(amount_eur)
            FROM dividend_history
            WHERE lei = ?
            AND ex_date >= date('now', '-' || ? || ' years')
        """

        result = conn.execute(query, (lei, years)).fetchone()

        if result and result[0]:
            return result[0]
        return 0.0
    except Exception as e:
        return 0.0
