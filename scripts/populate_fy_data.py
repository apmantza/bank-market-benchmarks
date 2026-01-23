#!/usr/bin/env python3
"""
Populate market_financial_years with comprehensive shareholder remuneration data.

This script:
1. Fetches net income from yfinance financials
2. Aggregates dividends from dividend_history table by fiscal year
3. Fetches buybacks from cash flow statements
4. Calculates all derived metrics (payout ratios, yields, splits)
5. Updates market_financial_years table

Usage:
    python scripts/populate_fy_data.py                  # All banks
    python scripts/populate_fy_data.py --ticker BBVA.MC # Single bank
    python scripts/populate_fy_data.py --validate       # Run BBVA validation
"""

import argparse
import sqlite3
import sys
import io
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np

# Set UTF-8 output encoding for Windows console
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import yfinance as yf
from market_bench.config import DB_NAME
from market_bench.data import get_fx_rate

# Currency map for ticker suffixes
CURRENCY_MAP = {
    '.MC': 'EUR', '.MI': 'EUR', '.PA': 'EUR', '.AS': 'EUR', '.BR': 'EUR',
    '.AT': 'EUR', '.VI': 'EUR', '.XC': 'EUR', '.DE': 'EUR', '.LS': 'EUR',
    '.L': 'GBp', '.SW': 'CHF', '.CO': 'DKK', '.ST': 'SEK', '.OL': 'NOK',
    '.WA': 'PLN', '.PR': 'CZK', '.BD': 'HUF', '.RO': 'RON', '.IC': 'ISK',
}

def get_currency_for_ticker(ticker: str) -> str:
    """Get currency for a ticker based on exchange suffix."""
    for suffix, currency in CURRENCY_MAP.items():
        if ticker.endswith(suffix):
            return currency
    return 'EUR'


def fetch_net_income_by_fy(ticker: str) -> dict:
    """
    Fetch annual net income from yfinance financials.
    
    Returns:
        dict: {2021: amount_eur, 2022: amount_eur, ...}
    """
    try:
        stock = yf.Ticker(ticker)
        financials = stock.financials
        
        if financials is None or financials.empty:
            return {}
        
        # Get currency and FX rate
        currency = get_currency_for_ticker(ticker)
        fx_rate = get_fx_rate(currency) or 1.0
        
        # Handle GBp (pence) - financials are in GBP (pounds)
        # No need for pence conversion on financials
        
        result = {}
        
        # Look for Net Income row
        net_income_labels = ['Net Income', 'Net Income Common Stockholders', 
                            'Net Income From Continuing Operations']
        
        for label in net_income_labels:
            if label in financials.index:
                row = financials.loc[label]
                for date, value in row.items():
                    if pd.notna(value) and value != 0:
                        fy = date.year
                        amount_eur = float(value) * fx_rate
                        result[fy] = amount_eur
                break
        
        return result
        
    except Exception as e:
        print(f"    Error fetching net income for {ticker}: {str(e)[:80]}")
        return {}


def fetch_ttm_net_income(ticker: str) -> float:
    """Fetch Trailing Twelve Month net income."""
    try:
        stock = yf.Ticker(ticker)
        
        # 1. Try info dict for reliable TTM
        info = stock.info
        ttm_ni = info.get('netIncomeToCommon') or info.get('netIncome')
        if ttm_ni and ttm_ni > 0:
            # Info financials are usually in ticker's local currency, but let's assume same as financials
            # Need to verify if yf.info is ALWAYS in local currency like financials
            currency = get_currency_for_ticker(ticker)
            fx_rate = get_fx_rate(currency) or 1.0
            return float(ttm_ni) * fx_rate
            
        # 2. Fallback to quarterly financials sum
        q_financials = stock.quarterly_financials
        if q_financials is None or q_financials.empty:
            return 0
            
        currency = get_currency_for_ticker(ticker)
        fx_rate = get_fx_rate(currency) or 1.0
        
        # Labels
        net_income_labels = ['Net Income', 'Net Income Common Stockholders', 
                            'Net Income From Continuing Operations']
                            
        for label in net_income_labels:
            if label in q_financials.index:
                row = q_financials.loc[label]
                # Filter out NaNs and take last 4 quarters
                valid_vals = row.dropna().head(4)
                if len(valid_vals) > 0:
                    ttm_sum = valid_vals.sum()
                    return float(ttm_sum) * fx_rate
        return 0
    except:
        return 0


def fetch_buybacks_by_fy(ticker: str) -> dict:
    """
    Fetch buybacks from annual and quarterly cash flow statements.
    
    Returns:
        dict: {2021: amount_eur, 2022: amount_eur, ...}
    """
    try:
        stock = yf.Ticker(ticker)
        
        # Get currency and FX rate
        currency = get_currency_for_ticker(ticker)
        fx_rate = get_fx_rate(currency) or 1.0
        
        # Labels to look for - prioritize specific repurchase labels
        buyback_labels = ['Repurchase Of Capital Stock', 'Repurchase Of Common Stock', 'Purchase Of Stock']
        
        # Banks where 'Net Common Stock Issuance' is a known reliable proxy for buybacks
        fallback_tickers = ['UCG.MI', 'INGA.AS', 'ISP.MI', 'SAN.MC']
        if ticker in fallback_tickers:
            buyback_labels.append('Net Common Stock Issuance')
        
        result = {}
        
        # 1. Start with Annual Data
        cashflow = stock.cashflow
        if cashflow is not None and not cashflow.empty:
            for label in buyback_labels:
                if label in cashflow.index:
                    row = cashflow.loc[label]
                    found_any = False
                    for date, value in row.items():
                        if pd.notna(value) and value != 0:
                            fy = date.year
                            # Buybacks are negative in cash flow (outflow)
                            amount_eur = abs(float(value)) * fx_rate if value < 0 else 0
                            if amount_eur > 0:
                                result[fy] = amount_eur
                                found_any = True
                    if found_any:
                        break # Only break if we found actual data for this label
        
        # 2. Augment with Quarterly Data (for years missing in annual)
        q_cashflow = stock.quarterly_cashflow
        if q_cashflow is not None and not q_cashflow.empty:
            q_result = {}
            for label in buyback_labels:
                if label in q_cashflow.index:
                    row = q_cashflow.loc[label]
                    for date, value in row.items():
                        if pd.notna(value) and value != 0:
                            year = date.year
                            # Keep track of quarters per year
                            amount_eur = abs(float(value)) * fx_rate if value < 0 else 0
                            if amount_eur > 0:
                                if year not in q_result:
                                    q_result[year] = 0
                                q_result[year] += amount_eur
                    break
            
            # Merge: If year in q_result has more or unique data, prefer it for recent years
            for year, q_amount in q_result.items():
                if year not in result or (year >= datetime.now().year - 1 and q_amount > result.get(year, 0)):
                    result[year] = q_amount
                    
        return result
        
    except Exception as e:
        print(f"    Error fetching buybacks for {ticker}: {str(e)[:80]}")
        return {}


def fetch_ttm_buybacks(ticker: str) -> float:
    """Fetch Trailing Twelve Month buybacks from quarterly cash flow."""
    try:
        stock = yf.Ticker(ticker)
        q_cf = stock.quarterly_cashflow
        
        if q_cf is None or q_cf.empty:
            return 0
            
        # Get currency and FX rate
        currency = get_currency_for_ticker(ticker)
        fx_rate = get_fx_rate(currency) or 1.0
        
        buyback_labels = ['Repurchase Of Capital Stock', 'Repurchase Of Common Stock', 'Purchase Of Stock']
        
        fallback_tickers = ['UCG.MI', 'INGA.AS', 'ISP.MI', 'SAN.MC']
        if ticker in fallback_tickers:
            buyback_labels.append('Net Common Stock Issuance')
                         
        for label in buyback_labels:
            if label in q_cf.index:
                row = q_cf.loc[label]
                valid_vals = row.dropna().head(4)
                # Filter for actual negative values (outflows)
                actual_buybacks = valid_vals[valid_vals < 0]
                if len(actual_buybacks) > 0:
                    ttm_sum = abs(actual_buybacks.sum())
                    return float(ttm_sum) * fx_rate
        return 0
    except:
        return 0


def fetch_tangible_bv_by_fy(ticker: str) -> dict:
    """Fetch Tangible Book Value from balance sheet."""
    try:
        stock = yf.Ticker(ticker)
        bs = stock.balance_sheet
        if bs is None or bs.empty:
            return {}
            
        currency = get_currency_for_ticker(ticker)
        fx_rate = get_fx_rate(currency) or 1.0
        
        result = {}
        if 'Tangible Book Value' in bs.index:
            row = bs.loc['Tangible Book Value']
            for date, val in row.items():
                if pd.notna(val):
                    result[date.year] = float(val) * fx_rate
        elif 'Common Stock Equity' in bs.index: # Fallback to Common Equity if TBV missing
            row = bs.loc['Common Stock Equity']
            for date, val in row.items():
                 if pd.notna(val):
                    result[date.year] = float(val) * fx_rate
                    
        return result
    except:
        return {}


def fetch_prices_by_fy(ticker: str) -> dict:
    """Fetch start/end prices for each fiscal year."""
    try:
        stock = yf.Ticker(ticker)
        # Fetch data back to 2020 to get 2021 start
        hist = stock.history(start="2020-01-01")
        if hist.empty:
            return {}
            
        currency = get_currency_for_ticker(ticker)
        fx_rate = get_fx_rate(currency) or 1.0
        # If GBp, convert to GBP by dividing by 100
        if currency == 'GBp':
            fx_rate /= 100.0
            
        result = {}
        years = sorted(hist.index.year.unique())
        
        for year in years:
            if year < 2021: continue
            
            # Start price: Close of last day of previous year
            prev_year_data = hist[hist.index.year == year - 1]
            if not prev_year_data.empty:
                p_start = prev_year_data.iloc[-1]['Close']
            else:
                # Fallback to first day of current year
                year_data = hist[hist.index.year == year]
                p_start = year_data.iloc[0]['Close'] if not year_data.empty else 0
                
            # End price: Close of last day of current year
            year_data = hist[hist.index.year == year]
            if not year_data.empty:
                p_end = year_data.iloc[-1]['Close']
            else:
                p_end = 0
            
            if p_start > 0 and p_end > 0:
                result[year] = {
                    'start': float(p_start) * fx_rate,
                    'end': float(p_end) * fx_rate
                }
        return result
    except:
        return {}


def fetch_shares_by_fy(ticker: str) -> dict:
    """
    Fetch historical shares outstanding by fiscal year from balance sheet.
    
    Uses 'Ordinary Shares Number' from balance sheet for accurate historical data.
    Falls back to current shares if historical data not available.
    
    Returns:
        dict: {2021: shares, 2022: shares, ...}
    """
    try:
        stock = yf.Ticker(ticker)
        balance = stock.balance_sheet
        
        result = {}
        
        if balance is not None and not balance.empty:
            # Look for shares row - prioritize 'Ordinary Shares Number'
            shares_labels = ['Ordinary Shares Number', 'Share Issued', 
                            'Common Stock Shares Outstanding', 'Common Stock']
            
            for label in shares_labels:
                if label in balance.index:
                    row = balance.loc[label]
                    for date, value in row.items():
                        if pd.notna(value) and value > 0:
                            fy = date.year
                            result[fy] = float(value)
                    break
        
        # If no balance sheet data, try income statement for average shares
        if not result:
            financials = stock.financials
            if financials is not None and not financials.empty:
                if 'Basic Average Shares' in financials.index:
                    row = financials.loc['Basic Average Shares']
                    for date, value in row.items():
                        if pd.notna(value) and value > 0:
                            fy = date.year
                            result[fy] = float(value)
        
        # Final fallback to current shares
        if not result:
            info = stock.info
            current_shares = info.get('sharesOutstanding')
            if current_shares:
                result = {datetime.now().year: current_shares}
        
        return result
        
    except Exception as e:
        print(f"    Error fetching shares for {ticker}: {str(e)[:80]}")
        return {}


def fetch_avg_price_by_fy(ticker: str) -> dict:
    """
    Calculate average price for each fiscal year from historical daily prices.
    
    Returns:
        dict: {2021: avg_price_eur, 2022: avg_price_eur, ...}
    """
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="5y")
        
        if hist is None or hist.empty:
            return {}
        
        # Get currency and FX rate
        currency = get_currency_for_ticker(ticker)
        fx_rate = get_fx_rate(currency) or 1.0
        price_divisor = 100.0 if currency == 'GBp' else 1.0
        
        result = {}
        
        # Group by year and calculate average
        hist['Year'] = hist.index.year
        yearly_avg = hist.groupby('Year')['Close'].mean()
        
        for fy, avg_price in yearly_avg.items():
            avg_price_eur = (avg_price / price_divisor) * fx_rate
            result[fy] = avg_price_eur
        
        return result
        
    except Exception as e:
        print(f"    Error calculating avg price for {ticker}: {str(e)[:80]}")
        return {}


def fetch_avg_market_cap_by_fy(ticker: str, shares_by_fy: dict = None) -> dict:
    """
    Calculate average market cap for each fiscal year.
    
    Formula: Avg Market Cap = Avg Daily Price × Shares Outstanding (for that year)
    
    Uses historical shares from balance sheet for accurate calculation.
    
    Args:
        ticker: Yahoo Finance ticker
        shares_by_fy: Optional pre-fetched shares by FY (to avoid duplicate API calls)
    
    Returns:
        dict: {2021: avg_market_cap_eur, 2022: avg_market_cap_eur, ...}
    """
    try:
        # Get historical prices
        avg_prices = fetch_avg_price_by_fy(ticker)
        if not avg_prices:
            return {}
        
        # Get historical shares (or use provided)
        if shares_by_fy is None:
            shares_by_fy = fetch_shares_by_fy(ticker)
        
        if not shares_by_fy:
            # Fall back to current shares
            stock = yf.Ticker(ticker)
            current_shares = stock.info.get('sharesOutstanding', 0)
            if current_shares == 0:
                return {}
            shares_by_fy = {y: current_shares for y in avg_prices.keys()}
        
        result = {}
        
        for fy, avg_price in avg_prices.items():
            # Get shares for this FY, or closest available year
            shares = shares_by_fy.get(fy)
            if not shares and shares_by_fy:
                # Use closest year's shares
                available_years = sorted(shares_by_fy.keys())
                closest = min(available_years, key=lambda x: abs(x - fy))
                shares = shares_by_fy[closest]
            
            if shares and shares > 0:
                result[fy] = avg_price * shares
        
        return result
        
    except Exception as e:
        print(f"    Error calculating avg market cap for {ticker}: {str(e)[:80]}")
        return {}


def fetch_ttm_dividends_for_lei(lei: str, ticker: str, conn) -> dict:
    """
    Calculate TTM dividends based on payments in the last 365 days.
    """
    try:
        # We need per-share amounts from dividend_history
        query = """
            SELECT ex_date, amount_eur
            FROM dividend_history
            WHERE lei = ? AND ex_date >= date('now', '-365 days')
        """
        df = pd.read_sql_query(query, conn, params=(lei,))
        
        if df.empty:
            return {'total': 0, 'dps': 0}
            
        # Get latest shares
        stock = yf.Ticker(ticker)
        shares = stock.info.get('sharesOutstanding', 0)
        
        if shares == 0:
            # Fallback to DB
            cursor = conn.cursor()
            cursor.execute("SELECT shares_outstanding FROM market_financial_years WHERE lei = ? ORDER BY fy DESC LIMIT 1", (lei,))
            row = cursor.fetchone()
            shares = row[0] if row else 0
            
        ttm_dps = df['amount_eur'].sum()
        ttm_total = ttm_dps * shares
        
        return {'total': ttm_total, 'dps': ttm_dps}
    except:
        return {'total': 0, 'dps': 0}


def aggregate_dividends_by_fy(lei: str, ticker: str, conn) -> dict:
    """
    Aggregate dividends from dividend_history by fiscal year.
    
    IMPORTANT: dividend_history stores per-share amounts (DPS).
    We need to multiply by shares outstanding to get total dividend amounts.
    
    Attribution Rule:
    - Dividends with ex_date in Jan-Jun of Year X → FY X-1 (supplementary)
    - Dividends with ex_date in Jul-Dec of Year X → FY X (interim)
    
    Returns:
        dict: {2021: {'total': X, 'regular': Y, 'special': Z, 'dps': D}, ...}
    """
    query = """
        SELECT ex_date, amount_eur, dividend_type
        FROM dividend_history
        WHERE lei = ?
        ORDER BY ex_date
    """
    
    df = pd.read_sql_query(query, conn, params=(lei,))
    
    if df.empty:
        return {}
    
    # Get shares by FY for conversion
    shares_by_fy = fetch_shares_by_fy(ticker)
    
    # Fallback to current shares if no historical data
    if not shares_by_fy:
        try:
            stock = yf.Ticker(ticker)
            current_shares = stock.info.get('sharesOutstanding', 0)
            if current_shares > 0:
                shares_by_fy = {y: current_shares for y in range(2015, 2030)}
        except:
            pass
    
    result = {}
    
    for _, row in df.iterrows():
        ex_date = datetime.strptime(row['ex_date'], '%Y-%m-%d')
        dps = row['amount_eur'] or 0  # This is per-share amount
        div_type = row['dividend_type'] or 'regular'
        
        # Attribution Rule:
        # Jan-July (Months 1-7) -> attributed to previous FY (Settlement of Final dividends)
        # Aug-Dec (Months 8-12) -> attributed to current FY (Interim / Early distributions)
        if ex_date.month <= 7:
            fy = ex_date.year - 1
        else:
            fy = ex_date.year
        
        if fy not in result:
            result[fy] = {'total': 0, 'dps': 0}
        
        # Get shares for this FY (or closest available)
        shares = shares_by_fy.get(fy)
        if not shares and shares_by_fy:
            # Use closest year's shares
            available_years = sorted(shares_by_fy.keys())
            closest = min(available_years, key=lambda x: abs(x - fy))
            shares = shares_by_fy[closest]
        shares = shares or 0
        
        # Convert per-share to total amount
        total_amount = dps * shares
        
        result[fy]['total'] += total_amount
        result[fy]['dps'] += dps  # Keep track of DPS too
    
    return result


def calculate_fy_metrics(net_income: float, dividends: float, buybacks: float,
                         avg_market_cap: float, shares: float, tangible_bv: float = 0,
                         price_start: float = 0, price_end: float = 0) -> dict:
    """
    Calculate all derived FY metrics.
    """
    total_payout = dividends + buybacks
    
    metrics = {
        'dividend_amt': dividends,
        'buyback_amt': buybacks,
        'net_income': net_income,
        'avg_market_cap': avg_market_cap,
        'shares_outstanding': shares,
        'tangible_book_value': tangible_bv,
        'price_start': price_start,
        'price_end': price_end
    }
    
    # Payout ratios (only if net income is positive)
    if net_income and net_income > 0:
        metrics['payout_ratio'] = total_payout / net_income
    else:
        metrics['payout_ratio'] = None
    
    # Yields (only if market cap is positive)
    if avg_market_cap and avg_market_cap > 0:
        metrics['dividend_yield_fy'] = dividends / avg_market_cap
        metrics['buyback_yield_fy'] = buybacks / avg_market_cap
        metrics['total_yield_fy'] = total_payout / avg_market_cap
    else:
        metrics['dividend_yield_fy'] = None
        metrics['buyback_yield_fy'] = None
        metrics['total_yield_fy'] = None
    
    # Payout split
    if total_payout > 0:
        metrics['dividend_share_pct'] = dividends / total_payout
        metrics['buyback_share_pct'] = buybacks / total_payout
    else:
        metrics['dividend_share_pct'] = None
        metrics['buyback_share_pct'] = None
    
    # Per-share metrics
    if shares and shares > 0:
        metrics['eps_fy'] = net_income / shares if net_income else None
        metrics['dps_fy'] = dividends / shares if dividends else None
    else:
        metrics['eps_fy'] = None
        metrics['dps_fy'] = None
        
    # Analyst Metrics
    if tangible_bv and tangible_bv > 0:
        metrics['p_tbv'] = avg_market_cap / tangible_bv if avg_market_cap > 0 else None
        metrics['rote'] = net_income / tangible_bv if net_income > 0 else None
    else:
        metrics['p_tbv'] = None
        metrics['rote'] = None
        
    # Price Performance
    if price_start and price_start > 0 and price_end and price_end > 0:
        metrics['price_perf_fy'] = (price_end / price_start) - 1
        dps = metrics['dps_fy'] or 0
        metrics['total_return_fy'] = (price_end + dps - price_start) / price_start
    else:
        metrics['price_perf_fy'] = None
        metrics['total_return_fy'] = None
        
    return metrics


def populate_fy_data_for_bank(lei: str, ticker: str, name: str, conn, verbose: bool = False):
    """
    Populate market_financial_years for a single bank.
    """
    print(f"  Fetching financial data...")
    
    # Fetch all data sources - fetch shares first so we can reuse it
    shares_by_fy = fetch_shares_by_fy(ticker)
    net_income_by_fy = fetch_net_income_by_fy(ticker)
    buybacks_by_fy = fetch_buybacks_by_fy(ticker)
    avg_market_cap_by_fy = fetch_avg_market_cap_by_fy(ticker, shares_by_fy)  # Pass shares for accuracy
    dividends_by_fy = aggregate_dividends_by_fy(lei, ticker, conn)
    tbv_by_fy = fetch_tangible_bv_by_fy(ticker)
    prices_by_fy = fetch_prices_by_fy(ticker)
    
    if verbose:
        print(f"    Net Income FYs: {list(net_income_by_fy.keys())}")
        print(f"    Buybacks FYs: {list(buybacks_by_fy.keys())}")
        print(f"    Dividends FYs: {list(dividends_by_fy.keys())}")
        print(f"    TBV FYs: {list(tbv_by_fy.keys())}")
        print(f"    Price Data FYs: {list(prices_by_fy.keys())}")
    
    # Get all fiscal years we have data for
    all_fys = set()
    all_fys.update(net_income_by_fy.keys())
    all_fys.update(buybacks_by_fy.keys())
    all_fys.update(dividends_by_fy.keys())
    
    if not all_fys:
        print(f"    No fiscal year data found")
        return 0
    
    # Filter to reasonable range (last 7 years)
    current_year = datetime.now().year
    all_fys = {fy for fy in all_fys if current_year - 7 <= fy <= current_year}
    
    records_updated = 0
    
    for fy in sorted(all_fys):
        net_income = net_income_by_fy.get(fy, 0)
        buybacks = buybacks_by_fy.get(fy, 0)
        div_data = dividends_by_fy.get(fy, {'total': 0, 'regular': 0, 'special': 0, 'dps': 0})
        dividends = div_data['total']
        tbv = tbv_by_fy.get(fy, 0)
        
        # --- TTM HYBRID LOGIC FOR CURRENT YEAR (2025) ---
        if fy == 2025 and net_income == 0:
            ttm_ni = fetch_ttm_net_income(ticker)
            if ttm_ni > 0:
                net_income = ttm_ni
                if verbose: print(f"    FY 2025: Using TTM Net Income: €{net_income/1e9:.2f}B")
                
            ttm_bb = fetch_ttm_buybacks(ticker)
            if ttm_bb > 0 and buybacks == 0:
                buybacks = ttm_bb
                if verbose: print(f"    FY 2025: Using TTM Buybacks: €{buybacks/1e9:.2f}B")
                
            ttm_div = fetch_ttm_dividends_for_lei(lei, ticker, conn)
            if ttm_div['total'] > 0 and dividends == 0:
                dividends = ttm_div['total']
                div_data['dps'] = ttm_div['dps']
                if verbose: print(f"    FY 2025: Using TTM Dividends: €{dividends/1e9:.2f}B")
            
            # Fetch TTM TBV from info
            if tbv == 0:
                try:
                    stock = yf.Ticker(ticker)
                    info = stock.info
                    # Use 'bookValue' * shares as approx TBV if literal TBV not available
                    # Actually check balance sheet's last row first
                    if tbv_by_fy:
                        latest_year = max(tbv_by_fy.keys())
                        tbv = tbv_by_fy[latest_year]
                    elif 'bookValue' in info:
                        currency = get_currency_for_ticker(ticker)
                        fx_rate = get_fx_rate(currency) or 1.0
                        shares_now = info.get('sharesOutstanding', 0)
                        tbv = info['bookValue'] * shares_now * fx_rate
                except: pass

        price_data = prices_by_fy.get(fy, {'start': 0, 'end': 0})
        shares = shares_by_fy.get(fy, shares_by_fy.get(max(shares_by_fy.keys()), 0) if shares_by_fy else 0)
        avg_mkt_cap = avg_market_cap_by_fy.get(fy, 0)
        
        # For current year, if avg market cap is 0, use current market cap
        if fy == 2025 and avg_mkt_cap == 0:
            try:
                stock = yf.Ticker(ticker)
                avg_mkt_cap = stock.info.get('marketCap', 0)
            except: pass
        
        # Calculate metrics
        metrics = calculate_fy_metrics(net_income, dividends, buybacks, avg_mkt_cap, shares, tbv, 
                                       price_data['start'], price_data['end'])
        
        # Delete existing record for this LEI+FY, then insert
        conn.execute("DELETE FROM market_financial_years WHERE lei = ? AND fy = ?", (lei, fy))
        
        conn.execute("""
            INSERT INTO market_financial_years 
            (lei, ticker, fy, net_income, dividend_amt, buyback_amt, avg_market_cap,
             shares_outstanding, dividend_yield_fy, buyback_yield_fy, total_yield_fy,
             dividend_share_pct, buyback_share_pct, eps_fy, dps_fy, tangible_book_value, rote, p_tbv,
             price_start, price_end, price_perf_fy, total_return_fy)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            lei, ticker, fy,
            metrics['net_income'], metrics['dividend_amt'], metrics['buyback_amt'],
            metrics['avg_market_cap'], metrics['shares_outstanding'],
            metrics['dividend_yield_fy'], metrics['buyback_yield_fy'], metrics['total_yield_fy'],
            metrics['dividend_share_pct'], metrics['buyback_share_pct'],
            metrics['eps_fy'], metrics['dps_fy'],
            metrics['tangible_book_value'], metrics['rote'], metrics['p_tbv'],
            metrics['price_start'], metrics['price_end'], metrics['price_perf_fy'], metrics['total_return_fy']
        ))
        
        records_updated += 1
        
        if verbose:
            print(f"    FY {fy}: NI={net_income/1e9:.2f}B, Div={dividends/1e9:.2f}B, BB={buybacks/1e9:.2f}B")
    
    conn.commit()
    print(f"    [OK] Updated {records_updated} fiscal years")
    return records_updated


def populate_all_banks(verbose: bool = False):
    """Populate FY data for all banks with tickers."""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT lei, ticker, name
        FROM institutions
        WHERE ticker IS NOT NULL
        ORDER BY name
    """)
    banks = cursor.fetchall()
    
    total = len(banks)
    print(f"Populating FY data for {total} banks...\n")
    
    success = 0
    failed = 0
    
    for i, (lei, ticker, name) in enumerate(banks, 1):
        pct = (i / total) * 100
        print(f"[{i}/{total}] ({pct:.1f}%) {name} ({ticker})")
        
        try:
            records = populate_fy_data_for_bank(lei, ticker, name, conn, verbose)
            if records > 0:
                success += 1
            else:
                failed += 1
        except Exception as e:
            print(f"    [ERROR] {str(e)[:80]}")
            failed += 1
    
    conn.close()
    
    print(f"\n{'='*60}")
    print(f"Complete! Success: {success}, Failed: {failed}")
    print(f"{'='*60}")


def populate_single_bank(ticker: str, verbose: bool = False):
    """Populate FY data for a single bank."""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT lei, ticker, name
        FROM institutions
        WHERE ticker = ?
    """, (ticker,))
    
    bank = cursor.fetchone()
    
    if not bank:
        print(f"Ticker {ticker} not found in database")
        conn.close()
        return
    
    lei, ticker, name = bank
    print(f"Populating FY data for {name} ({ticker})\n")
    
    try:
        populate_fy_data_for_bank(lei, ticker, name, conn, verbose)
    except Exception as e:
        print(f"[ERROR] {str(e)}")
    
    conn.close()


def validate_bbva():
    """Validate BBVA data against ground truth."""
    print("="*60)
    print("BBVA VALIDATION - Comparing against ground truth")
    print("="*60)
    
    # Ground truth from bbva_final_comprehensive_report.md
    expected = {
        2021: {'div': 2.0e9, 'bb': 1.5e9, 'net': 4.65e9, 'payout': 0.43},
        2022: {'div': 2.75e9, 'bb': 2.08e9, 'net': 6.45e9, 'payout': 0.49},
        2023: {'div': 3.4e9, 'bb': 1.78e9, 'net': 8.02e9, 'payout': 0.52},
        2024: {'div': 4.1e9, 'bb': 0.993e9, 'net': 10.09e9, 'payout': 0.50},
    }
    
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT fy, dividend_amt, buyback_amt, net_income, 
               dividend_share_pct, buyback_share_pct
        FROM market_financial_years
        WHERE ticker = 'BBVA.MC' OR ticker = 'BBVA'
        ORDER BY fy
    """)
    
    rows = cursor.fetchall()
    conn.close()
    
    if not rows:
        print("\n[ERROR] No BBVA data found in market_financial_years!")
        print("Run: python scripts/populate_fy_data.py --ticker BBVA.MC")
        return
    
    print("\n{:<6} | {:>12} {:>12} | {:>12} {:>12} | {:>8} {:>8}".format(
        "FY", "Div (Exp)", "Div (Act)", "BB (Exp)", "BB (Act)", "Var Div", "Var BB"
    ))
    print("-" * 80)
    
    for row in rows:
        fy, div_act, bb_act, net_act, div_pct, bb_pct = row
        
        if fy not in expected:
            continue
        
        exp = expected[fy]
        div_exp = exp['div']
        bb_exp = exp['bb']
        
        div_var = ((div_act or 0) - div_exp) / div_exp * 100 if div_exp else 0
        bb_var = ((bb_act or 0) - bb_exp) / bb_exp * 100 if bb_exp else 0
        
        status_div = "✓" if abs(div_var) < 10 else "✗"
        status_bb = "✓" if abs(bb_var) < 10 else "✗"
        
        print("{:<6} | {:>10.2f}B {:>10.2f}B | {:>10.2f}B {:>10.2f}B | {:>+7.1f}% {:>+7.1f}%  {} {}".format(
            fy,
            div_exp / 1e9, (div_act or 0) / 1e9,
            bb_exp / 1e9, (bb_act or 0) / 1e9,
            div_var, bb_var,
            status_div, status_bb
        ))
    
    print("\nLegend: ✓ = within 10% tolerance, ✗ = outside tolerance")


def main():
    parser = argparse.ArgumentParser(
        description='Populate market_financial_years with shareholder remuneration data'
    )
    
    parser.add_argument('--ticker', help='Populate specific ticker only')
    parser.add_argument('--validate', action='store_true', help='Validate BBVA data')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    if args.validate:
        validate_bbva()
    elif args.ticker:
        populate_single_bank(args.ticker, args.verbose)
    else:
        populate_all_banks(args.verbose)


if __name__ == '__main__':
    main()
