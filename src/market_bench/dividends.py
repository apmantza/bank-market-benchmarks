"""
Dividend analysis and shareholder remuneration utilities.

Functions for fetching, classifying, and analyzing dividend and buyback data
from Yahoo Finance for European banks.
"""

import pandas as pd
import numpy as np
import sqlite3
import yfinance as yf
from datetime import datetime
from .config import DB_NAME
from .data import get_fx_rate

# Conditional streamlit import for caching
try:
    import streamlit as st
    cache_decorator = st.cache_data(ttl=3600)
except:
    # Fallback when running as script (no streamlit)
    def cache_decorator(func):
        return func


def get_currency_for_ticker(ticker: str) -> str:
    """
    Determine currency for a given ticker.

    Args:
        ticker: Yahoo Finance ticker symbol

    Returns:
        Currency code (e.g., 'EUR', 'GBp', 'PLN')
    """
    # Map exchange suffixes to currencies
    currency_map = {
        '.MC': 'EUR',    # Madrid
        '.MI': 'EUR',    # Milan
        '.PA': 'EUR',    # Paris
        '.AS': 'EUR',    # Amsterdam
        '.BR': 'EUR',    # Brussels
        '.AT': 'EUR',    # Athens
        '.VI': 'EUR',    # Vienna
        '.LS': 'EUR',    # Lisbon
        '.L': 'GBp',     # London (pence)
        '.SW': 'CHF',    # Swiss
        '.CO': 'DKK',    # Copenhagen
        '.ST': 'SEK',    # Stockholm
        '.OL': 'NOK',    # Oslo
        '.WA': 'PLN',    # Warsaw
        '.PR': 'CZK',    # Prague
        '.BD': 'HUF',    # Budapest
        '.RO': 'RON',    # Bucharest
        '.IC': 'ISK',    # Iceland
        '.XC': 'EUR',    # XETRA
        '.DE': 'EUR',    # Deutsche Boerse
    }

    for suffix, currency in currency_map.items():
        if ticker.endswith(suffix):
            return currency

    # Default to EUR if unknown
    return 'EUR'


def fetch_dividend_history(ticker: str, lei: str) -> pd.DataFrame:
    """
    Fetch complete dividend history from yfinance.

    Args:
        ticker: Yahoo Finance ticker symbol
        lei: Legal Entity Identifier

    Returns:
        DataFrame with columns:
            - lei: Legal Entity Identifier
            - ex_date: Ex-dividend date
            - amount_local: Dividend in local currency
            - amount_eur: Converted to EUR
            - currency: Original currency
            - fx_rate: Exchange rate used
            - fx_date: Date of FX rate (same as ex_date)
    """
    try:
        stock = yf.Ticker(ticker)
        divs = stock.dividends

        if divs is None or len(divs) == 0:
            return pd.DataFrame()

        # Convert to DataFrame
        df = divs.to_frame('amount_local')
        df.reset_index(inplace=True)
        df.rename(columns={df.columns[0]: 'ex_date'}, inplace=True)

        # Add LEI
        df['lei'] = lei

        # Determine currency
        currency = get_currency_for_ticker(ticker)
        df['currency'] = currency

        # Handle GBp (pence) - dividends come in pence
        if currency == 'GBp':
            # Keep amount_local in pence, but note for user
            pass

        # Get FX rate and convert to EUR
        fx_rate = get_fx_rate(currency)

        if fx_rate is None:
            print(f"Warning: Could not get FX rate for {currency}. Using 1.0")
            fx_rate = 1.0

        df['fx_rate'] = fx_rate
        df['fx_date'] = df['ex_date']  # Use ex_date as FX date

        # Calculate EUR amount
        if currency == 'GBp':
            # Convert pence to pounds first, then to EUR
            df['amount_eur'] = (df['amount_local'] / 100.0) * fx_rate
        else:
            df['amount_eur'] = df['amount_local'] * fx_rate

        # Convert ex_date to string format for SQLite
        df['ex_date'] = df['ex_date'].dt.strftime('%Y-%m-%d')
        df['fx_date'] = df['fx_date'].dt.strftime('%Y-%m-%d')

        # Reorder columns
        df = df[['lei', 'ex_date', 'amount_local', 'amount_eur', 'currency', 'fx_rate', 'fx_date']]

        return df

    except Exception as e:
        print(f"Error fetching dividends for {ticker}: {str(e)[:100]}")
        return pd.DataFrame()



def extract_buybacks(ticker: str) -> pd.DataFrame:
    """
    Extract buyback amounts from cashflow statements.

    Args:
        ticker: Yahoo Finance ticker symbol

    Returns:
        DataFrame with:
            - date: Period end date
            - buyback_amt: Absolute amount in local currency
            - granularity: 'quarterly' or 'annual'
    """
    try:
        stock = yf.Ticker(ticker)
        buybacks = []

        # Try quarterly cashflow first
        try:
            qcf = stock.quarterly_cashflow
            if qcf is not None and not qcf.empty:
                if 'Repurchase Of Capital Stock' in qcf.index:
                    bb_row = qcf.loc['Repurchase Of Capital Stock']
                    for date, amount in bb_row.items():
                        if pd.notna(amount) and amount != 0:
                            buybacks.append({
                                'date': date.strftime('%Y-%m-%d'),
                                'buyback_amt': abs(amount),  # Convert to positive
                                'granularity': 'quarterly'
                            })
        except Exception as e:
            print(f"Quarterly cashflow error for {ticker}: {str(e)[:50]}")

        # Try annual cashflow
        try:
            acf = stock.cashflow
            if acf is not None and not acf.empty:
                if 'Repurchase Of Capital Stock' in acf.index:
                    bb_row = acf.loc['Repurchase Of Capital Stock']
                    for date, amount in bb_row.items():
                        if pd.notna(amount) and amount != 0:
                            buybacks.append({
                                'date': date.strftime('%Y-%m-%d'),
                                'buyback_amt': abs(amount),
                                'granularity': 'annual'
                            })
        except Exception as e:
            print(f"Annual cashflow error for {ticker}: {str(e)[:50]}")

        if not buybacks:
            return pd.DataFrame()

        df = pd.DataFrame(buybacks)

        # Remove duplicates (prefer quarterly over annual for same period)
        df = df.sort_values(['date', 'granularity'])
        df = df.drop_duplicates(subset=['date'], keep='first')

        return df

    except Exception as e:
        print(f"Error extracting buybacks for {ticker}: {str(e)[:100]}")
        return pd.DataFrame()


def get_share_actions(ticker: str, lei: str) -> pd.DataFrame:
    """
    Extract stock splits and share changes from yfinance.

    Args:
        ticker: Yahoo Finance ticker symbol
        lei: Legal Entity Identifier

    Returns:
        DataFrame with:
            - lei: Legal Entity Identifier
            - action_date: Date of action
            - action_type: 'split', 'reverse_split'
            - ratio: Split ratio (e.g., 2.0 for 2:1 split)
            - shares_affected: NULL for splits (calculated from balance sheet)
    """
    try:
        stock = yf.Ticker(ticker)
        actions = stock.splits

        if actions is None or len(actions) == 0:
            return pd.DataFrame()

        # Convert to DataFrame
        df = actions.to_frame('ratio')
        df.reset_index(inplace=True)
        df.rename(columns={df.columns[0]: 'action_date'}, inplace=True)

        # Add LEI
        df['lei'] = lei

        # Classify split type
        df['action_type'] = df['ratio'].apply(
            lambda x: 'split' if x > 1.0 else 'reverse_split'
        )

        # Shares affected is NULL (calculated from balance sheet separately)
        df['shares_affected'] = None

        # Convert date to string
        df['action_date'] = df['action_date'].dt.strftime('%Y-%m-%d')

        # Reorder columns
        df = df[['lei', 'action_date', 'action_type', 'ratio', 'shares_affected']]

        return df

    except Exception as e:
        print(f"Error extracting share actions for {ticker}: {str(e)[:100]}")
        return pd.DataFrame()


def upsert_dividend_history(lei: str, dividends: pd.DataFrame):
    """
    Insert or update dividend history in database.

    Args:
        lei: Legal Entity Identifier
        dividends: DataFrame with dividend data (must include dividend_type)
    """
    if dividends.empty:
        return

    conn = sqlite3.connect(DB_NAME)

    for _, row in dividends.iterrows():
        conn.execute("""
            INSERT INTO dividend_history
            (lei, ex_date, payment_date, amount_local, amount_eur, currency,
             fx_rate, fx_date, dividend_type)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(lei, ex_date) DO UPDATE SET
                payment_date=excluded.payment_date,
                amount_local=excluded.amount_local,
                amount_eur=excluded.amount_eur,
                currency=excluded.currency,
                fx_rate=excluded.fx_rate,
                fx_date=excluded.fx_date,
                dividend_type=excluded.dividend_type
        """, (
            row['lei'],
            row['ex_date'],
            row.get('payment_date'),
            row['amount_local'],
            row['amount_eur'],
            row['currency'],
            row['fx_rate'],
            row['fx_date'],
            row.get('dividend_type', 'regular')
        ))

    conn.commit()
    conn.close()


def upsert_share_actions(lei: str, actions: pd.DataFrame):
    """
    Insert or update share actions in database.

    Args:
        lei: Legal Entity Identifier
        actions: DataFrame with share action data
    """
    if actions.empty:
        return

    conn = sqlite3.connect(DB_NAME)

    for _, row in actions.iterrows():
        conn.execute("""
            INSERT INTO share_actions
            (lei, action_date, action_type, ratio, shares_affected)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(lei, action_date, action_type) DO UPDATE SET
                ratio=excluded.ratio,
                shares_affected=excluded.shares_affected
        """, (
            row['lei'],
            row['action_date'],
            row['action_type'],
            row['ratio'],
            row['shares_affected']
        ))

    conn.commit()
    conn.close()


@cache_decorator
def get_dividend_history(lei: str = None) -> pd.DataFrame:
    """
    Retrieve dividend history from database.

    Args:
        lei: Optional LEI to filter by. If None, returns all.

    Returns:
        DataFrame with dividend history
    """
    conn = sqlite3.connect(DB_NAME)

    if lei:
        query = """
            SELECT dh.*, i.name, i.ticker
            FROM dividend_history dh
            JOIN institutions i ON dh.lei = i.lei
            WHERE dh.lei = ?
            ORDER BY dh.ex_date DESC
        """
        df = pd.read_sql_query(query, conn, params=(lei,))
    else:
        query = """
            SELECT dh.*, i.name, i.ticker
            FROM dividend_history dh
            JOIN institutions i ON dh.lei = i.lei
            ORDER BY i.name, dh.ex_date DESC
        """
        df = pd.read_sql_query(query, conn)

    conn.close()
    return df


@cache_decorator
def get_share_actions_history(lei: str = None) -> pd.DataFrame:
    """
    Retrieve share actions from database.

    Args:
        lei: Optional LEI to filter by. If None, returns all.

    Returns:
        DataFrame with share action history
    """
    conn = sqlite3.connect(DB_NAME)

    if lei:
        query = """
            SELECT sa.*, i.name, i.ticker
            FROM share_actions sa
            JOIN institutions i ON sa.lei = i.lei
            WHERE sa.lei = ?
            ORDER BY sa.action_date DESC
        """
        df = pd.read_sql_query(query, conn, params=(lei,))
    else:
        query = """
            SELECT sa.*, i.name, i.ticker
            FROM share_actions sa
            JOIN institutions i ON sa.lei = i.lei
            ORDER BY i.name, sa.action_date DESC
        """
        df = pd.read_sql_query(query, conn)

    conn.close()
    return df
