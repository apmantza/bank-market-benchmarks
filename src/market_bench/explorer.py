"""
Data Explorer utilities for fetching and displaying raw yfinance data.
"""

import pandas as pd
import yfinance as yf
from datetime import datetime

# Conditional streamlit import for caching
try:
    import streamlit as st
    cache_decorator = st.cache_data(ttl=3600)
except:
    def cache_decorator(func):
        return func


@cache_decorator
def get_raw_yfinance_data(ticker: str, data_type: str) -> pd.DataFrame:
    """
    Fetch raw data from yfinance without processing.

    Args:
        ticker: Yahoo Finance ticker symbol
        data_type: One of 'history', 'dividends', 'splits', 'actions',
                   'quarterly_cashflow', 'cashflow', 'quarterly_income',
                   'income_stmt', 'quarterly_balance', 'balance_sheet', 'info'

    Returns:
        DataFrame with raw data, or None if error/not available
    """
    try:
        stock = yf.Ticker(ticker)

        if data_type == 'history':
            # Price and volume history (max available)
            df = stock.history(period="max")
            if df is not None and not df.empty:
                df.reset_index(inplace=True)
                return df
            return pd.DataFrame()

        elif data_type == 'dividends':
            # Dividend history
            divs = stock.dividends
            if divs is not None and len(divs) > 0:
                df = divs.to_frame('Dividend')
                df.reset_index(inplace=True)
                df.rename(columns={'Date': 'Ex-Dividend Date'}, inplace=True)
                return df
            return pd.DataFrame()

        elif data_type == 'splits':
            # Stock splits
            splits = stock.splits
            if splits is not None and len(splits) > 0:
                df = splits.to_frame('Split Ratio')
                df.reset_index(inplace=True)
                df.rename(columns={'Date': 'Split Date'}, inplace=True)
                return df
            return pd.DataFrame()

        elif data_type == 'actions':
            # Combined dividends and splits
            actions = stock.actions
            if actions is not None and not actions.empty:
                df = actions.copy()
                df.reset_index(inplace=True)
                return df
            return pd.DataFrame()

        elif data_type == 'quarterly_cashflow':
            # Quarterly cashflow statement
            qcf = stock.quarterly_cashflow
            if qcf is not None and not qcf.empty:
                # Transpose for better readability
                df = qcf.T
                df.reset_index(inplace=True)
                df.rename(columns={'index': 'Date'}, inplace=True)
                return df
            return pd.DataFrame()

        elif data_type == 'cashflow':
            # Annual cashflow statement
            cf = stock.cashflow
            if cf is not None and not cf.empty:
                df = cf.T
                df.reset_index(inplace=True)
                df.rename(columns={'index': 'Date'}, inplace=True)
                return df
            return pd.DataFrame()

        elif data_type == 'quarterly_income':
            # Quarterly income statement
            qis = stock.quarterly_income_stmt
            if qis is not None and not qis.empty:
                df = qis.T
                df.reset_index(inplace=True)
                df.rename(columns={'index': 'Date'}, inplace=True)
                return df
            return pd.DataFrame()

        elif data_type == 'income_stmt':
            # Annual income statement
            inc = stock.income_stmt
            if inc is not None and not inc.empty:
                df = inc.T
                df.reset_index(inplace=True)
                df.rename(columns={'index': 'Date'}, inplace=True)
                return df
            return pd.DataFrame()

        elif data_type == 'quarterly_balance':
            # Quarterly balance sheet
            qbs = stock.quarterly_balance_sheet
            if qbs is not None and not qbs.empty:
                df = qbs.T
                df.reset_index(inplace=True)
                df.rename(columns={'index': 'Date'}, inplace=True)
                return df
            return pd.DataFrame()

        elif data_type == 'balance_sheet':
            # Annual balance sheet
            bs = stock.balance_sheet
            if bs is not None and not bs.empty:
                df = bs.T
                df.reset_index(inplace=True)
                df.rename(columns={'index': 'Date'}, inplace=True)
                return df
            return pd.DataFrame()

        elif data_type == 'info':
            # Info dictionary as DataFrame
            info = stock.info
            if info:
                # Convert dict to DataFrame with keys and values
                df = pd.DataFrame(list(info.items()), columns=['Field', 'Value'])
                return df
            return pd.DataFrame()

        else:
            return pd.DataFrame()

    except Exception as e:
        print(f"Error fetching {data_type} for {ticker}: {str(e)[:100]}")
        return pd.DataFrame()


def format_dataframe_for_display(df: pd.DataFrame, data_type: str) -> pd.DataFrame:
    """
    Format DataFrame for better display in Streamlit.

    Args:
        df: Raw DataFrame
        data_type: Type of data (for context-specific formatting)

    Returns:
        Formatted DataFrame
    """
    if df.empty:
        return df

    # Make a copy to avoid modifying original
    df_display = df.copy()

    # Format date columns
    date_columns = [col for col in df_display.columns if 'date' in col.lower() or col == 'Date']
    for col in date_columns:
        if df_display[col].dtype == 'datetime64[ns]' or pd.api.types.is_datetime64_any_dtype(df_display[col]):
            df_display[col] = df_display[col].dt.strftime('%Y-%m-%d')

    # Format large numbers (billions, millions)
    if data_type in ['quarterly_cashflow', 'cashflow', 'quarterly_income', 'income_stmt',
                     'quarterly_balance', 'balance_sheet']:
        # Financial statements - format large numbers
        for col in df_display.columns:
            if col != 'Date' and pd.api.types.is_numeric_dtype(df_display[col]):
                # Convert to millions for readability
                df_display[col] = df_display[col].apply(
                    lambda x: f"{x/1e6:.2f}M" if pd.notna(x) and abs(x) >= 1e6 else
                             (f"{x/1e3:.2f}K" if pd.notna(x) and abs(x) >= 1e3 else
                              (f"{x:.2f}" if pd.notna(x) else ""))
                )

    return df_display


def get_data_summary(df: pd.DataFrame, data_type: str) -> dict:
    """
    Generate summary statistics for a dataset.

    Args:
        df: DataFrame to summarize
        data_type: Type of data

    Returns:
        Dictionary with summary info
    """
    if df.empty:
        return {'records': 0}

    summary = {
        'records': len(df),
        'columns': len(df.columns)
    }

    # Add date range if applicable
    date_columns = [col for col in df.columns if 'date' in col.lower() or col == 'Date']
    if date_columns:
        date_col = date_columns[0]
        if pd.api.types.is_datetime64_any_dtype(df[date_col]):
            summary['date_range'] = f"{df[date_col].min().strftime('%Y-%m-%d')} to {df[date_col].max().strftime('%Y-%m-%d')}"
        elif df[date_col].dtype == object:
            # Already string format
            try:
                summary['date_range'] = f"{df[date_col].min()} to {df[date_col].max()}"
            except:
                pass

    return summary
