#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Refresh dividend history and shareholder remuneration data.

This script fetches dividend history, classifies dividends, and extracts
share actions from Yahoo Finance for all public European banks.

Usage:
    python scripts/refresh_dividends.py                 # Refresh all banks
    python scripts/refresh_dividends.py --ticker BBVA.MC # Single bank
    python scripts/refresh_dividends.py --dry-run        # Preview without saving
    python scripts/refresh_dividends.py --verbose        # Detailed output
"""

import argparse
import sqlite3
import time
import sys
import io
from pathlib import Path
from datetime import datetime

# Set UTF-8 output encoding for Windows console
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from market_bench.config import DB_NAME
from market_bench.dividends import (
    fetch_dividend_history,
    extract_buybacks,
    get_share_actions,
    upsert_dividend_history,
    upsert_share_actions
)


def refresh_dividend_data(lei: str, ticker: str, name: str, dry_run: bool = False, verbose: bool = False):
    """
    Fetch and store dividend history and share actions for a single bank.

    Args:
        lei: Legal Entity Identifier
        ticker: Yahoo Finance ticker symbol
        name: Bank name (for display)
        dry_run: If True, preview without saving to database
        verbose: If True, show detailed output
    """
    print(f"  Fetching dividend data...")

    # ========================================================================
    # 1. Fetch dividend history
    # ========================================================================

    divs = fetch_dividend_history(ticker, lei)

    if divs is None or divs.empty:
        print(f"    No dividend data available")
    else:
        print(f"    Found {len(divs)} dividend records")

        if verbose:
            print(f"    Date range: {divs['ex_date'].min()} to {divs['ex_date'].max()}")
            print(f"    Currency: {divs['currency'].iloc[0]}")
            print(f"    Total amount (EUR): {divs['amount_eur'].sum():.2f}")

        # Classify dividends
        # Default all dividends to 'regular' as special heuristic is removed
        divs['dividend_type'] = 'regular'

        if dry_run:
            print(f"    [DRY RUN] Would insert {len(divs)} dividend records")
        else:
            upsert_dividend_history(lei, divs)
            print(f"    [OK] Inserted/updated {len(divs)} dividend records")

    # ========================================================================
    # 2. Extract share actions (splits)
    # ========================================================================

    actions = get_share_actions(ticker, lei)

    if actions is None or actions.empty:
        if verbose:
            print(f"    No share actions (splits) found")
    else:
        print(f"    Found {len(actions)} share action(s)")

        if verbose:
            for _, row in actions.iterrows():
                print(f"      {row['action_date']}: {row['action_type']} ({row['ratio']}:1)")

        if dry_run:
            print(f"    [DRY RUN] Would insert {len(actions)} share action records")
        else:
            upsert_share_actions(lei, actions)
            print(f"    [OK] Inserted/updated {len(actions)} share action records")


def refresh_all_banks(dry_run: bool = False, verbose: bool = False):
    """
    Refresh dividend data for all public banks.

    Args:
        dry_run: If True, preview without saving to database
        verbose: If True, show detailed output
    """
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT lei, ticker, name
        FROM institutions
        WHERE ticker IS NOT NULL
        ORDER BY name
    """)
    banks = cursor.fetchall()
    conn.close()

    total = len(banks)
    print(f"Refreshing dividend data for {total} banks")
    print(f"Dry run: {dry_run}")
    print(f"Verbose: {verbose}\n")

    start_time = datetime.now()
    success = 0
    failed = 0
    skipped = 0

    for i, (lei, ticker, name) in enumerate(banks, 1):
        pct = (i / total) * 100
        print(f"[{i}/{total}] ({pct:.1f}%) {name} ({ticker})")

        try:
            refresh_dividend_data(lei, ticker, name, dry_run, verbose)
            success += 1

            # Rate limiting - be nice to yfinance API
            if i < total:  # Don't sleep after last bank
                time.sleep(0.5)

        except Exception as e:
            print(f"    [ERROR] {str(e)[:100]}")
            failed += 1
            continue

    duration = (datetime.now() - start_time).total_seconds()

    print("\n" + "="*70)
    print("Refresh complete!")
    print("="*70)
    print(f"Duration: {duration:.1f}s")
    print(f"Success: {success}")
    print(f"Failed: {failed}")
    print(f"Skipped: {skipped}")
    print("="*70)


def refresh_single_bank(ticker: str, dry_run: bool = False, verbose: bool = False):
    """
    Refresh dividend data for a single bank by ticker.

    Args:
        ticker: Yahoo Finance ticker symbol
        dry_run: If True, preview without saving to database
        verbose: If True, show detailed output
    """
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT lei, ticker, name
        FROM institutions
        WHERE ticker = ?
    """, (ticker,))

    bank = cursor.fetchone()
    conn.close()

    if not bank:
        print(f"Ticker {ticker} not found in database")
        return

    lei, ticker, name = bank
    print(f"Refreshing {name} ({ticker})\n")

    try:
        refresh_dividend_data(lei, ticker, name, dry_run, verbose)
        print("\n[OK] Refresh complete")
    except Exception as e:
        print(f"\n[ERROR] {str(e)}")


def main():
    parser = argparse.ArgumentParser(
        description='Refresh dividend and shareholder remuneration data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/refresh_dividends.py                    # Refresh all banks
  python scripts/refresh_dividends.py --dry-run          # Preview changes
  python scripts/refresh_dividends.py --ticker BBVA.MC   # Single bank
  python scripts/refresh_dividends.py --verbose          # Detailed output
        """
    )

    parser.add_argument(
        '--ticker',
        help='Refresh specific ticker only'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Preview changes without saving to database'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Show detailed output'
    )

    args = parser.parse_args()

    if args.ticker:
        refresh_single_bank(args.ticker, args.dry_run, args.verbose)
    else:
        refresh_all_banks(args.dry_run, args.verbose)


if __name__ == '__main__':
    main()
