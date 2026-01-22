# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **European Bank Market Benchmarking** application built with Streamlit that provides comparative analysis of publicly-traded European banks. The application fetches live market data from Yahoo Finance, calculates intrinsic valuations, and displays comprehensive benchmarking visualizations comparing banks across regions, sizes, and financial metrics.

## Architecture

### Core Components

1. **Streamlit App** (`src/market_bench/app.py`)
   - Main UI with two tabs: Benchmarking and Ranking
   - Benchmarking tab: Compares a selected "base bank" against structured peer groups
   - Ranking tab: Displays rankings across all EU banks by selected metric
   - Uses extensive caching via `@st.cache_data` for performance

2. **Data Layer** (`src/market_bench/data.py`)
   - `fetch_yahoo_data()`: Fetches and normalizes market data from Yahoo Finance
   - Currency normalization to EUR using live FX rates
   - Intrinsic value calculation using Residual Income Model: `IV = BV * (ROE - g) / (COE - g)`
   - Structured benchmark data ordering: Base Bank → Domestic → Regional → Core EU → CEE → EU Average
   - Special handling for GBp (pence) tickers with 100x price divisor

3. **Visualization** (`src/market_bench/plotting.py`)
   - `plot_benchmark_bar()`: Standard bar charts with color-coded peers and averages
   - `plot_market_history()`: Time series charts for 5-year historical trends
   - Color scheme: Base bank (blue), peers (grey), averages (green variants)

4. **Configuration** (`src/market_bench/config.py`)
   - `DB_NAME`: SQLite database path at `data/market_data.db`
   - `CHART_COLORS`: Color palette for visualizations
   - `DISPLAY_SETTINGS`: Chart dimensions and formatting preferences

### Database Schema

**SQLite database** at `data/market_data.db` with four tables:

- `institutions`: Bank metadata (LEI, name, ticker, country, region, size_category)
- `market_data`: Current snapshot of all market metrics including intrinsic_value and upside
- `market_history`: 5-year daily historical data (close prices, dividends, yields)
- `market_financial_years`: Annual financial year summaries (FY-level payouts, yields, earnings)

### Peer Group Logic

The benchmarking tab structures peers in a specific spotlight order:
1. **Base Bank**: The selected institution
2. **Domestic Peers**: Same country (+ Bank of Cyprus for Greek banks)
3. **Domestic Average**: Mean of domestic group including base bank
4. **Regional Peers**: Same region, excluding domestics and small banks
5. **Regional Average**: Mean of regional large/medium banks
6. **Core Europe Peers**: Western/Northern Europe large/medium banks
7. **Core Europe Average**
8. **CEE Peers**: All Central & Eastern European banks (excluding duplicates)
9. **CEE Average**: Mean of all CEE banks
10. **EU Wide Average**: All banks in dataset

### Valuation Model

**Intrinsic Value** uses a Residual Income approach:
- **ROE** = EPS / Book Value
- **COE** = Risk-Free Rate + Beta × Equity Risk Premium
- **Assumptions**: RF = 3.5%, ERP = 5.5%, g = 2.0%
- **Formula**: `IV = BV × (ROE - g) / (COE - g)`
- **Upside** = (IV - Current Price) / Current Price

### Special Data Handling

- **Currency Conversion**: All metrics normalized to EUR via `get_fx_rate()`
- **GBp Tickers**: Price divided by 100, but book value/EPS in GBP (no adjustment)
- **Market Cap Validation**: Uses calculated cap if YF data deviates >20%
- **Buyback Yield**: Calculated from TTM cashflow, tries quarterly then annual data
- **Manual Overrides**: Hardcoded fixes for specific banks (e.g., NLB shares/book value)

## Common Commands

### Database Initialization

```bash
python scripts/init_db.py
```
Initializes `data/market_data.db` by copying institution data from a sibling `eba-benchmarking` repository. Creates tables and indices. **Note**: This script expects `../eba-benchmarking/data/eba_data.db` to exist.

### Refresh Market Data

```bash
python scripts/refresh_data.py
```
Fetches current market data from Yahoo Finance for all public banks and updates the `market_data` table. This includes:
- Live prices, market caps, ratios (P/E, P/B)
- Historical returns (1Y, 3Y, 5Y)
- Dividend and buyback yields
- Intrinsic value calculations

### Run the Application

```bash
streamlit run src/market_bench/app.py
```
Launches the Streamlit web interface. Default base bank is "National Bank of Greece" if available.

### Debug Scripts

Multiple `debug_*.py` scripts exist in the root for troubleshooting specific data issues:
- `debug_bbva_deep.py`: Deep dive into BBVA cashflow and share count data
- `debug_currency.py`: Test currency conversion logic
- `debug_normalization.py`: Verify data normalization
- `debug_pl_and_intrinsic.py`: Validate P&L and intrinsic value calculations
- Other verification scripts: `verify_final.py`, `verify_iv.py`, `verify_normalization_final.py`

These are utility scripts for development/debugging and not part of the main application flow.

## Key Implementation Notes

### When Modifying Data Fetching

- Currency conversions must handle both **price** (with GBp divisor) and **metrics** (without divisor)
- Always validate market cap against calculated value (current_price × shares)
- Buyback yield requires parsing cashflow statements—look for "Repurchase Of Capital Stock"
- Use `try/except` liberally; Yahoo Finance data is often incomplete

### When Adding New Metrics

1. Add column to SQL schema (or modify `market_data` table)
2. Update `fetch_yahoo_data()` in `data.py` to calculate/fetch the metric
3. Add to `numeric_cols` list in `get_structured_benchmark_data()` for averaging
4. Update `fy_numeric_cols` if it's a financial year metric
5. Add visualization in `app.py` using `plot_benchmark_bar()`

### When Adding New Banks

Banks must be in the `institutions` table with:
- **LEI**: Legal Entity Identifier (primary key)
- **ticker**: Yahoo Finance ticker symbol (e.g., "BBVA.MC")
- **country**: ISO country code
- **region**: One of "Western Europe", "Northern Europe", "Southern Europe", "CEE"
- **size_category**: "Large (>500bn)", "Medium (50-500bn)", or "Small (<50bn)"

### Caching Behavior

The app uses `@st.cache_data(ttl=3600)` extensively. Data is cached for 1 hour. Use the "Clear Cache" button in the sidebar to force refresh during development. When running outside Streamlit (e.g., scripts), the cache decorator is a no-op.

### Chart Ordering

The `get_structured_benchmark_data()` function returns a DataFrame pre-sorted in the "spotlight" peer group order. When plotting, **do NOT re-sort by value** if you want to preserve this grouping—use `sort_by_value=False` in `plot_benchmark_bar()`.

## Dependencies

Core Python packages (inferred from imports):
- `streamlit`: Web application framework
- `pandas`: Data manipulation
- `sqlite3`: Database (standard library)
- `yfinance`: Yahoo Finance API
- `plotly`: Interactive charting
- `numpy`: Numerical operations

No `requirements.txt` exists yet. Create one if needed with the above packages.
