# Shareholder Remuneration Enhancement Roadmap

## Executive Summary

This roadmap outlines enhancements to the European Bank Market Benchmarking application to provide comprehensive shareholder remuneration analysis. The goal is to understand each bank's capital return strategy through historical dividend/buyback data, payout ratios, and a new data exploration interface.

---

## Current State Assessment

### What's Already Tracked

| Data Point | Granularity | Source |
|------------|-------------|--------|
| Dividend Yield | TTM + Annual FY | yfinance `dividendYield` |
| Buyback Yield | TTM + Annual FY | Cashflow "Repurchase Of Capital Stock" |
| Total Payout Yield | TTM + Annual FY | Calculated (div + buyback) |
| Payout Ratio | TTM + Annual FY | (Div + Buyback) / Net Income |
| EPS, DPS | Trailing | yfinance `trailingEps`, `dividendRate` |
| Net Income | Annual FY | Income statement |
| 5-Year Price History | Daily | yfinance `history()` |

1. **Extraordinary/Special Dividends** - Handled as part of total dividends (no separate classification)
2. **Dividend vs Buyback Split Visualization** - Data exists but not prominently displayed
3. **Historical Trailing Yields** - Only current TTM stored; no time series of trailing yields
4. **Per-Year Rankings** - Rankings tab shows only current snapshot
5. **5-Year Cumulative Returns** - Not calculated or displayed
6. **Raw Data Explorer** - No way to inspect underlying yfinance data

---

## Proposed Enhancements

### Phase 1: Data Layer Enhancements

#### 1.1 Expand Yahoo Finance Data Extraction

**New data to fetch in `fetch_yahoo_data()`:**

```python
# From yfinance Ticker.info
- 'exDividendDate'           # Track ex-dividend timing
- 'lastDividendValue'        # Most recent dividend amount
- 'lastDividendDate'         # When last dividend paid
- 'fiveYearAvgDividendYield' # Yahoo's 5Y average

- Full dividend history       # Every dividend payment with date & amount

# From yfinance Ticker.actions
- Stock splits               # Adjust historical per-share metrics
- Dividend actions           # Complete dividend timeline

# From yfinance Ticker.quarterly_cashflow / Ticker.cashflow
- Already fetching buybacks
- Add: 'Cash Dividends Paid' # Cross-validate dividend amounts
- Add: 'Issuance Of Stock'   # Net share issuance (dilution tracking)

# From yfinance Ticker.quarterly_income_stmt / Ticker.income_stmt
- 'Net Income'               # Already have annual
- Quarterly net income       # For quarterly payout ratio calculations

# From yfinance Ticker.balance_sheet
- 'Total Stockholders Equity' # Book value validation
- 'Retained Earnings'         # Accumulated vs distributed
```

#### 1.2 New Database Tables

**`dividend_history`** - Individual dividend payments
```sql
CREATE TABLE dividend_history (
    lei TEXT NOT NULL,
    ex_date DATE NOT NULL,
    payment_date DATE,
    amount_local REAL,      -- Original currency
    amount_eur REAL,        -- Normalized to EUR
    currency TEXT,
    dividend_type TEXT,     -- 'regular', 'special', 'interim'
    PRIMARY KEY (lei, ex_date)
);
CREATE INDEX idx_div_hist_lei ON dividend_history(lei);
CREATE INDEX idx_div_hist_date ON dividend_history(ex_date);
```

**`trailing_yields_history`** - Monthly snapshots of trailing yields
```sql
CREATE TABLE trailing_yields_history (
    lei TEXT NOT NULL,
    snapshot_date DATE NOT NULL,
    dividend_yield_ttm REAL,
    buyback_yield_ttm REAL,
    payout_yield_ttm REAL,
    market_cap REAL,
    current_price REAL,
    PRIMARY KEY (lei, snapshot_date)
);
```

**`share_actions`** - Corporate actions affecting share count
```sql
CREATE TABLE share_actions (
    lei TEXT NOT NULL,
    action_date DATE NOT NULL,
    action_type TEXT,       -- 'split', 'reverse_split', 'issuance', 'buyback'
    ratio REAL,             -- For splits: 2.0 = 2:1 split
    shares_affected REAL,
    PRIMARY KEY (lei, action_date, action_type)
);
```

#### 1.3 Enhanced Financial Year Data

Modify `market_financial_years` to add:
```sql
ALTER TABLE market_financial_years ADD COLUMN regular_dividend_amt REAL;
ALTER TABLE market_financial_years ADD COLUMN special_dividend_amt REAL;
ALTER TABLE market_financial_years ADD COLUMN dividend_share_pct REAL;  -- div / (div + buyback)
ALTER TABLE market_financial_years ADD COLUMN buyback_share_pct REAL;   -- buyback / (div + buyback)
ALTER TABLE market_financial_years ADD COLUMN shares_outstanding REAL;
ALTER TABLE market_financial_years ADD COLUMN shares_repurchased REAL;
```

---

### Phase 2: Data Processing Logic

#### 2.2 Trailing Yield Time Series Builder

```python
def build_trailing_yield_history(lei: str, years: int = 5) -> pd.DataFrame:
    """
    Reconstruct historical trailing yields by:
    1. Getting all dividend payments
    2. For each month-end, sum prior 12 months of dividends
    3. Divide by market cap at that date
    Returns monthly time series of trailing yields
    """
```

#### 2.3 Payout Split Calculator

```python
def calculate_payout_split(lei: str, fy: int) -> dict:
    """
    Returns:
    - dividend_share: % of total payout via dividends
    - buyback_share: % of total payout via buybacks
    - regular_div_share: % via regular dividends
    - special_div_share: % via special dividends
    """
```

---

### Phase 3: New Data Explorer Tab

#### 3.1 Tab Structure

```
Tab 3: Data Explorer
├── Bank Selector (dropdown)
├── Data Category Selector
│   ├── Price & Volume History
│   ├── Dividend History (full list)
│   ├── Cashflow Statements
│   ├── Income Statements
│   ├── Balance Sheets
│   └── Raw yfinance Info Dictionary
├── Date Range Filter
├── Data Table (sortable, searchable)
└── Export Button (CSV/Excel)
```

#### 3.2 Implementation Approach

```python
# New function in data.py
def get_raw_yfinance_data(ticker: str, data_type: str) -> pd.DataFrame:
    """
    Fetches and returns raw yfinance data without processing.
    data_type: 'info', 'history', 'dividends', 'actions',
               'quarterly_financials', 'quarterly_cashflow',
               'quarterly_balance_sheet', 'income_stmt',
               'cashflow', 'balance_sheet'
    """
    stock = yf.Ticker(ticker)

    if data_type == 'info':
        return pd.DataFrame([stock.info])
    elif data_type == 'history':
        return stock.history(period="max")
    elif data_type == 'dividends':
        return stock.dividends.to_frame()
    # ... etc
```

#### 3.3 UI Components

- **Collapsible JSON viewer** for `info` dictionary
- **Interactive DataFrames** with st.dataframe() for tabular data
- **Date filtering** with st.date_input() range selector
- **Column selector** for large tables
- **Quick stats** summary at top (row count, date range, etc.)

---

### Phase 4: Enhanced Rankings Tab

#### 4.1 Year Selector Feature

```
Rankings Tab (Enhanced)
├── Metric Selector (existing)
├── Time Period Selector (NEW)
│   ├── Current (TTM)
│   ├── FY 2024
│   ├── FY 2023
│   ├── FY 2022
│   ├── FY 2021
│   ├── FY 2020
│   └── 5-Year Cumulative
├── Ranking Table
├── Bar Chart (Top 20)
└── Trend Sparklines (NEW - mini charts per bank)
```

#### 4.2 New Ranking Metrics

Add to ranking dropdown:
- Dividend Yield (by FY)
- Buyback Yield (by FY)
- Total Payout Yield (by FY)
- Payout Ratio (by FY)
- Dividend Growth Rate (5Y CAGR)
- Buyback Consistency Score (years with buybacks / 5)
- Total Shareholder Return (price + dividends, 1Y/3Y/5Y)
- Cumulative Payout (5Y sum in EUR)

#### 4.3 5-Year Cumulative Calculations

```python
def calculate_cumulative_metrics(lei: str) -> dict:
    """
    Returns 5-year cumulative/average metrics:
    - total_dividends_5y: Sum of all dividends (EUR)
    - total_buybacks_5y: Sum of all buybacks (EUR)
    - total_payout_5y: Combined total (EUR)
    - avg_dividend_yield_5y: Average annual yield
    - avg_payout_ratio_5y: Average annual payout ratio
    - dividend_cagr_5y: Compound annual growth rate
    - tsr_5y: Total shareholder return (price appreciation + dividends)
    """
```

---

### Phase 5: Visualization Enhancements

#### 5.1 New Charts for Benchmarking Tab

**Payout Composition Stacked Bar:**
```python
def plot_payout_composition(df: pd.DataFrame) -> go.Figure:
    """
    Stacked bar showing:
    - Regular Dividends (blue)
    - Special Dividends (gold)
    - Buybacks (grey)
    Per bank, for selected FY or TTM
    """
```

**Historical Trailing Yield Line Chart:**
```python
def plot_trailing_yield_history(base_lei: str, peer_leis: list) -> go.Figure:
    """
    Multi-line chart showing 5 years of monthly trailing yields
    Base bank highlighted, peers in background
    """
```

**Dividend Policy Heatmap:**
```python
def plot_dividend_heatmap(df: pd.DataFrame) -> go.Figure:
    """
    Banks on Y-axis, Years on X-axis
    Color intensity = dividend yield or payout ratio
    Shows consistency and changes in policy
    """
```

#### 5.2 Rankings Tab Visualizations

**Trend Sparklines:**
- Mini line charts (50x20px) showing 5-year trend per metric
- Embedded in ranking table as rightmost column

**Year-over-Year Comparison:**
```python
def plot_yoy_comparison(metric: str, year1: int, year2: int) -> go.Figure:
    """
    Scatter plot: X = Year1 value, Y = Year2 value
    Diagonal line = no change
    Above line = improved, below = declined
    """
```

---

### Phase 6: Data Refresh Enhancements

#### 6.1 New Refresh Script Capabilities

Modify `scripts/refresh_data.py` to:

1. **Populate dividend_history table:**
```python
def refresh_dividend_history(lei: str, ticker: str):
    stock = yf.Ticker(ticker)
    dividends = stock.dividends
    # Convert to EUR, classify as regular/special
    # Upsert into dividend_history table
```

2. **Build trailing yields history:**
```python
def refresh_trailing_yields(lei: str):
    # Calculate monthly trailing yields for past 5 years
    # Insert/update trailing_yields_history table
```

3. **Update financial year splits:**
```python
def refresh_fy_payout_splits(lei: str):
    # Calculate dividend vs buyback percentages
    # Update market_financial_years with split columns
```

#### 6.2 Incremental Refresh Option

```bash
# Full refresh (existing behavior)
python scripts/refresh_data.py

# Incremental - only update changed data
python scripts/refresh_data.py --incremental

# Specific bank refresh
python scripts/refresh_data.py --ticker BBVA.MC

# Refresh only new tables
python scripts/refresh_data.py --dividends-only
```

---

## Implementation Priority & Sequence

### Sprint 1: Database & Data Layer (Week 1-2)
1. Create new database tables (dividend_history, trailing_yields_history)
2. Add columns to market_financial_years
3. Implement dividend classification algorithm
4. Extend fetch_yahoo_data() with new fields
5. Update refresh_data.py for new tables

### Sprint 2: Data Explorer Tab (Week 2-3)
1. Create basic tab structure in app.py
2. Implement get_raw_yfinance_data() function
3. Build UI components (selectors, tables, export)
4. Add caching for raw data views
5. Test with all 40 banks

### Sprint 3: Rankings Tab Enhancement (Week 3-4)
1. Add year selector UI component
2. Implement per-year data retrieval functions
3. Calculate 5-year cumulative metrics
4. Add new ranking metrics to dropdown
5. Build sparkline visualizations

### Sprint 4: Visualizations & Polish (Week 4-5)
1. Create payout composition charts
2. Implement trailing yield history charts
3. Build dividend policy heatmap
4. Add YoY comparison scatter plots
5. UI polish and responsiveness

### Sprint 5: Testing & Documentation (Week 5-6)
1. Comprehensive data validation
2. Edge case handling (missing data, currency issues)
3. Performance optimization (caching, indices)
4. Update CLAUDE.md with new architecture
5. User documentation / help tooltips

---

## Technical Considerations

### Performance
- Add database indices for new tables on (lei, date) columns
- Use `@st.cache_data` with appropriate TTL for new functions
- Consider lazy loading for Data Explorer (paginate large datasets)
- Pre-calculate 5-year cumulative metrics during refresh (not on-demand)

### Data Quality
- Handle missing dividend data gracefully (many smaller banks have gaps)
- Validate special dividend classification manually for top 10 banks
- Cross-reference buyback amounts with share count changes
- Document data limitations prominently in UI

### Currency Handling
- All historical dividends must be converted to EUR at historical FX rates
- Store both local and EUR amounts in dividend_history
- Use period-end FX rates for FY calculations

### API Rate Limits
- yfinance has unofficial rate limits (~2000 requests/hour)
- Add delays between bank fetches in refresh script
- Cache raw data locally to minimize re-fetching

---

## Files to Create/Modify

### New Files
- `src/market_bench/explorer.py` - Data explorer tab logic
- `src/market_bench/dividends.py` - Dividend analysis functions
- `scripts/refresh_dividends.py` - Dividend history refresh
- `scripts/migrate_schema.py` - Database schema migration

### Modified Files
- `src/market_bench/app.py` - Add Data Explorer tab, enhance Rankings
- `src/market_bench/data.py` - New fetch functions, calculations
- `src/market_bench/plotting.py` - New chart functions
- `src/market_bench/config.py` - New color schemes, settings
- `scripts/refresh_data.py` - Extended refresh logic
- `CLAUDE.md` - Updated architecture documentation

---

## Success Metrics

1. **Data Completeness**: 90%+ of banks have 5-year dividend history populated
2. **Classification Accuracy**: Special dividends correctly identified for major banks
3. **Performance**: Rankings tab loads in <3 seconds with year selector
4. **Usability**: Data Explorer provides access to all yfinance data types
5. **Insight Quality**: Users can clearly see dividend vs buyback strategy shifts

---

## Open Questions for Discussion

1. **Special Dividend Threshold**: What % above trailing average qualifies as "special"? (Proposed: 50%)

2. **Trailing Yield Calculation**: Use month-end market cap or average? (Proposed: month-end)

3. **Missing Data Display**: Show as blank, zero, or "N/A"? (Proposed: "N/A" with tooltip)

4. **5-Year Cumulative Base**: Calendar years or fiscal years? (Proposed: fiscal years)

5. **Data Explorer Access**: All banks or just public tickers? (Proposed: public only)

---

## Next Steps

1. Review this roadmap and provide feedback
2. Prioritize which phases are most valuable
3. Clarify any open questions
4. Begin Sprint 1 implementation

---

*Document created: January 2026*
*Last updated: January 2026*
