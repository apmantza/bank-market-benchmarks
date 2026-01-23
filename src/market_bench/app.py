import streamlit as st
import pandas as pd
import numpy as np
import sys
import os

# Add src to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from market_bench.config import DISPLAY_SETTINGS
from market_bench.data import (
    get_all_banks, get_market_data, get_market_history,
    get_structured_benchmark_data, get_market_financial_years, get_market_fy_averages,
    get_ranking_data, get_ranking_data_by_year, calculate_cumulative_metrics,
    calculate_total_shareholder_return
)
from market_bench.plotting import plot_benchmark_bar, plot_market_history
from market_bench.dividends import get_dividend_history, get_share_actions_history
from market_bench.explorer import get_raw_yfinance_data, format_dataframe_for_display, get_data_summary

# Page Config
st.set_page_config(
    page_title="European Bank Benchmarking",
    page_icon="📈",
    layout="wide"
)

def main():
    st.title("📈 European Bank Market Benchmarking")
    
    # --- Sidebar ---
    st.sidebar.header("Configuration")
    
    df_banks = get_all_banks()
    if df_banks.empty:
        st.error("No bank data found. Please run initialization scripts.")
        return

    # Base Bank Selection
    # Default to National Bank of Greece if available
    nbg_idx = 0
    try:
        nbg_list = df_banks[df_banks['name'].str.contains("National Bank of Greece", case=False)]
        if not nbg_list.empty:
            nbg_name = nbg_list.iloc[0]['name']
            nbg_idx = df_banks['name'].tolist().index(nbg_name)
    except:
        pass

    base_bank_name = st.sidebar.selectbox(
        "Select Base Bank",
        df_banks['name'].unique(),
        index=nbg_idx
    )
    
    # Get Base Bank Metadata
    base_row = df_banks[df_banks['name'] == base_bank_name].iloc[0]
    base_lei = base_row['lei']
    base_country = base_row['country']
    base_region = base_row['region']
    base_size = base_row['size_category']
    
    st.sidebar.markdown(f"""
    **Base Bank Details**
    - 🌍 **Region**: {base_region}
    - 🏛️ **Country**: {base_country}
    - 📏 **Size**: {base_size}
    """)
    
    if st.sidebar.button("Clear Cache"):
        st.cache_data.clear()
        st.rerun()

    # --- TABS ---
    tab_bench, tab_rank, tab_remun, tab_explorer = st.tabs(["📊 Benchmarking", "🏆 Ranking", "💰 Remuneration", "🔍 Data Explorer"])
    
    # =========================================================================
    # TAB 1: BENCHMARKING
    # =========================================================================
    with tab_bench:
        # Fetch Structured Data (The "Spotlight" Order)
        df_structured = get_structured_benchmark_data(base_lei)
        
        if df_structured.empty:
            st.warning("No benchmarking data available.")
        else:
            # 2. Key Metrics Snapshot (Base Bank)
            base_mkt = df_structured[df_structured['lei'] == base_lei]
            if not base_mkt.empty:
                m = base_mkt.iloc[0]
                st.subheader(f"Snapshot: {base_bank_name}")
                
                c1, c2, c3, c4, c5 = st.columns(5)
                c1.metric("Stock Price", f"€{m.get('current_price', 0):.2f}")
                c2.metric("Intrinsic Value", f"€{m.get('intrinsic_value', 0):.2f}" if pd.notna(m.get('intrinsic_value')) else "N/A")
                c3.metric("Upside", f"{m.get('upside', 0)*100:.1f}%" if pd.notna(m.get('upside')) else "N/A")
                c4.metric("P/B Ratio", f"{m.get('price_to_book', 0):.2f}" if m.get('price_to_book') else "N/A")
                c5.metric("P/E Ratio", f"{m.get('pe_trailing', 0):.1f}" if m.get('pe_trailing') else "N/A")

                c6, c7, c8, c9, c10 = st.columns(5)
                c6.metric("Div Yield", f"{m.get('dividend_yield', 0)*100:.2f}%" if m.get('dividend_yield') is not None else "N/A")
                c7.metric("Buyback Yield", f"{m.get('buyback_yield', 0)*100:.2f}%" if pd.notna(m.get('buyback_yield')) else "N/A")
                c8.metric("Payout Yield", f"{m.get('payout_yield', 0)*100:.2f}%" if pd.notna(m.get('payout_yield')) else "N/A")
                c9.metric("Beta", f"{m.get('beta', 0):.2f}" if pd.notna(m.get('beta')) else "N/A")
                c10.metric("YTD Return", f"{m.get('ytd_return', 0)*100:.1f}%" if pd.notna(m.get('ytd_return')) else "N/A")
                st.divider()

            # 3. Valuation & Return Charts
            st.subheader("📊 Valuation & Returns Benchmarking")
            st.caption("Peer Group Order: Domestic → Regional (>Small) → Core Europe (>Small) → CEE Peers → EU Average")
            
            c1, c2 = st.columns(2)
            with c1:
                st.plotly_chart(plot_benchmark_bar(df_structured, 'intrinsic_value', "Intrinsic Value (€)", base_bank_name, format_pct=False, sort_by_value=False), use_container_width=True)
            with c2:
                st.plotly_chart(plot_benchmark_bar(df_structured, 'upside', "Implied Upside (%)", base_bank_name, format_pct=True, sort_by_value=False), use_container_width=True)

            c3, c4 = st.columns(2)
            with c3:
                st.plotly_chart(plot_benchmark_bar(df_structured, 'price_to_book', "Price-to-Book Ratio", base_bank_name, format_pct=False, scale_amounts=False, sort_by_value=False), use_container_width=True)
            with c4:
                st.plotly_chart(plot_benchmark_bar(df_structured, 'pe_trailing', "P/E Ratio (Trailing)", base_bank_name, format_pct=False, scale_amounts=False, sort_by_value=False), use_container_width=True)

            # Market Cap
            c_mc1, c_mc2 = st.columns(2)
            with c_mc1:
                df_mc = df_structured.copy()
                df_mc['market_cap_B'] = df_mc['market_cap'] / 1e9
                st.plotly_chart(plot_benchmark_bar(df_mc, 'market_cap_B', "Market Cap (€B)", base_bank_name, format_pct=False, scale_amounts=False, sort_by_value=False), use_container_width=True)
            with c_mc2:
                st.empty()

            # Shareholder Returns
            st.subheader("💰 Shareholder Returns (Trailing 12m)")
            c5, c6 = st.columns(2)
            with c5:
                st.plotly_chart(plot_benchmark_bar(df_structured, 'dividend_yield', "Dividend Yield", base_bank_name, format_pct=True, sort_by_value=False), use_container_width=True)
            with c6:
                st.plotly_chart(plot_benchmark_bar(df_structured, 'buyback_yield', "Buyback Yield", base_bank_name, format_pct=True, sort_by_value=False), use_container_width=True)

            c7, c8 = st.columns(2)
            with c7:
                st.plotly_chart(plot_benchmark_bar(df_structured, 'payout_yield', "Total Payout Yield", base_bank_name, format_pct=True, sort_by_value=False), use_container_width=True)
            with c8:
                st.empty()

            # Per Share Metrics
            st.subheader("🪙 Per Share Metrics (Trailing 12m)")
            c9, c10 = st.columns(2)
            with c9:
                st.plotly_chart(plot_benchmark_bar(df_structured, 'eps_trailing', "EPS (Trailing)", base_bank_name, format_pct=False, sort_by_value=False), use_container_width=True)
            with c10:
                st.plotly_chart(plot_benchmark_bar(df_structured, 'dps_trailing', "DPS (Trailing)", base_bank_name, format_pct=False, sort_by_value=False), use_container_width=True)

            # Risk & Performance
            st.subheader("⚠️ Risk & Performance")
            c11, c12 = st.columns(2)
            with c11:
                st.plotly_chart(plot_benchmark_bar(df_structured, 'beta', "Beta (Systematic Risk)", base_bank_name, format_pct=False, sort_by_value=False), use_container_width=True)
            with c12:
                st.plotly_chart(plot_benchmark_bar(df_structured, 'return_1y', "1-Year Return", base_bank_name, format_pct=True, sort_by_value=False), use_container_width=True)

            c13, c14 = st.columns(2)
            with c13:
                st.plotly_chart(plot_benchmark_bar(df_structured, 'return_3y', "3-Year Return", base_bank_name, format_pct=True, sort_by_value=False), use_container_width=True)
            with c14:
                st.plotly_chart(plot_benchmark_bar(df_structured, 'return_5y', "5-Year Return", base_bank_name, format_pct=True, sort_by_value=False), use_container_width=True)

            # 4. Strategic Analysis (FY)
            # Todo: Order the FY pivot table columns to match the structured order?
            # It's a bit harder since columns are dynamic names. But we can display the table as is.
            st.divider()
            st.subheader("🏛️ Strategic Financial Year Analysis")
            
            df_fy = get_market_financial_years([base_lei]) 
            df_fy_bench = get_market_fy_averages(base_country, base_region, base_size)
            
            if not df_fy.empty:
                # Pivot logic
                def get_fy_pivot(df, val_col, bench_df=None):
                    if bench_df is not None and not bench_df.empty:
                        df_combined = pd.concat([df[['fy', 'name', val_col]], bench_df[['fy', 'name', val_col]]], ignore_index=True)
                    else:
                        df_combined = df
                    pivot = df_combined.pivot(index='fy', columns='name', values=val_col)
                    pivot = pivot.sort_index(ascending=False)
                    return pivot
                    
                t1, t2, t3 = st.tabs(["Yields", "Payouts", "Absolutes"])
                
                with t1:
                    st.write("**Total Strategic Yield (%)**")
                    st.dataframe(get_fy_pivot(df_fy, 'total_yield_fy', df_fy_bench).style.format("{:.2%}"), use_container_width=True)
                with t2:
                    st.write("**Total Payout Ratio (%)**")
                    st.dataframe(get_fy_pivot(df_fy, 'payout_ratio_fy', df_fy_bench).style.format("{:.1%}"), use_container_width=True)
                with t3:
                    st.write("**Net Income & Distributions (€M)**")
                    df_disp = df_fy.copy()
                    df_disp['Net Income'] = df_disp['net_income'] / 1e6
                    df_disp['Dividends'] = df_disp['dividend_amt'] / 1e6
                    df_disp['Buybacks'] = df_disp['buyback_amt'] / 1e6
                    st.dataframe(df_disp[['fy', 'Net Income', 'Dividends', 'Buybacks']].set_index('fy').style.format("{:,.1f}"), use_container_width=True)

            # 5. History
            st.divider()
            st.subheader("📈 Historical Trends")
            df_hist = get_market_history([base_lei])
            if not df_hist.empty:
                st.plotly_chart(plot_market_history(df_hist, 'close', "Stock Price History (5Y)", base_bank_name, format_pct=False, show_legend=False), use_container_width=True)
            
            # Download
            st.markdown("### 📥 Download Market Data")
            csv = df_structured.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download Benchmarking Data (CSV)", data=csv, file_name='benchmarking_data.csv', mime='text/csv')

    # =========================================================================
    # TAB 2: RANKING
    # =========================================================================
    with tab_rank:
        st.subheader("🏆 EU Bank Rankings")

        # Time Period Selector (NEW)
        col_period, col_metric = st.columns([1, 2])

        with col_period:
            time_periods = [
                "Current (TTM)",
                "FY 2024",
                "FY 2023",
                "FY 2022",
                "FY 2021",
                "FY 2020",
                "5-Year Cumulative"
            ]

            time_period = st.selectbox(
                "Time Period",
                time_periods,
                key="ranking_time_period"
            )

        # Fetch data based on time period selection
        df_rank = pd.DataFrame()

        if time_period == "Current (TTM)":
            df_rank = get_ranking_data()
            available_metrics = [
                'market_cap', 'intrinsic_value', 'upside', 'price_to_book',
                'dividend_yield', 'buyback_yield', 'payout_yield',
                'pe_trailing', 'return_1y', 'return_3y', 'return_5y',
                'eps_trailing', 'dps_trailing', 'payout_ratio'
            ]

        elif time_period.startswith("FY"):
            # Extract year from "FY 2024" format
            fy = int(time_period.split()[1])
            df_rank = get_ranking_data_by_year(fy)
            available_metrics = [
                'dividend_yield_fy', 'buyback_yield_fy', 'total_yield_fy',
                'payout_ratio_fy', 'dividend_payout_ratio_fy',
                'earnings_yield_fy', 'eps_fy', 'dps_fy',
                'dividend_amt', 'buyback_amt'
            ]

        elif time_period == "5-Year Cumulative":
            df_rank = calculate_cumulative_metrics()
            available_metrics = [
                'total_dividends_5y', 'total_buybacks_5y', 'total_payout_5y',
                'avg_dividend_yield_5y', 'avg_buyback_yield_5y', 'avg_payout_yield_5y',
                'avg_payout_ratio_5y', 'dividend_cagr_5y', 'consistency_score'
            ]

        if df_rank.empty:
            st.warning(f"No data available for {time_period}.")
        else:
            with col_metric:
                # Filter metrics to only those present in the dataframe
                valid_metrics = [m for m in available_metrics if m in df_rank.columns]

                rank_metric = st.selectbox(
                    "Rank By",
                    valid_metrics,
                    format_func=lambda x: x.replace('_', ' ').title(),
                    key="ranking_metric"
                )

            # Sort and rank
            df_rank_sorted = df_rank[['name', 'ticker', 'country', 'region', rank_metric]].dropna()
            df_rank_sorted = df_rank_sorted.sort_values(rank_metric, ascending=False).reset_index(drop=True)
            df_rank_sorted.index += 1  # 1-based ranking
            df_rank_sorted.insert(0, 'Rank', df_rank_sorted.index)

            # Display
            c_rank1, c_rank2 = st.columns([1, 2])

            with c_rank1:
                # Format based on metric type
                format_dict = {
                    'market_cap': '€{:,.0f}',
                    'intrinsic_value': '€{:.2f}',
                    'upside': '{:.1%}',
                    'price_to_book': '{:.2f}x',
                    'pe_trailing': '{:.1f}x',
                    'dividend_yield': '{:.2%}',
                    'buyback_yield': '{:.2%}',
                    'payout_yield': '{:.2%}',
                    'dividend_yield_fy': '{:.2%}',
                    'buyback_yield_fy': '{:.2%}',
                    'total_yield_fy': '{:.2%}',
                    'avg_dividend_yield_5y': '{:.2%}',
                    'avg_buyback_yield_5y': '{:.2%}',
                    'avg_payout_yield_5y': '{:.2%}',
                    'payout_ratio': '{:.1%}',
                    'payout_ratio_fy': '{:.1%}',
                    'dividend_payout_ratio_fy': '{:.1%}',
                    'avg_payout_ratio_5y': '{:.1%}',
                    'return_1y': '{:.1%}',
                    'return_3y': '{:.1%}',
                    'return_5y': '{:.1%}',
                    'eps_trailing': '€{:.2f}',
                    'dps_trailing': '€{:.2f}',
                    'eps_fy': '€{:.2f}',
                    'dps_fy': '€{:.2f}',
                    'earnings_yield_fy': '{:.2%}',
                    'dividend_amt': '€{:,.0f}',
                    'buyback_amt': '€{:,.0f}',
                    'total_dividends_5y': '€{:,.0f}',
                    'total_buybacks_5y': '€{:,.0f}',
                    'total_payout_5y': '€{:,.0f}',
                    'dividend_cagr_5y': '{:.1%}',
                    'consistency_score': '{:.0%}'
                }

                # Apply formatting only for columns present in the dataframe
                format_to_apply = {k: v for k, v in format_dict.items() if k in df_rank_sorted.columns}

                st.dataframe(
                    df_rank_sorted.style.format(format_to_apply),
                    use_container_width=True,
                    height=600
                )

            with c_rank2:
                # Top 20 Chart
                st.markdown(f"**Top 20 by {rank_metric.replace('_', ' ').title()}**")
                top_20 = df_rank_sorted.head(20)
                st.bar_chart(top_20.set_index('name')[rank_metric])

            # Download button
            st.divider()
            csv = df_rank_sorted.to_csv(index=False)
            file_name = f"eu_bank_rankings_{time_period.replace(' ', '_')}_{rank_metric}_{pd.Timestamp.now().strftime('%Y%m%d')}.csv"
            st.download_button(
                label="📥 Download Rankings CSV",
                data=csv,
                file_name=file_name,
                mime="text/csv",
                key="download_rankings"
            )

    # =========================================================================
    # TAB 3: REMUNERATION (NEW)
    # =========================================================================
    with tab_remun:
        st.subheader("💰 Shareholder Remuneration Detail")
        st.markdown("Deep dive into historical dividends, buybacks, and key per-share metrics for every bank.")

        # 1. Download ALL Data
        st.markdown("### 📥 Full Dataset")
        # Fetch all data for all banks
        all_banks_lei = df_banks['lei'].tolist()
        df_all_remun = get_market_financial_years(all_banks_lei)
        
        if not df_all_remun.empty:
            csv_all = df_all_remun.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download All Remuneration Data (CSV)",
                data=csv_all,
                file_name=f'shareholder_remuneration_full_{pd.Timestamp.now().strftime("%Y%m%d")}.csv',
                mime='text/csv'
            )
        else:
            st.warning("No remuneration data available in database.")

        st.divider()

        # 2. Bank Selector
        st.markdown("### 🏦 Bank Analysis")
        
        # Default to Base Bank if present in list
        default_banks = [base_bank_name] if base_bank_name in df_banks['name'].tolist() else []
        
        selected_banks = st.multiselect(
            "Select Banks to View",
            options=df_banks['name'].unique(),
            default=default_banks,
            key="remun_bank_selector"
        )

        if selected_banks:
            # Get LEIs for selected banks
            selected_leis = df_banks[df_banks['name'].isin(selected_banks)]['lei'].tolist()
            df_remun = get_market_financial_years(selected_leis)

            if not df_remun.empty:
                for bank_name in selected_banks:
                    # Filter for specific bank
                    bank_data = df_remun[df_remun['name'] == bank_name].copy()
                    
                    if bank_data.empty:
                        st.info(f"No detailed data available for **{bank_name}**")
                        continue

                    # Get Ticker and Country for header
                    ticker = bank_data.iloc[0]['ticker'] if pd.notna(bank_data.iloc[0]['ticker']) else "N/A"
                    country = df_banks[df_banks['name'] == bank_name].iloc[0]['country']
                    
                    st.markdown(f"#### {bank_name} ({ticker}) - {country}")
                    
                    # Target years 2021-2025
                    target_years = [2021, 2022, 2023, 2024, 2025]
                    
                    # Prepare display data
                    # Sort by FY ascending for growth/CAGR calculations
                    bank_data_sorted = bank_data[bank_data['fy'].isin(target_years)].sort_values('fy')
                    
                    if bank_data_sorted.empty:
                        st.info(f"No data within the 2021-2025 range for **{bank_name}**")
                        continue

                    # Define metrics and their characteristics
                    metrics_config = [
                        ('net_income', 'Net Income', 'currency'),
                        ('avg_market_cap', 'Market Cap', 'currency'),
                        ('dividend_amt', 'Dividends', 'currency'),
                        ('buyback_amt', 'Buybacks', 'currency'),
                        ('payout_ratio_fy', 'Payout %', 'pct'),
                        ('eps_fy', 'EPS', 'price'),
                        ('dps_fy', 'DPS', 'price'),
                        ('dividend_yield_fy', 'Div Yield', 'pct'),
                        ('buyback_yield_fy', 'BB Yield', 'pct'),
                        ('total_yield_fy', 'Total Yield', 'pct'),
                        ('rote', 'RoTE', 'pct'),
                        ('p_tbv', 'P/TBV', 'price'),
                    ]

                    # Helper for safe formatting
                    def fmt_val(val, fmt_type):
                        if pd.isna(val) or (isinstance(val, float) and np.isinf(val)): return "-"
                        if fmt_type == 'currency': return f"€{val:,.0f}"
                        if fmt_type == 'price': return f"€{val:.2f}"
                        if fmt_type == 'pct': return f"{val:.1%}"
                        if fmt_type == 'growth': return f"{val:+.1%}"
                        return str(val)

                    pivot_rows = []
                    for col_id, label, fmt_type in metrics_config:
                        if col_id not in bank_data_sorted.columns: continue
                        
                        # Get year-by-year values
                        # Use a dict to handle missing years in the range
                        year_map = dict(zip(bank_data_sorted['fy'], bank_data_sorted[col_id]))
                        row_vals = [year_map.get(y, np.nan) for y in target_years]
                        
                        # Cumulative / Average
                        if col_id in ['net_income', 'dividend_amt', 'buyback_amt']:
                            summary_val = np.nansum(row_vals) if not all(np.isnan(row_vals)) else np.nan
                            summary_label = "Cumulative"
                        else:
                            summary_val = np.nanmean(row_vals) if not all(np.isnan(row_vals)) else np.nan
                            summary_label = "Average"
                            
                        # CAGR (Based on FY data only, excluding TTM/2025)
                        fy_years = [y for y in target_years if y < 2025]
                        valid_fy_vals = [year_map.get(y) for y in fy_years if pd.notna(year_map.get(y)) and year_map.get(y) > 0]
                        
                        if len(valid_fy_vals) >= 2:
                            # Use first and last available FY in the range
                            start_fy = min([y for y in fy_years if pd.notna(year_map.get(y)) and year_map.get(y) > 0])
                            end_fy = max([y for y in fy_years if pd.notna(year_map.get(y)) and year_map.get(y) > 0])
                            
                            start_val = year_map.get(start_fy)
                            end_val = year_map.get(end_fy)
                            n_years = end_fy - start_fy
                            
                            if n_years > 0:
                                cagr = (end_val / start_val) ** (1/n_years) - 1
                            else:
                                cagr = np.nan
                        else:
                            cagr = np.nan

                        # Add Data Row
                        data_row = {'Metric': label}
                        for y, v in zip(target_years, row_vals):
                            col_name = f"{y} (TTM)" if y == 2025 else str(y)
                            data_row[col_name] = fmt_val(v, fmt_type)
                        data_row[f'{summary_label}'] = fmt_val(summary_val, fmt_type)
                        data_row['CAGR'] = fmt_val(cagr, 'pct')
                        pivot_rows.append(data_row)
                        
                        # --- VIRTUAL ROW: BUYBACK EPS RISE ---
                        if col_id == 'buyback_yield_fy':
                            accretion_row = {'Metric': "  BB EPS Rise (Est.)"}
                            for i in range(len(target_years)):
                                col_name = f"{target_years[i]} (TTM)" if target_years[i] == 2025 else str(target_years[i])
                                y = row_vals[i]
                                if pd.notna(y) and y > 0 and y < 1:
                                    # EPS Rise = 1/(1-yield) - 1
                                    accretion = (1 / (1 - y)) - 1
                                    accretion_row[col_name] = fmt_val(accretion, 'pct')
                                else:
                                    accretion_row[col_name] = "-"
                            accretion_row[f'{summary_label}'] = ""
                            accretion_row['CAGR'] = ""
                            pivot_rows.append(accretion_row)

                        # Add YoY Growth Row for primary financial metrics
                        if col_id in ['net_income', 'dividend_amt', 'buyback_amt', 'eps_fy', 'dps_fy']:
                            yoy_row = {'Metric': f"  {label} YoY %"}
                            for i in range(len(target_years)):
                                col_name = f"{target_years[i]} (TTM)" if target_years[i] == 2025 else str(target_years[i])
                                if i == 0:
                                    yoy_row[col_name] = ""
                                else:
                                    prev = row_vals[i-1]
                                    curr = row_vals[i]
                                    if pd.notna(prev) and pd.notna(curr) and prev != 0:
                                        yoy_row[col_name] = fmt_val((curr/prev) - 1, 'growth')
                                    else:
                                        yoy_row[col_name] = ""
                            yoy_row[f'{summary_label}'] = ""
                            yoy_row['CAGR'] = ""
                            pivot_rows.append(yoy_row)

                    df_pivot = pd.DataFrame(pivot_rows)
                    
                    st.table(df_pivot) # Use st.table for clearer visualization of the pivoted data
                    st.divider()
            else:
                st.warning("No data found for the selected banks.")
        else:
            st.info("Please select at least one bank to view detailed remuneration tables.")

    # =========================================================================
    # TAB 4: DATA EXPLORER
    # =========================================================================
    with tab_explorer:
        st.header("🔍 Data Explorer")
        st.markdown("Explore raw financial data from Yahoo Finance and our database.")

        # Bank Selector (independent from sidebar)
        col1, col2 = st.columns([1, 2])

        with col1:
            explorer_bank_name = st.selectbox(
                "Select Bank",
                df_banks['name'].unique(),
                key="explorer_bank"
            )

            # Get bank details
            explorer_row = df_banks[df_banks['name'] == explorer_bank_name].iloc[0]
            explorer_lei = explorer_row['lei']
            explorer_ticker = explorer_row.get('ticker', None)

            if not explorer_ticker:
                st.warning(f"⚠️ {explorer_bank_name} does not have a ticker symbol. Only database data available.")

        with col2:
            # Data Category Selector
            data_categories = [
                "Dividend History (Database)",
                "Share Actions / Splits (Database)",
                "Price & Volume History",
                "Dividends (yfinance)",
                "Stock Splits (yfinance)",
                "Cashflow Statements (Annual)",
                "Cashflow Statements (Quarterly)",
                "Income Statements (Annual)",
                "Income Statements (Quarterly)",
                "Balance Sheets (Annual)",
                "Balance Sheets (Quarterly)",
                "Raw Info Dictionary"
            ]

            data_category = st.selectbox(
                "Data Category",
                data_categories,
                key="data_category"
            )

        st.divider()

        # Fetch and display data based on selection
        df_data = pd.DataFrame()
        data_dict = None

        try:
            if data_category == "Dividend History (Database)":
                df_data = get_dividend_history(explorer_lei)
                if not df_data.empty:
                    # Select relevant columns for display
                    display_cols = ['ex_date', 'amount_local', 'amount_eur', 'currency',
                                   'dividend_type', 'fx_rate']
                    df_data = df_data[display_cols]

            elif data_category == "Share Actions / Splits (Database)":
                df_data = get_share_actions_history(explorer_lei)
                if not df_data.empty:
                    display_cols = ['action_date', 'action_type', 'ratio']
                    df_data = df_data[display_cols]

            elif explorer_ticker:
                # Yahoo Finance data
                if data_category == "Price & Volume History":
                    df_data = get_raw_yfinance_data(explorer_ticker, 'history')

                elif data_category == "Dividends (yfinance)":
                    df_data = get_raw_yfinance_data(explorer_ticker, 'dividends')

                elif data_category == "Stock Splits (yfinance)":
                    df_data = get_raw_yfinance_data(explorer_ticker, 'splits')

                elif data_category == "Cashflow Statements (Annual)":
                    df_data = get_raw_yfinance_data(explorer_ticker, 'cashflow')

                elif data_category == "Cashflow Statements (Quarterly)":
                    df_data = get_raw_yfinance_data(explorer_ticker, 'quarterly_cashflow')

                elif data_category == "Income Statements (Annual)":
                    df_data = get_raw_yfinance_data(explorer_ticker, 'income_stmt')

                elif data_category == "Income Statements (Quarterly)":
                    df_data = get_raw_yfinance_data(explorer_ticker, 'quarterly_income')

                elif data_category == "Balance Sheets (Annual)":
                    df_data = get_raw_yfinance_data(explorer_ticker, 'balance_sheet')

                elif data_category == "Balance Sheets (Quarterly)":
                    df_data = get_raw_yfinance_data(explorer_ticker, 'quarterly_balance')

                elif data_category == "Raw Info Dictionary":
                    df_data = get_raw_yfinance_data(explorer_ticker, 'info')
                    if not df_data.empty:
                        # Convert to dict for JSON display
                        data_dict = dict(zip(df_data['Field'], df_data['Value']))
            else:
                st.info("This data category requires a ticker symbol.")

        except Exception as e:
            st.error(f"Error fetching data: {str(e)[:200]}")

        # Display data
        if data_dict is not None:
            # JSON display for info dictionary
            st.subheader("📋 Info Dictionary")
            st.json(data_dict)

        elif not df_data.empty:
            # Summary stats
            summary = get_data_summary(df_data, data_category)

            col_sum1, col_sum2, col_sum3 = st.columns(3)
            with col_sum1:
                st.metric("Records", f"{summary.get('records', 0):,}")
            with col_sum2:
                st.metric("Columns", summary.get('columns', 0))
            with col_sum3:
                if 'date_range' in summary:
                    st.metric("Date Range", summary['date_range'])

            st.divider()

            # Data table
            st.subheader("📊 Data Table")

            # Display with scrolling
            st.dataframe(
                df_data,
                use_container_width=True,
                height=400
            )

            # Export button
            csv = df_data.to_csv(index=False)
            file_name = f"{explorer_ticker or explorer_lei}_{data_category.replace(' ', '_').replace('/', '_')}_{pd.Timestamp.now().strftime('%Y%m%d')}.csv"

            st.download_button(
                label="📥 Download CSV",
                data=csv,
                file_name=file_name,
                mime="text/csv",
                key="download_csv"
            )

        else:
            st.info(f"ℹ️ No data available for {explorer_bank_name} - {data_category}")

if __name__ == "__main__":
    main()
