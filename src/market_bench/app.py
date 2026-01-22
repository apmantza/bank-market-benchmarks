import streamlit as st
import pandas as pd
import sys
import os

# Add src to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from market_bench.config import DISPLAY_SETTINGS
from market_bench.data import (
    get_all_banks, get_market_data, get_market_history, 
    get_structured_benchmark_data, get_market_financial_years, get_market_fy_averages,
    get_ranking_data
)
from market_bench.plotting import plot_benchmark_bar, plot_market_history

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
    tab_bench, tab_rank = st.tabs(["📊 Benchmarking", "🏆 Ranking"])
    
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
            st.markdown("### � Download Market Data")
            csv = df_structured.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download Benchmarking Data (CSV)", data=csv, file_name='benchmarking_data.csv', mime='text/csv')

    # =========================================================================
    # TAB 2: RANKING
    # =========================================================================
    with tab_rank:
        st.subheader("🏆 EU Bank Rankings")
        df_rank = get_ranking_data()
        
        if df_rank.empty:
            st.warning("No data available.")
        else:
            rank_metric = st.selectbox("Rank By", 
                ['market_cap', 'intrinsic_value', 'upside', 'price_to_book', 'dividend_yield', 'buyback_yield', 'payout_yield', 
                 'pe_trailing', 'return_1y', 'eps_trailing', 'dps_trailing'],
                format_func=lambda x: x.replace('_', ' ').title()
            )
            
            df_rank_sorted = df_rank[['name', 'ticker', 'country', 'region', rank_metric]].dropna().sort_values(rank_metric, ascending=False).reset_index(drop=True)
            df_rank_sorted.index += 1 # 1-based ranking
            
            c_rank1, c_rank2 = st.columns([1, 2])
            with c_rank1:
                st.dataframe(df_rank_sorted.style.format({
                    'market_cap': '€{:,.0f}',
                    'intrinsic_value': '€{:.2f}',
                    'upside': '{:.1%}',
                    'price_to_book': '{:.2f}x',
                    'pe_trailing': '{:.1f}x',
                    'dividend_yield': '{:.2%}',
                    'buyback_yield': '{:.2%}',
                    'payout_yield': '{:.2%}',
                    'return_1y': '{:.1%}',
                    'eps_trailing': '€{:.2f}',
                    'dps_trailing': '€{:.2f}'
                }), use_container_width=True)
            
            with c_rank2:
                # Top 20 Chart
                top_20 = df_rank_sorted.head(20)
                st.bar_chart(top_20.set_index('name')[rank_metric])

if __name__ == "__main__":
    main()
