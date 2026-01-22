import plotly.graph_objects as go
import pandas as pd
from .config import CHART_COLORS, DISPLAY_SETTINGS

# --- HELPER FUNCTIONS ---

def sort_with_base_first(df, base_bank_name, metric_col, ascending=False):
    """Ensures the base bank is first, then sorts peers by value, and places benchmarks at end."""
    if df.empty: return df
    is_avg = df['name'].str.contains('Avg|Average|Regional|Core|CEE', case=False, na=False)
    base_df = df[(df['name'] == base_bank_name) & (~is_avg)].copy()
    bench_df = df[is_avg].copy()
    peers_df = df[(df['name'] != base_bank_name) & (~is_avg)].copy()
    peers_df = peers_df.sort_values(metric_col, ascending=ascending)
    return pd.concat([base_df, peers_df, bench_df], ignore_index=True)

def apply_standard_layout(fig, title, height=450, xaxis_type=None, yaxis_tickformat=None):
    """Standardizes chart layout."""
    fig.update_layout(
        title=dict(text=title, font=dict(size=16), y=0.95),
        height=height, 
        margin=dict(l=20, r=20, t=80, b=40),
        legend=dict(
            orientation="h", 
            yanchor="bottom", y=1.02, xanchor="right", x=1
        ),
        paper_bgcolor='white',
        plot_bgcolor='white',
        autosize=True,
        xaxis=dict(type=xaxis_type, tickangle=45, showgrid=False) if xaxis_type else dict(tickangle=45, showgrid=False),
        yaxis=dict(showgrid=True, gridcolor='#f0f0f0', tickformat=yaxis_tickformat)
    )
    fig.update_layout(font=dict(family="Inter, sans-serif"))
    return fig

def get_color_sequence(n=None):
    base_seq = [CHART_COLORS[f'cat{i+1}'] for i in range(10)]
    if n is None: return base_seq
    full_seq = []
    for i in range(n): full_seq.append(base_seq[i % len(base_seq)])
    return full_seq

# --- PLOTTING FUNCTIONS ---

def plot_benchmark_bar(df, metric_col, title, base_bank_name, format_pct=True, height=400, scale_amounts=True, sort_by_value=True):
    """Standard Benchmark Bar Chart with Base Bank First"""
    if metric_col not in df.columns or df[metric_col].isna().all():
        return go.Figure().update_layout(title=f"{title} (No Data)")
    
    if sort_by_value:
        df_plot = sort_with_base_first(df, base_bank_name, metric_col)
    else:
        # Respect input order, but ensure base bank is highlighted if present
        df_plot = df.copy()
    
    colors = []
    for x in df_plot['name']:
        if x == base_bank_name: colors.append(CHART_COLORS['base_bank'])
        elif "Domestic" in x: colors.append(CHART_COLORS['domestic_avg'])
        elif "Regional" in x: colors.append(CHART_COLORS['eu_avg']) # Using generic average color
        elif "Core" in x: colors.append(CHART_COLORS['eu_avg'])
        elif "CEE" in x and "Avg" in x: colors.append(CHART_COLORS['eu_avg'])
        elif "EU" in x: colors.append(CHART_COLORS['benchmark_line'])
        elif "Avg" in x: colors.append(CHART_COLORS['average'])
        else: colors.append(CHART_COLORS['peer'])

    if not format_pct and scale_amounts:
        # Check magnitude? No, just assume usually needs scaling if requested
        # But market cap is absolute, ratios are not.
        pass 

    fig = go.Figure(data=[go.Bar(
        x=df_plot['name'], 
        y=df_plot[metric_col], 
        marker_color=colors,
        text=df_plot[metric_col], 
        texttemplate='%{y:.1%}' if format_pct else '%{y:,.2f}', 
        textposition='auto',
        hovertemplate='%{x}: %{y:.2%}' if format_pct else '%{x}: %{y:,.2f}<extra></extra>'
    )])
    
    y_fmt = '.1%' if format_pct else None
    return apply_standard_layout(fig, title, height, yaxis_tickformat=y_fmt)

def plot_market_history(df, metric, title, base_bank_name, format_pct=False, show_legend=True):
    """Plots historical market data trends for multiple banks."""
    if df.empty or metric not in df.columns: return go.Figure()
    
    fig = go.Figure()
    banks = df['name'].unique()
    colors = get_color_sequence(len(banks))
    
    for i, bank in enumerate(banks):
        df_bank = df[df['name'] == bank].sort_values('date')
        is_base = bank == base_bank_name
        color = CHART_COLORS['base_bank'] if is_base else colors[i % len(colors)]
        width = 3 if is_base else 1.5
        opacity = 1.0 if is_base else 0.6
        
        hovertemplate = f"<b>{bank}</b><br>%{{x}}<br>{metric}: %{{y:.2%}}<extra></extra>" if format_pct else f"<b>{bank}</b><br>%{{x}}<br>{metric}: %{{y:,.2f}}<extra></extra>"
        
        fig.add_trace(go.Scatter(
            x=df_bank['date'], y=df_bank[metric],
            name=bank, mode='lines',
            line=dict(color=color, width=width), opacity=opacity,
            hovertemplate=hovertemplate
        ))
    
    y_fmt = '.1%' if format_pct else None
    fig = apply_standard_layout(fig, title, xaxis_type='date', yaxis_tickformat=y_fmt)
    if not show_legend: fig.update_layout(showlegend=False)
    return fig
