import os

# Get the root directory of the project
# src/market_bench/config.py -> src/market_bench -> src -> root
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Path to the database
DB_NAME = os.path.join(ROOT_DIR, 'data', 'market_data.db')

# =============================================================================
# CHART COLORS
# =============================================================================

CHART_COLORS = {
    'base_bank': '#1f77b4',     # Blue (NBG)
    'peer': '#E0E0E0',          # Light Grey
    'average': '#2ca02c',       # Green (General Avg)
    'domestic_avg': '#228B22',  # Forest Green
    'eu_avg': '#98FB98',        # Pale Green (Regional/Core/EU)
    'benchmark_line': '#32CD32', # Lime Green
    
    # Standard Categorical (Plotly Default adapted)
    'cat1': '#1f77b4',  # Blue
    'cat2': '#ff7f0e',  # Orange
    'cat3': '#2ca02c',  # Green
    'cat4': '#d62728',  # Red
    'cat5': '#9467bd',  # Purple
    'cat6': '#8c564b',  # Brown
    'cat7': '#e377c2',  # Pink
    'cat8': '#7f7f7f',  # Grey
    'cat9': '#bcbd22',  # Olive
    'cat10': '#17becf', # Cyan
}

# =============================================================================
# DISPLAY SETTINGS
# =============================================================================

DISPLAY_SETTINGS = {
    'chart_height': 450,
    'chart_height_small': 300,
    'decimal_places_percent': 1,
    'decimal_places_ratio': 2,
    'amount_unit': 1e6,  # Display in millions
    'amount_suffix': 'M',
}
