"""
Shared utility functions for IBN HS Analytics
"""
import pandas as pd
import numpy as np


def fmt_currency(x, prefix="$"):
    """Format number as currency."""
    if pd.isna(x):
        return "—"
    return f"{prefix}{x:,.0f}"


def fmt_percent(x, decimals=1):
    """Format number as percentage."""
    if pd.isna(x):
        return "—"
    return f"{x * 100:,.{decimals}f}%"


def fmt_number(x, decimals=0):
    """Format number with commas."""
    if pd.isna(x):
        return "—"
    return f"{x:,.{decimals}f}"


def fmt_roas(x):
    """Format ROAS value."""
    if pd.isna(x) or x == 0:
        return "n/a"
    return f"{x:,.2f}x"


def get_status_color(status: str) -> str:
    """Return color for status band."""
    colors = {
        "🟢 High": "#22C55E",
        "🟡 Medium": "#EAB308",
        "🟣 Low": "#A855F7",
        "🔴 Loss": "#EF4444",
        "⚪ No media": "#9CA3AF"
    }
    return colors.get(status, "#6B7280")


def get_status_emoji(status: str) -> str:
    """Extract emoji from status string."""
    if status and len(status) >= 2:
        return status[:2].strip()
    return "⚪"


def calculate_trend(values: pd.Series, periods: int = 3) -> tuple:
    """
    Calculate trend direction and magnitude.
    
    Returns (direction, delta_pct) where direction is '↑', '↓', or '→'
    """
    if len(values) < 2:
        return "→", 0.0
    
    n = min(periods, len(values) // 2)
    if n < 1:
        return "→", 0.0
    
    recent = values.iloc[-n:].sum()
    prior = values.iloc[-2*n:-n].sum() if len(values) >= 2*n else values.iloc[:-n].sum()
    
    if prior == 0:
        return "→", 0.0
    
    delta_pct = (recent - prior) / abs(prior)
    
    if delta_pct > 0.05:
        direction = "↑"
    elif delta_pct < -0.05:
        direction = "↓"
    else:
        direction = "→"
    
    return direction, delta_pct


def safe_divide(numerator, denominator, default=np.nan):
    """Safe division that returns default for zero denominator."""
    if denominator == 0 or pd.isna(denominator):
        return default
    return numerator / denominator


def filter_dataframe(df: pd.DataFrame, filters: dict) -> pd.DataFrame:
    """
    Apply multiple filters to a DataFrame.
    
    filters is a dict like:
    {
        'column_name': value,  # exact match
        'column_name__gte': value,  # greater than or equal
        'column_name__lte': value,  # less than or equal
        'column_name__contains': value,  # string contains (case insensitive)
    }
    """
    result = df.copy()
    
    for key, value in filters.items():
        if value is None:
            continue
        
        if "__gte" in key:
            col = key.replace("__gte", "")
            result = result[result[col] >= value]
        elif "__lte" in key:
            col = key.replace("__lte", "")
            result = result[result[col] <= value]
        elif "__contains" in key:
            col = key.replace("__contains", "")
            result = result[result[col].astype(str).str.lower().str.contains(str(value).lower(), na=False)]
        else:
            result = result[result[key] == value]
    
    return result
