"""
Data loading utilities for IBN HS Analytics
"""
import pandas as pd
import streamlit as st
from io import BytesIO


@st.cache_data(show_spinner="Loading events data...")
def load_events(file) -> pd.DataFrame:
    """Load events master file from uploaded Excel."""
    if file is None:
        return None
    
    try:
        df = pd.read_excel(file)
        return df
    except Exception as e:
        st.error(f"Error loading events file: {e}")
        return None


@st.cache_data(show_spinner="Loading origin performance data...")
def load_origin_perf(file) -> pd.DataFrame:
    """Load origin performance file from uploaded Excel."""
    if file is None:
        return None
    
    try:
        df = pd.read_excel(file)
        return df
    except Exception as e:
        st.error(f"Error loading origin performance file: {e}")
        return None


@st.cache_data(show_spinner="Loading media data...")
def load_media_raw(file) -> pd.DataFrame:
    """Load media raw file from uploaded Excel."""
    if file is None:
        return None
    
    try:
        df = pd.read_excel(file)
        return df
    except Exception as e:
        st.error(f"Error loading media file: {e}")
        return None


def load_all_data():
    """
    Load all data files from session state uploads.
    Returns dict with events, origin_perf, media_raw DataFrames.
    """
    data = {}
    
    if 'events_file' in st.session_state:
        data['events'] = load_events(st.session_state['events_file'])
    
    if 'origin_file' in st.session_state:
        data['origin_perf'] = load_origin_perf(st.session_state['origin_file'])
    
    if 'media_file' in st.session_state:
        data['media_raw'] = load_media_raw(st.session_state['media_file'])
    
    return data


def export_to_excel(df: pd.DataFrame, filename: str) -> bytes:
    """Convert DataFrame to Excel bytes for download."""
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Data')
    return output.getvalue()
