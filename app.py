"""
IBN HS Analytics - Main Entry Point
Builder Economics & Referral Network Analysis Platform
"""
import streamlit as st
from pathlib import Path

st.set_page_config(
    page_title="IBN HS Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header { font-size: 2.5rem; font-weight: 700; color: #0F172A; margin-bottom: 0.5rem; }
    .sub-header { font-size: 1.1rem; color: #6B7280; margin-bottom: 2rem; }
    div[data-testid="stButton"] > button { width: 100%; }
</style>
""", unsafe_allow_html=True)


def get_available_pages():
    """Get dict of available pages with their paths."""
    pages_dir = Path(__file__).parent / "pages"
    pages = {}
    
    if not pages_dir.exists():
        return pages
    
    for f in pages_dir.iterdir():
        if f.suffix == '.py' and not f.name.startswith('_'):
            name_lower = f.name.lower()
            if 'builder' in name_lower or 'pnl' in name_lower:
                pages['pnl'] = f"pages/{f.name}"
            elif 'orphan' in name_lower:
                pages['orphan'] = f"pages/{f.name}"
            elif 'referral' in name_lower or 'network' in name_lower:
                pages['network'] = f"pages/{f.name}"
            elif 'postcode' in name_lower or 'suburb' in name_lower:
                pages['postcode'] = f"pages/{f.name}"
    
    return pages


def main():
    st.markdown('<p class="main-header">📊 IBN HS Analytics</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Builder Economics & Referral Network Analysis Platform</p>', unsafe_allow_html=True)
    
    # Data upload section in sidebar
    with st.sidebar:
        st.header("📁 Data Upload")
        st.markdown("Upload your data files to begin analysis.")
        
        events_file = st.file_uploader("Events Master (ref_events_master_*.xlsx)", type=["xlsx"], key="events_upload")
        origin_file = st.file_uploader("Origin Performance (ad_month_origin_perf.xlsx)", type=["xlsx"], key="origin_upload")
        media_file = st.file_uploader("Media Raw (media_raw_base_*.xlsx)", type=["xlsx"], key="media_upload")
        
        # Store in session state
        if events_file:
            st.session_state['events_file'] = events_file
        if origin_file:
            st.session_state['origin_file'] = origin_file
        if media_file:
            st.session_state['media_file'] = media_file
        
        st.divider()
        st.subheader("Data Status")
        
        # FIX: Use proper if/else instead of ternary (avoids DeltaGenerator display bug)
        col1, col2 = st.columns(2)
        with col1:
            if 'events_file' in st.session_state:
                st.success("Events ✓")
            else:
                st.warning("Events ✗")
        with col2:
            if 'origin_file' in st.session_state:
                st.success("Origin ✓")
            else:
                st.warning("Origin ✗")
        
        col3, _ = st.columns(2)
        with col3:
            if 'media_file' in st.session_state:
                st.success("Media ✓")
            else:
                st.warning("Media ✗")
    
    st.markdown("---")
    
    # Check if data is loaded
    if 'events_file' not in st.session_state:
        st.warning("👈 **Please upload your Events data file in the sidebar to get started.**")
    else:
        st.success("✅ **Data loaded!** Choose a dashboard below to begin analysis.")
    
    st.markdown("### 🚀 Choose a Dashboard")
    
    # Get available pages
    pages = get_available_pages()
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### 📊 Builder P&L")
        st.markdown("""
        Analyze builder economics across multiple lenses:
        - **Recipient**: Who receives the leads
        - **Payer**: Who funds the media
        - **Origin**: Original lead source
        """)
        if 'pnl' in pages:
            if st.button("📊 Open Builder P&L", key="btn_pnl", use_container_width=True):
                st.switch_page(pages['pnl'])
        else:
            st.error("Page not found: pages/1_Builder_PnL.py")
    
    with col2:
        st.markdown("#### 🎯 Orphan Media")
        st.markdown("""
        Identify wasted ad spend:
        - Zero-lead campaigns (true orphans)
        - Leads with no referrals (zombie funnels)
        - Active vs Paused analysis
        """)
        if 'orphan' in pages:
            if st.button("🎯 Open Orphan Media", key="btn_orphan", use_container_width=True):
                st.switch_page(pages['orphan'])
        else:
            st.error("Page not found: pages/2_Orphan_Media.py")
    
    with col3:
        st.markdown("#### 🔗 Referral Networks")
        st.markdown("""
        Explore referral ecosystems:
        - Community clustering (Louvain)
        - Network visualization
        - Media efficiency pathfinding
        """)
        if 'network' in pages:
            if st.button("🔗 Open Referral Networks", key="btn_network", use_container_width=True):
                st.switch_page(pages['network'])
        else:
            st.error("Page not found: pages/3_Referral_Networks.py")

    col4, col5, col6 = st.columns(3)
    with col4:
        st.markdown("#### 📍 Postcode Insights")
        st.markdown("""
        Understand location performance:
        - Postcode/suburb conversion rates
        - Campaign density hotspots
        - Opportunity targeting guidance
        """)
        if 'postcode' in pages:
            if st.button("📍 Open Postcode Insights", key="btn_postcode", use_container_width=True):
                st.switch_page(pages['postcode'])
        else:
            st.error("Page not found: pages/4_Postcode_Insights.py")
    
    # Debug info
    pages_dir = Path(__file__).parent / "pages"
    src_dir = Path(__file__).parent / "src"
    
    with st.expander("🔧 Debug: Directory Structure", expanded=True):
        st.write(f"**App location:** `{Path(__file__).parent}`")
        st.write(f"**Pages directory exists:** {pages_dir.exists()}")
        st.write(f"**Src directory exists:** {src_dir.exists()}")
        
        if not pages_dir.exists() or not src_dir.exists():
            st.error("⚠️ **Missing directories!** You need to create the folder structure.")
            st.code("""
# Run these commands in your project root:

mkdir -p pages src .streamlit

# Then rename your files:
# src-data-loader.py → src/data_loader.py
# src-normalization.py → src/normalization.py  
# src-builder-pnl.py → src/builder_pnl.py
# src-orphan-media.py → src/orphan_media.py
# src-referral-clusters.py → src/referral_clusters.py
# src-utils.py → src/utils.py
# src-init.py → src/__init__.py

# page-builder-pnl.py → pages/1_Builder_PnL.py
# page-orphan-media.py → pages/2_Orphan_Media.py  
# page-referral-networks.py → pages/3_Referral_Networks.py
            """, language="bash")
        
        if pages_dir.exists():
            found = [f.name for f in pages_dir.iterdir() if f.suffix == '.py']
            st.write(f"**Pages found:** {found if found else 'None'}")
        
        if src_dir.exists():
            found = [f.name for f in src_dir.iterdir() if f.suffix == '.py']
            st.write(f"**Src modules found:** {found if found else 'None'}")
        
        st.write(f"**Pages dict:** {pages}")
    
    # Quick stats if data is loaded
    if 'events_file' in st.session_state:
        st.markdown("---")
        st.subheader("📈 Data Preview")
        
        try:
            import pandas as pd
            
            if 'events_df' not in st.session_state:
                st.session_state['events_file'].seek(0)
                df = pd.read_excel(st.session_state['events_file'])
                st.session_state['events_df'] = df
            else:
                df = st.session_state['events_df']
            
            m1, m2, m3, m4 = st.columns(4)
            
            with m1:
                st.metric("Total Events", f"{len(df):,}")
            
            with m2:
                if 'Dest_BuilderRegionKey' in df.columns:
                    st.metric("Unique Builders", f"{df['Dest_BuilderRegionKey'].nunique():,}")
                else:
                    st.metric("Unique Builders", "N/A")
            
            with m3:
                media_col = next((c for c in ['MediaCost_referral_event', 'MediaCost_builder_touch', 'MediaCost'] if c in df.columns), None)
                if media_col:
                    st.metric("Total Media Spend", f"${df[media_col].sum():,.0f}")
                else:
                    st.metric("Total Media Spend", "N/A")
            
            with m4:
                rev_col = next((c for c in ['RPL_from_job', 'ReferralRevenue_event', 'Revenue'] if c in df.columns), None)
                if rev_col:
                    st.metric("Total Revenue", f"${df[rev_col].sum():,.0f}")
                else:
                    st.metric("Total Revenue", "N/A")
            
            with st.expander("📋 View Data Columns"):
                st.write(f"**Total columns:** {len(df.columns)}")
                st.write(df.columns.tolist())
                
        except Exception as e:
            st.error(f"Error reading file: {e}")


if __name__ == "__main__":
    main()
