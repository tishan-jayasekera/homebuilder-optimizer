# IBN HS Analytics - Streamlit Application

## Repository Structure

```
ibn-hs-analytics/
├── .streamlit/
│   └── config.toml              # Streamlit configuration
├── src/
│   ├── __init__.py
│   ├── data_loader.py           # File upload & data loading
│   ├── normalization.py         # Data cleaning & normalization
│   ├── builder_pnl.py           # Builder P&L calculations
│   ├── orphan_media.py          # Phase 3 - Orphan media analysis
│   ├── referral_clusters.py     # Phase 4 - Network clustering
│   └── utils.py                 # Shared utilities
├── pages/
│   ├── 1_📊_Builder_PnL.py      # Builder economics dashboard
│   ├── 2_🎯_Orphan_Media.py     # Orphan media analysis
│   └── 3_🔗_Referral_Networks.py # Referral ecosystem explorer
├── app.py                        # Main entry point (Home page)
├── requirements.txt
├── README.md
└── .gitignore
```

## Setup Instructions

### 1. Clone & Install
```bash
git clone https://github.com/your-org/ibn-hs-analytics.git
cd ibn-hs-analytics
pip install -r requirements.txt
```

### 2. Run Locally
```bash
streamlit run app.py
```

### 3. Deploy to Streamlit Cloud
1. Push to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your repository
4. Deploy

## Required Data Files

Upload these Excel files via the sidebar:
- `ref_events_master_v5_referral_cost.xlsx` - Event-level data
- `ad_month_origin_perf.xlsx` - Monthly origin spend bridge
- `media_raw_base_phase0.xlsx` - Daily media export

## Features

### Builder P&L Dashboard
- **Lens options**: Recipient, Payer, Origin
- **Time grains**: All-time, Monthly, Weekly
- **Date basis**: Lead date or Referral date
- **Metrics**: Revenue, Media Cost, Profit, ROAS, Margin

### Orphan Media Analysis
- Identify wasted ad spend (no leads generated)
- Active vs Paused campaign analysis
- Kill list generation for optimization

### Referral Network Explorer
- Community detection (Louvain clustering)
- Network visualization with Plotly
- Media efficiency pathfinding
- Downstream cascade analysis
