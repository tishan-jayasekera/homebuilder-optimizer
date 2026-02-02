#!/bin/bash
# Run this from your project root to fix the folder structure

# Create directories
mkdir -p pages
mkdir -p src
mkdir -p .streamlit

# Move/rename src files (adjust source names if different)
mv src-data-loader.py src/data_loader.py 2>/dev/null || echo "data_loader.py - check manually"
mv src-normalization.py src/normalization.py 2>/dev/null || echo "normalization.py - check manually"
mv src-builder-pnl.py src/builder_pnl.py 2>/dev/null || echo "builder_pnl.py - check manually"
mv src-orphan-media.py src/orphan_media.py 2>/dev/null || echo "orphan_media.py - check manually"
mv src-referral-clusters.py src/referral_clusters.py 2>/dev/null || echo "referral_clusters.py - check manually"
mv src-network-optimization.py src/network_optimization.py 2>/dev/null || echo "network_optimization.py - check manually"
mv src-utils.py src/utils.py 2>/dev/null || echo "utils.py - check manually"
mv src-init.py src/__init__.py 2>/dev/null || echo "__init__.py - check manually"

# Move/rename page files (Streamlit requires specific naming pattern)
mv page-builder-pnl.py "pages/1_Builder_PnL.py" 2>/dev/null || echo "Builder_PnL.py - check manually"
mv page-orphan-media.py "pages/2_Orphan_Media.py" 2>/dev/null || echo "Orphan_Media.py - check manually"
mv page-referral-networks.py "pages/3_Referral_Networks.py" 2>/dev/null || echo "Referral_Networks.py - check manually"

# Move streamlit config
mv streamlit-config*.txt .streamlit/config.toml 2>/dev/null || echo "config.toml - check manually"

# Optional: Rename gitignore
mv gitignore.txt .gitignore 2>/dev/null

echo ""
echo "Expected structure:"
echo "ibn-hs-analytics/"
echo "├── app.py"
echo "├── requirements.txt"
echo "├── .streamlit/"
echo "│   └── config.toml"
echo "├── src/"
echo "│   ├── __init__.py"
echo "│   ├── data_loader.py"
echo "│   ├── normalization.py"
echo "│   ├── builder_pnl.py"
echo "│   ├── orphan_media.py"
echo "│   ├── referral_clusters.py"
echo "│   ├── network_optimization.py"
echo "│   └── utils.py"
echo "└── pages/"
echo "    ├── 1_Builder_PnL.py"
echo "    ├── 2_Orphan_Media.py"
echo "    └── 3_Referral_Networks.py"
echo ""
echo "Done! Run: streamlit run app.py"