# 🌍 Geographic Analysis of Restaurant Locations

> **Data Analytics Internship Project**  
> Domain: Geospatial Intelligence · Spatial Clustering · Location Analytics

---

## 📋 Project Overview

This project performs a comprehensive geographic analysis of a global restaurant dataset. Using coordinate-based mapping and unsupervised machine learning, the analysis plots restaurant locations across the world and identifies meaningful spatial patterns, density clusters, and market concentration zones.

**Core Objectives:**
- Plot every restaurant on a world coordinate map using latitude and longitude
- Apply DBSCAN (Density-Based Spatial Clustering of Applications with Noise) to discover natural geographic clusters
- Visualise city-level density and rating distributions
- Produce an executive summary with actionable findings

---

## 📁 Folder Structure

```
geographic_analysis_project/
│
├── Dataset.csv                          ← Source dataset (9,551 restaurant records)
│
├── notebooks/
│   ├── Geographic_Analysis_Notebook.ipynb  ← Step-by-step Jupyter notebook
│   └── geographic_analysis.py              ← Full analysis & chart generation script
│
├── outputs/
│   ├── 01_global_distribution_map.png      ← World map coloured by rating
│   ├── 02_dbscan_clustering_map.png        ← Spatial clusters on global map
│   ├── 03_city_deepdive.png                ← Top-5 city scatter plots
│   ├── 04_cluster_statistics_dashboard.png ← KPI & statistics dashboard
│   ├── 05_convex_hull_boundaries.png       ← Cluster boundary visualisation
│   ├── 06_executive_summary_infographic.png← One-page summary infographic
│   └── cluster_summary_table.csv           ← Per-cluster statistics (CSV)
│
├── README.md                            ← This file
└── TASK_EXPLANATION.txt                 ← Plain-English task walkthrough
```

---

## 🗃️ Dataset Description

| Column | Type | Description |
|---|---|---|
| Restaurant ID | int | Unique identifier |
| Restaurant Name | str | Name of the establishment |
| Country Code | int | Numeric country code |
| City | str | City name |
| Address / Locality | str | Street and locality details |
| **Longitude** | float | Geographic longitude coordinate |
| **Latitude** | float | Geographic latitude coordinate |
| Cuisines | str | Comma-separated cuisine tags |
| Average Cost for two | int | Average spend for two guests |
| Currency | str | Local currency |
| Has Table booking | str | Yes / No |
| Has Online delivery | str | Yes / No |
| Price range | int | 1 (Budget) → 4 (Luxury) |
| **Aggregate rating** | float | Overall customer rating (0–5) |
| Votes | int | Number of customer votes |

---

## 🔬 Methodology

### Step 1 — Data Cleaning
- Removed records with sentinel coordinates (0°, 0°)
- Validated coordinates within geographic bounds (±90° lat, ±180° lon)
- Dropped records missing city or coordinate values

### Step 2 — Global Distribution Map
- Plotted all 9,052 valid records as a scatter map
- Colour-encoded each point by aggregate rating using the `plasma` colour scale
- Annotated the five highest-density cities

### Step 3 — DBSCAN Spatial Clustering
- Converted decimal-degree coordinates to radians for haversine distance computation
- Applied DBSCAN with **ε = 50 km**, **min_samples = 30**
- Yielded **4 distinct clusters** and **1,423 noise points (isolated restaurants)**

### Step 4 — Cluster Visualisation
- Rendered each cluster in a distinct colour on the world map
- Computed cluster statistics: size, average rating, average votes, centroid
- Drew convex hull boundaries around the top-6 largest clusters

### Step 5 — Deep-Dive Analysis
- City-level scatter plots for the top 5 cities
- Rating distribution histogram with mean marker
- Price range vs. average rating bar chart
- Top cuisine breakdown

---

## 📊 Key Results

| Metric | Value |
|---|---|
| Total valid records | 9,052 |
| Unique cities | 141 |
| Countries covered | 15 |
| DBSCAN clusters found | 4 |
| Noise / isolated points | 1,423 |
| Dataset average rating | ~3.2 / 5 |

### Cluster Summary

| Cluster | Dominant City | Restaurant Count | Avg Rating |
|---|---|---|---|
| C1 | New Delhi | Largest cluster | ~3.1 |
| C2 | Gurgaon / Noida | Secondary NCR | ~3.2 |
| C3 | Manila | Southeast Asia | ~3.6 |
| C4 | London / Europe | Europe | ~4.0 |

---

## ▶️ How to Run

### Prerequisites
```bash
pip install pandas numpy matplotlib scikit-learn scipy
```

### Option A — Python Script (recommended)
```bash
cd geographic_analysis_project/notebooks
python3 geographic_analysis.py
```
All 6 charts and the cluster CSV will be saved to `../outputs/`.

### Option B — Jupyter Notebook
```bash
cd geographic_analysis_project/notebooks
jupyter notebook Geographic_Analysis_Notebook.ipynb
```
Run cells sequentially. The notebook walks through every analytical step with explanatory markdown.

---

## 📦 Output Files

| File | Purpose |
|---|---|
| `01_global_distribution_map.png` | High-level world view; shows geographic spread and rating distribution |
| `02_dbscan_clustering_map.png` | Colour-coded cluster assignments; reveals natural density zones |
| `03_city_deepdive.png` | City-level coordinate scatter; highlights intra-city density patterns |
| `04_cluster_statistics_dashboard.png` | Multi-panel KPI dashboard; cluster size, ratings, cuisines, price tiers |
| `05_convex_hull_boundaries.png` | Polygon boundaries around top clusters; quantifies geographic footprints |
| `06_executive_summary_infographic.png` | Presentation-ready one-page summary with all key metrics and findings |
| `cluster_summary_table.csv` | Machine-readable cluster statistics for further analysis or reporting |

---

## 💡 Findings & Insights

1. **South Asian NCR Market Dominance** — New Delhi and the surrounding National Capital Region collectively account for over 70% of all data points, indicating strong geographic concentration.

2. **Four Natural Clusters** — DBSCAN surfaced four spatially coherent groups: the NCR core, a secondary NCR suburb cluster, a Southeast Asian cluster centred on Manila, and a smaller European cluster.

3. **Quality–Price Correlation** — Premium-tier establishments (Price Range 4) consistently score higher ratings than budget-tier (Price Range 1), with an approximate 0.5-point gap.

4. **Cuisine Diversity** — North Indian and Chinese cuisines dominate volumetrically, while international markets exhibit broader cuisine diversity per locality.

5. **Urban Core Alignment** — Restaurant density tightly follows commercial district geography, confirming that location-selection behaviour aligns with urban footfall patterns.

---

*Data Analytics Internship Project — Geographic Analysis Module*
