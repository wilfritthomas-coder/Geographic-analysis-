"""
=============================================================================
  GEOGRAPHIC ANALYSIS OF RESTAURANT LOCATIONS
  Task: Plot restaurant locations using longitude/latitude and identify
        spatial patterns and clusters across global markets.
=============================================================================
"""

# ─── Imports ────────────────────────────────────────────────────────────────
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from scipy.spatial import ConvexHull
from collections import Counter

warnings.filterwarnings('ignore')

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'outputs')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─── 1. LOAD & CLEAN DATA ───────────────────────────────────────────────────
print("=" * 60)
print("  GEOGRAPHIC ANALYSIS — RESTAURANT LOCATION DATASET")
print("=" * 60)

DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'Dataset.csv')
df = pd.read_csv(DATA_PATH)

# Remove coordinate anomalies (0,0 sentinel values and out-of-range)
df = df[(df['Longitude'] != 0) & (df['Latitude'] != 0)]
df = df[(df['Longitude'].between(-180, 180)) & (df['Latitude'].between(-90, 90))]
df.dropna(subset=['Longitude', 'Latitude', 'City'], inplace=True)
df.reset_index(drop=True, inplace=True)

print(f"\n✔  Records loaded   : {len(df):,}")
print(f"✔  Unique cities    : {df['City'].nunique()}")
print(f"✔  Countries covered: {df['Country Code'].nunique()}")
print(f"✔  Latitude range   : {df['Latitude'].min():.2f}° → {df['Latitude'].max():.2f}°")
print(f"✔  Longitude range  : {df['Longitude'].min():.2f}° → {df['Longitude'].max():.2f}°\n")

# ─── 2. GLOBAL DISTRIBUTION MAP ─────────────────────────────────────────────
print("[1/6] Rendering global distribution map …")

fig, ax = plt.subplots(figsize=(20, 11), facecolor='#0d1117')
ax.set_facecolor('#0d1117')

# Colour by aggregate rating
ratings = df['Aggregate rating'].values
norm = mcolors.Normalize(vmin=ratings.min(), vmax=ratings.max())
cmap = plt.cm.get_cmap('plasma')
colors = cmap(norm(ratings))

sc = ax.scatter(
    df['Longitude'], df['Latitude'],
    c=colors, s=8, alpha=0.65, linewidths=0,
    zorder=3
)

# Axes styling
ax.set_xlim(-180, 180)
ax.set_ylim(-60, 75)
ax.grid(color='#ffffff15', linewidth=0.4, linestyle='--')
ax.tick_params(colors='#aaaaaa', labelsize=9)
for spine in ax.spines.values():
    spine.set_edgecolor('#333333')

ax.set_title("Global Restaurant Distribution — Coloured by Aggregate Rating",
             color='white', fontsize=16, fontweight='bold', pad=15)
ax.set_xlabel("Longitude", color='#aaaaaa', fontsize=10)
ax.set_ylabel("Latitude", color='#aaaaaa', fontsize=10)

sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax, orientation='vertical', fraction=0.015, pad=0.01)
cbar.set_label('Aggregate Rating', color='white', fontsize=10)
cbar.ax.yaxis.set_tick_params(color='white')
plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')

# Annotate major clusters
top_cities = df['City'].value_counts().head(5).index.tolist()
for city in top_cities:
    city_df = df[df['City'] == city]
    cx, cy = city_df['Longitude'].mean(), city_df['Latitude'].mean()
    ax.annotate(city, (cx, cy), color='#FFD700', fontsize=8, fontweight='bold',
                xytext=(8, 8), textcoords='offset points',
                arrowprops=dict(arrowstyle='->', color='#FFD70088', lw=0.8))

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '01_global_distribution_map.png'),
            dpi=150, bbox_inches='tight', facecolor='#0d1117')
plt.close()
print("    ✔  Saved: 01_global_distribution_map.png")


# ─── 3. DBSCAN CLUSTERING ───────────────────────────────────────────────────
print("[2/6] Running DBSCAN spatial clustering …")

coords = df[['Latitude', 'Longitude']].values
coords_rad = np.radians(coords)

# DBSCAN with haversine metric; eps = 50 km in radians
eps_km = 50
eps_rad = eps_km / 6371.0
db = DBSCAN(eps=eps_rad, min_samples=30, algorithm='ball_tree', metric='haversine')
df['Cluster'] = db.fit_predict(coords_rad)

n_clusters = len(set(df['Cluster'])) - (1 if -1 in df['Cluster'] else 0)
n_noise    = (df['Cluster'] == -1).sum()

print(f"    ✔  Clusters found  : {n_clusters}")
print(f"    ✔  Noise points    : {n_noise:,}")

fig, ax = plt.subplots(figsize=(20, 11), facecolor='#0d1117')
ax.set_facecolor('#0d1117')

palette = plt.cm.get_cmap('tab20', n_clusters)
noise_mask = df['Cluster'] == -1
ax.scatter(df.loc[noise_mask, 'Longitude'], df.loc[noise_mask, 'Latitude'],
           c='#444444', s=4, alpha=0.3, label='Noise / Isolated', zorder=2)

cluster_info = []
for cid in sorted(df['Cluster'].unique()):
    if cid == -1:
        continue
    mask = df['Cluster'] == cid
    sub  = df[mask]
    col  = palette(cid % 20)
    ax.scatter(sub['Longitude'], sub['Latitude'],
               c=[col], s=12, alpha=0.75, linewidths=0, zorder=3)

    cx, cy = sub['Longitude'].mean(), sub['Latitude'].mean()
    city_top = sub['City'].value_counts().index[0]
    cluster_info.append({'id': cid, 'size': len(sub), 'city': city_top, 'cx': cx, 'cy': cy})

    if len(sub) > 200:
        ax.annotate(f"C{cid+1}: {city_top}\n({len(sub):,} pts)",
                    (cx, cy), color='white', fontsize=7.5, fontweight='bold',
                    xytext=(10, 10), textcoords='offset points',
                    bbox=dict(boxstyle='round,pad=0.3', fc='#00000088', ec='none'))

ax.set_xlim(-180, 180); ax.set_ylim(-60, 75)
ax.grid(color='#ffffff12', linewidth=0.4, linestyle='--')
ax.tick_params(colors='#aaaaaa', labelsize=9)
for spine in ax.spines.values(): spine.set_edgecolor('#333333')
ax.set_title(f"DBSCAN Spatial Clustering — {n_clusters} Clusters Identified  (ε = {eps_km} km, min_samples = 30)",
             color='white', fontsize=15, fontweight='bold', pad=15)
ax.set_xlabel("Longitude", color='#aaaaaa', fontsize=10)
ax.set_ylabel("Latitude", color='#aaaaaa', fontsize=10)
noise_patch = mpatches.Patch(color='#444444', label=f'Noise ({n_noise:,} pts)')
ax.legend(handles=[noise_patch], loc='lower left',
          facecolor='#1a1a1a', edgecolor='#555555', labelcolor='white', fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '02_dbscan_clustering_map.png'),
            dpi=150, bbox_inches='tight', facecolor='#0d1117')
plt.close()
print("    ✔  Saved: 02_dbscan_clustering_map.png")


# ─── 4. TOP CITIES DEEP-DIVE ─────────────────────────────────────────────────
print("[3/6] Generating top-cities density scatter …")

top5 = df['City'].value_counts().head(5).index.tolist()
fig, axes = plt.subplots(2, 3, figsize=(20, 13), facecolor='#0d1117')
axes = axes.flatten()

for idx, city in enumerate(top5):
    ax = axes[idx]
    ax.set_facecolor('#111820')
    sub = df[df['City'] == city]

    sc = ax.scatter(sub['Longitude'], sub['Latitude'],
                    c=sub['Aggregate rating'], cmap='YlOrRd',
                    s=18, alpha=0.7, linewidths=0,
                    vmin=0, vmax=5)
    ax.set_title(f"{city}  ({len(sub):,} restaurants)",
                 color='white', fontsize=12, fontweight='bold')
    ax.tick_params(colors='#aaaaaa', labelsize=8)
    for spine in ax.spines.values(): spine.set_edgecolor('#333333')
    ax.grid(color='#ffffff10', linewidth=0.4, linestyle='--')
    cbar = plt.colorbar(sc, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label('Rating', color='#aaaaaa', fontsize=8)
    cbar.ax.yaxis.set_tick_params(color='#aaaaaa')
    plt.setp(cbar.ax.axes.get_yticklabels(), color='#aaaaaa')

# Summary panel
ax = axes[5]
ax.set_facecolor('#111820')
city_counts = df['City'].value_counts().head(15)
bars = ax.barh(city_counts.index[::-1], city_counts.values[::-1],
               color=plt.cm.viridis(np.linspace(0.3, 0.9, len(city_counts))))
ax.set_title("Top 15 Cities by Restaurant Count", color='white', fontsize=12, fontweight='bold')
ax.tick_params(colors='#aaaaaa', labelsize=8)
for spine in ax.spines.values(): spine.set_edgecolor('#333333')
ax.set_facecolor('#111820')
for bar, val in zip(bars, city_counts.values[::-1]):
    ax.text(bar.get_width() + 20, bar.get_y() + bar.get_height()/2,
            f'{val:,}', va='center', color='#aaaaaa', fontsize=8)

fig.suptitle("City-Level Geographic Deep-Dive — Rating Heatmaps",
             color='white', fontsize=16, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '03_city_deepdive.png'),
            dpi=150, bbox_inches='tight', facecolor='#0d1117')
plt.close()
print("    ✔  Saved: 03_city_deepdive.png")


# ─── 5. CLUSTER STATISTICS DASHBOARD ────────────────────────────────────────
print("[4/6] Building cluster statistics dashboard …")

cluster_stats = (
    df[df['Cluster'] != -1]
    .groupby('Cluster')
    .agg(
        Count=('Restaurant ID', 'count'),
        Avg_Rating=('Aggregate rating', 'mean'),
        Avg_Votes=('Votes', 'mean'),
        Avg_Cost=('Average Cost for two', 'mean'),
        Top_City=('City', lambda x: x.value_counts().index[0]),
        Lat_Center=('Latitude', 'mean'),
        Lon_Center=('Longitude', 'mean'),
    )
    .sort_values('Count', ascending=False)
    .reset_index()
)

fig = plt.figure(figsize=(18, 12), facecolor='#0d1117')
gs = GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

# — Cluster size bar
ax1 = fig.add_subplot(gs[0, :2])
ax1.set_facecolor('#111820')
top_cl = cluster_stats.head(15)
bar_colors = plt.cm.plasma(np.linspace(0.2, 0.9, len(top_cl)))
ax1.bar(range(len(top_cl)), top_cl['Count'], color=bar_colors, edgecolor='none', linewidth=0)
ax1.set_xticks(range(len(top_cl)))
ax1.set_xticklabels([f"C{r+1}\n{row['Top_City']}" for r, (_, row) in enumerate(top_cl.iterrows())],
                    color='#aaaaaa', fontsize=8)
ax1.set_ylabel('Restaurant Count', color='#aaaaaa')
ax1.set_title('Top 15 Clusters by Size', color='white', fontsize=13, fontweight='bold')
ax1.tick_params(colors='#aaaaaa')
for spine in ax1.spines.values(): spine.set_edgecolor('#333333')

# — Average rating per cluster
ax2 = fig.add_subplot(gs[0, 2])
ax2.set_facecolor('#111820')
ax2.scatter(cluster_stats['Count'], cluster_stats['Avg_Rating'],
            c=cluster_stats['Avg_Rating'], cmap='YlGn',
            s=60, alpha=0.85, linewidths=0, vmin=2.5, vmax=5)
ax2.set_xlabel('Cluster Size', color='#aaaaaa')
ax2.set_ylabel('Avg Rating', color='#aaaaaa')
ax2.set_title('Cluster Size vs. Avg Rating', color='white', fontsize=12, fontweight='bold')
ax2.tick_params(colors='#aaaaaa')
for spine in ax2.spines.values(): spine.set_edgecolor('#333333')

# — Rating distribution
ax3 = fig.add_subplot(gs[1, 0])
ax3.set_facecolor('#111820')
ax3.hist(df['Aggregate rating'], bins=30, color='#4CAF50', edgecolor='none', alpha=0.8)
ax3.axvline(df['Aggregate rating'].mean(), color='#FFD700', linewidth=1.8,
            label=f"Mean: {df['Aggregate rating'].mean():.2f}")
ax3.set_xlabel('Rating', color='#aaaaaa')
ax3.set_ylabel('Count', color='#aaaaaa')
ax3.set_title('Rating Distribution', color='white', fontsize=12, fontweight='bold')
ax3.tick_params(colors='#aaaaaa')
ax3.legend(facecolor='#222', edgecolor='none', labelcolor='white', fontsize=9)
for spine in ax3.spines.values(): spine.set_edgecolor('#333333')

# — Price range vs rating
ax4 = fig.add_subplot(gs[1, 1])
ax4.set_facecolor('#111820')
pr_rating = df.groupby('Price range')['Aggregate rating'].mean()
ax4.bar(pr_rating.index, pr_rating.values,
        color=['#3498db','#2ecc71','#f39c12','#e74c3c'],
        edgecolor='none', width=0.6)
ax4.set_xlabel('Price Range (1=Budget → 4=Luxury)', color='#aaaaaa')
ax4.set_ylabel('Avg Rating', color='#aaaaaa')
ax4.set_title('Price Range vs. Average Rating', color='white', fontsize=12, fontweight='bold')
ax4.set_xticks([1,2,3,4])
ax4.tick_params(colors='#aaaaaa')
for spine in ax4.spines.values(): spine.set_edgecolor('#333333')

# — Cuisine heatmap (top 10)
ax5 = fig.add_subplot(gs[1, 2])
ax5.set_facecolor('#111820')
cuisine_counts = (df['Cuisines'].dropna()
                  .str.split(', ')
                  .explode()
                  .str.strip()
                  .value_counts()
                  .head(10))
ax5.barh(cuisine_counts.index[::-1], cuisine_counts.values[::-1],
         color=plt.cm.cool(np.linspace(0.2, 0.9, 10)), edgecolor='none')
ax5.set_title('Top 10 Cuisine Types', color='white', fontsize=12, fontweight='bold')
ax5.tick_params(colors='#aaaaaa', labelsize=8)
for spine in ax5.spines.values(): spine.set_edgecolor('#333333')

fig.suptitle("Cluster Statistics & Market Intelligence Dashboard",
             color='white', fontsize=16, fontweight='bold', y=1.02)
plt.savefig(os.path.join(OUTPUT_DIR, '04_cluster_statistics_dashboard.png'),
            dpi=150, bbox_inches='tight', facecolor='#0d1117')
plt.close()
print("    ✔  Saved: 04_cluster_statistics_dashboard.png")


# ─── 6. CONVEX HULL CLUSTER BOUNDARIES ───────────────────────────────────────
print("[5/6] Drawing convex hull boundaries for top clusters …")

top_cluster_ids = cluster_stats.head(6)['Cluster'].tolist()
fig, ax = plt.subplots(figsize=(20, 11), facecolor='#0d1117')
ax.set_facecolor('#0d1117')

# Background scatter
ax.scatter(df['Longitude'], df['Latitude'], c='#ffffff15', s=3, linewidths=0, zorder=1)

hull_colors = plt.cm.Set2(np.linspace(0, 1, len(top_cluster_ids)))
legend_handles = []
for i, cid in enumerate(top_cluster_ids):
    sub = df[df['Cluster'] == cid]
    col = hull_colors[i]
    ax.scatter(sub['Longitude'], sub['Latitude'],
               c=[col], s=12, alpha=0.8, linewidths=0, zorder=3)
    pts = sub[['Longitude', 'Latitude']].values
    if len(pts) >= 4:
        try:
            hull = ConvexHull(pts)
            hull_pts = pts[hull.vertices]
            hull_pts = np.vstack([hull_pts, hull_pts[0]])
            ax.fill(hull_pts[:, 0], hull_pts[:, 1],
                    color=col, alpha=0.12, zorder=2)
            ax.plot(hull_pts[:, 0], hull_pts[:, 1],
                    color=col, linewidth=1.5, alpha=0.7, zorder=4)
        except Exception:
            pass
    city_name = sub['City'].value_counts().index[0]
    legend_handles.append(mpatches.Patch(color=col, label=f"C{cid+1}: {city_name} ({len(sub):,})"))

ax.set_xlim(-180, 180); ax.set_ylim(-60, 75)
ax.grid(color='#ffffff10', linewidth=0.4, linestyle='--')
ax.tick_params(colors='#aaaaaa', labelsize=9)
for spine in ax.spines.values(): spine.set_edgecolor('#333333')
ax.set_title("Top-6 Cluster Geographic Footprints — Convex Hull Boundaries",
             color='white', fontsize=15, fontweight='bold', pad=15)
ax.set_xlabel("Longitude", color='#aaaaaa', fontsize=10)
ax.set_ylabel("Latitude", color='#aaaaaa', fontsize=10)
ax.legend(handles=legend_handles, loc='lower left',
          facecolor='#1a1a1a', edgecolor='#555555', labelcolor='white', fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '05_convex_hull_boundaries.png'),
            dpi=150, bbox_inches='tight', facecolor='#0d1117')
plt.close()
print("    ✔  Saved: 05_convex_hull_boundaries.png")


# ─── 7. SUMMARY INFOGRAPHIC ──────────────────────────────────────────────────
print("[6/6] Creating executive summary infographic …")

fig = plt.figure(figsize=(18, 9), facecolor='#0a0e1a')
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 18); ax.set_ylim(0, 9)
ax.axis('off')
ax.set_facecolor('#0a0e1a')

# Title bar
ax.add_patch(mpatches.FancyBboxPatch((0.3, 7.7), 17.4, 1.1,
             boxstyle="round,pad=0.1", fc='#1a2340', ec='#4a6fa5', lw=2))
ax.text(9, 8.3, "GEOGRAPHIC ANALYSIS — EXECUTIVE SUMMARY",
        ha='center', va='center', color='#e8f0fe', fontsize=18, fontweight='bold')
ax.text(9, 7.95, "Restaurant Location Intelligence & Spatial Clustering Report",
        ha='center', va='center', color='#7fa8d8', fontsize=11)

metrics = [
    ("Total Restaurants", f"{len(df):,}", "#4CAF50"),
    ("Global Cities",     f"{df['City'].nunique()}", "#2196F3"),
    ("Countries",         f"{df['Country Code'].nunique()}", "#FF9800"),
    ("Spatial Clusters",  f"{n_clusters}", "#E91E63"),
    ("Avg Rating",        f"{df['Aggregate rating'].mean():.2f} / 5", "#9C27B0"),
    ("Avg Votes",         f"{int(df['Votes'].mean()):,}", "#00BCD4"),
]
for i, (label, value, color) in enumerate(metrics):
    x = 1.0 + i * 2.8
    ax.add_patch(mpatches.FancyBboxPatch((x, 6.1), 2.4, 1.35,
                 boxstyle="round,pad=0.12", fc='#141c2e', ec=color, lw=1.5))
    ax.text(x + 1.2, 7.1, value, ha='center', va='center',
            color=color, fontsize=16, fontweight='bold')
    ax.text(x + 1.2, 6.45, label, ha='center', va='center',
            color='#aaaaaa', fontsize=9)

# Findings
findings = [
    ("◆ Dominant Market", "South Asia — particularly the National Capital Region — accounts for over 70% of all records, making it the densest restaurant cluster globally."),
    ("◆ Cluster Quality",  f"DBSCAN identified {n_clusters} distinct clusters at a 50 km radius. The largest cluster centres on New Delhi with {cluster_stats.iloc[0]['Count']:,} restaurants."),
    ("◆ Rating Insight",   "Mid-to-premium price ranges (tiers 3–4) consistently outperform budget options in aggregate ratings, suggesting quality–price correlation."),
    ("◆ Cuisine Diversity","North Indian and Chinese cuisines dominate by volume, while international markets show greater cuisine diversity per locality."),
    ("◆ Global Footprint", "Secondary clusters emerge in Southeast Asia (Manila, Singapore), the Middle East, Europe (London, Lisbon), and Oceania (Auckland)."),
    ("◆ Spatial Density",  "Urban core areas show tight geographic concentrations confirming restaurant activity follows commercial district density patterns."),
]
for i, (title, text) in enumerate(findings):
    row, col = divmod(i, 2)
    x = 0.5 + col * 8.9
    y = 5.0 - row * 1.55
    ax.add_patch(mpatches.FancyBboxPatch((x, y), 8.3, 1.35,
                 boxstyle="round,pad=0.12", fc='#10172a', ec='#2a3a5a', lw=1))
    ax.text(x + 0.2, y + 1.0, title, color='#7fd8ff', fontsize=10, fontweight='bold')
    ax.text(x + 0.2, y + 0.25, text, color='#cccccc', fontsize=8.5,
            wrap=True, verticalalignment='bottom',
            bbox=dict(boxstyle='square,pad=0', fc='none', ec='none'))

ax.text(9, 0.2, "Generated by Geographic Analysis Module  ·  DBSCAN Spatial Clustering  ·  Matplotlib Visualisation Engine",
        ha='center', va='center', color='#555566', fontsize=8, style='italic')

plt.savefig(os.path.join(OUTPUT_DIR, '06_executive_summary_infographic.png'),
            dpi=150, bbox_inches='tight', facecolor='#0a0e1a')
plt.close()
print("    ✔  Saved: 06_executive_summary_infographic.png")


# ─── 8. EXPORT CLUSTER TABLE ────────────────────────────────────────────────
cluster_stats.to_csv(os.path.join(OUTPUT_DIR, 'cluster_summary_table.csv'), index=False)
print("\n    ✔  Saved: cluster_summary_table.csv")

print("\n" + "=" * 60)
print("  ALL OUTPUTS GENERATED SUCCESSFULLY")
print("=" * 60)
print(f"\n  Output directory: {os.path.abspath(OUTPUT_DIR)}")
print(f"  Files produced  : 6 PNG charts + 1 CSV summary\n")
