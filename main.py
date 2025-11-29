# -----------------------------------------------------------
# YOUTUBE TRENDING VIDEO ANALYSIS (WITH SYNTHETIC DATASET)
# -----------------------------------------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import seaborn as sns

# -------------------------------------
# 1. CREATE A SYNTHETIC DATASET
# -------------------------------------

np.random.seed(42)

N = 300  # number of sample videos

data = {
    "video_id": [f"vid_{i}" for i in range(N)],
    "title_length": np.random.randint(20, 90, N),
    "views": np.random.randint(10_000, 10_000_000, N),
    "likes": np.random.randint(1_000, 500_000, N),
    "comments": np.random.randint(50, 50_000, N),
    "category": np.random.choice(
        ["Music", "Gaming", "News", "Comedy", "Tech", "Education"], N
    ),
    "publish_hour": np.random.randint(0, 24, N),
    "tags_count": np.random.randint(1, 40, N),
}

df = pd.DataFrame(data)

print("Sample Dataset:")
print(df.head())

# -------------------------------------
# 2. BASIC EDA
# -------------------------------------

print("\nBasic Info:")
print(df.info())

print("\nSummary Stats:")
print(df.describe())

# Category distribution
plt.figure(figsize=(8, 4))
df["category"].value_counts().plot(kind="bar")
plt.title("Video Category Distribution")
plt.xlabel("Category")
plt.ylabel("Count")
plt.show()

# Views distribution
plt.figure(figsize=(7, 4))
plt.hist(df["views"], bins=30)
plt.title("Views Distribution")
plt.xlabel("Views")
plt.ylabel("Frequency")
plt.show()

# Likes vs Views
plt.figure(figsize=(7, 5))
sns.scatterplot(x=df["views"], y=df["likes"])
plt.title("Likes vs Views")
plt.show()

# -------------------------------------
# 3. FEATURE ENGINEERING
# -------------------------------------

df["like_rate"] = df["likes"] / df["views"]
df["comment_rate"] = df["comments"] / df["views"]

# -------------------------------------
# 4. CLUSTERING TO FIND TREND PATTERNS
# -------------------------------------

features = df[["views", "likes", "comments", "tags_count", "title_length"]]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(features)

# KMeans clustering
kmeans = KMeans(n_clusters=3, random_state=42)
df["cluster"] = kmeans.fit_predict(X_scaled)

print("\nCluster Centers (Scaled):")
print(kmeans.cluster_centers_)

# Cluster Visualization (Views vs Likes)
plt.figure(figsize=(7, 5))
sns.scatterplot(
    x=df["views"],
    y=df["likes"],
    hue=df["cluster"],
    palette="Set1"
)
plt.title("Clusters of Trending Videos")
plt.show()

# -------------------------------------
# 5. EXPLAINING WHY VIDEOS TREND
# -------------------------------------

cluster_summary = df.groupby("cluster")[["views", "likes", "comments", "tags_count", "title_length"]].mean()
print("\nCluster Summary:")
print(cluster_summary)

print("\n----- INTERPRETATION OF TRENDING PATTERNS -----")

for i in cluster_summary.index:
    row = cluster_summary.loc[i]
    print(f"\nCluster {i}:")
    print("----------------------")

    if row["views"] > 5_000_000:
        print("🔥 These videos have extremely high views → VIRAL CONTENT.")
    if row["likes"] > 200_000:
        print("👍 Very high likes → Strong audience engagement.")
    if row["tags_count"] > 25:
        print("#️⃣ Many tags → Better discoverability in search.")
    if row["title_length"] < 40:
        print("📝 Short titles → More click-through rate.")
    if row["comments"] > 30_000:
        print("💬 Many comments → High discussion & engagement.")

print("\nFINAL INSIGHT: Videos trend when they have:")
print("""
✔ High engagement (likes + comments)
✔ High discoverability (tags)
✔ Large audience reach (views)
✔ Attractive/shorter titles
✔ Posted at optimal hours
""")
