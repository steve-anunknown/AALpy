import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Load dataset (Replace with actual file path if needed)
df = pd.read_csv("model_info.csv")

# Standardize numerical features for clustering
features = ["Hardness", "Sizes", "Inputs", "Longest Prefix", "Longest Suffix"]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[features])

# Apply K-Means clustering (adjust n_clusters as needed)
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
df["Cluster"] = kmeans.fit_predict(X_scaled)

# Visualization: Scatter plots of clusters
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Scatter: Hardness vs Size
sns.scatterplot(ax=axes[0, 0], x=df["Hardness"], y=df["Sizes"], hue=df["Cluster"], palette="Set1", alpha=0.8)
axes[0, 0].set_title("Clusters: Hardness vs Size")
axes[0, 0].set_xlabel("Hardness (log scale)")
axes[0, 0].set_ylabel("Size (States)")
axes[0, 0].set_xscale("log")

# Scatter: Hardness vs Inputs
sns.scatterplot(ax=axes[0, 1], x=df["Hardness"], y=df["Inputs"], hue=df["Cluster"], palette="Set1", alpha=0.8)
axes[0, 1].set_title("Clusters: Hardness vs Inputs")
axes[0, 1].set_xlabel("Hardness (log scale)")
axes[0, 1].set_ylabel("Inputs")
axes[0, 1].set_xscale("log")

# Scatter: Size vs Inputs
sns.scatterplot(ax=axes[1, 0], x=df["Sizes"], y=df["Inputs"], hue=df["Cluster"], palette="Set1", alpha=0.8)
axes[1, 0].set_title("Clusters: Size vs Inputs")
axes[1, 0].set_xlabel("Size (States)")
axes[1, 0].set_ylabel("Inputs")

# Scatter: Longest Prefix vs Longest Suffix
sns.scatterplot(ax=axes[1, 1], x=df["Longest Prefix"], y=df["Longest Suffix"], hue=df["Cluster"], palette="Set1", alpha=0.8)
axes[1, 1].set_title("Clusters: Longest Prefix vs Longest Suffix")
axes[1, 1].set_xlabel("Longest Prefix")
axes[1, 1].set_ylabel("Longest Suffix")

# Adjust layout and show plot
plt.tight_layout()
plt.savefig('model_clusters_scatterplots.pdf', format='pdf')

# Display cluster summary
cluster_summary = df.groupby("Cluster")[features].mean()
print("Cluster Summary:\n", cluster_summary)
