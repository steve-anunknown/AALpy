import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Load dataset
df = pd.read_csv("model_data.csv")

# Drop unnecessary column if present
df = df.drop(columns=["Unnamed: 0"], errors='ignore')
df = df.drop(columns=['Model'])

# Correlation Analysis
correlation_matrix = df.corr()
print("Correlation Matrix:\n", correlation_matrix)

methods = ["Random", "Linear", "Square", "Exponential"]
for method in methods:
    others = [m for m in methods if method != m]
    temp = df.drop(columns=others)

    # Scatter Plot Matrix
    sns.pairplot(temp, diag_kind='kde', plot_kws={'alpha':0.7})
    plt.suptitle(f"Feature Pairwise Relationships {method}", y=1.02)
    plt.savefig(f'feature_pairwise_relationships_{method}.pdf', format='pdf')
    plt.close()

exit()

# Standardize the data
features = ["Size", "Alphabet Size", "Longest Prefix", "Longest Suffix", "Hardness", "Random", "Linear", "Square", "Exponential"]
X = df[features]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply K-Means clustering (3 clusters as default)
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df["Cluster"] = kmeans.fit_predict(X_scaled)

# Scatter Plots with Clusters
fig, axes = plt.subplots(3, 3, figsize=(15, 15))
feature_pairs = [("Size", "Hardness"), ("Size", "Exponential"), ("Hardness", "Exponential"),
                 ("Alphabet Size", "Linear"), ("Longest Prefix", "Square"), ("Longest Suffix", "Random"),
                 ("Alphabet Size", "Random"), ("Size", "Linear"), ("Hardness", "Square")]

for ax, (x, y) in zip(axes.flat, feature_pairs):
    sns.scatterplot(ax=ax, x=df[x], y=df[y], hue=df["Cluster"], palette="Set1", alpha=0.7)
    ax.set_title(f"{x} vs {y}")
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    if x == "Hardness":
        ax.set_xscale("log")  # Log scale for hardness if needed

plt.tight_layout()
plt.savefig('scatterplots_with_clusters.pdf', format='pdf')
plt.close()

# Display Cluster Summary
cluster_summary = df.groupby("Cluster")[features].mean()
print("Cluster Summary:\n", cluster_summary)

