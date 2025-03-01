import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.patches as mpatches
import matplotlib.lines as mlines

# Load dataset
df = pd.read_csv("model_info.csv")

# Standardize numerical features for clustering
features = ["Hardness", "Sizes", "Inputs", "Longest Prefix", "Longest Suffix"]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[features])

# Apply K-Means clustering
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df["Cluster"] = kmeans.fit_predict(X_scaled)

# Define markers for different protocols
protocol_markers = {"TLS": "o", "MQTT": "s", "TCP": "D", "DTLS": "^"}
cluster_palette = sns.color_palette("Set1", n_colors=df["Cluster"].nunique())

### --- Scatter Plot of Clusters --- ###
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
scatter_plots = [
    ("Sizes", "Hardness"),
    ("Inputs", "Hardness"),
    ("Longest Prefix", "Hardness"),
    ("Longest Suffix", "Hardness"),
]

for ax, (x_feature, y_feature) in zip(axes.flat, scatter_plots):
    for protocol, marker in protocol_markers.items():
        subset = df[df["Protocol"] == protocol]
        sns.scatterplot(
            ax=ax,
            data=subset,
            x=x_feature,
            y=y_feature,
            hue="Cluster",
            style="Protocol",
            markers=protocol_markers,
            palette=cluster_palette,
            alpha=0.8,
            legend=False
        )

    ax.set_title(f"Clusters: {y_feature} vs {x_feature}")
    ax.set_xlabel(x_feature)
    ax.set_ylabel(y_feature)
    ax.set_yscale("log")

# Create legend handles for clusters (color)
cluster_handles = [
    mpatches.Patch(color=cluster_palette[i], label=f"Cluster {i}") for i in range(len(cluster_palette))
]

# Create legend handles for protocol markers
protocol_handles = [
    mlines.Line2D([], [], color='black', marker=marker, linestyle='None', markersize=8, label=protocol)
    for protocol, marker in protocol_markers.items()
]

# Add a single merged legend
legend1 = plt.legend(handles=cluster_handles, title="Clusters", loc="upper left", bbox_to_anchor=(1.02, 1))
legend2 = plt.legend(handles=protocol_handles, title="Protocols", loc="upper left", bbox_to_anchor=(1.02, 0.5))
plt.gca().add_artist(legend1)

# Adjust layout and save plots
plt.tight_layout()
plt.savefig("model_clusters_scatterplots.pdf", format="pdf")
plt.show()

### --- Hardness vs Size Figure --- ###
fig_hardness_size, ax_hardness_size = plt.subplots(figsize=(6, 5))

for protocol, marker in protocol_markers.items():
    subset = df[df["Protocol"] == protocol]
    sns.scatterplot(
        ax=ax_hardness_size,
        x=subset["Sizes"],
        y=subset["Hardness"],
        hue=subset["Cluster"],
        style=subset["Protocol"],
        markers=protocol_markers,
        palette=cluster_palette,
        alpha=0.8
    )

ax_hardness_size.set_title("Clusters: Hardness vs Size")
ax_hardness_size.set_xlabel("Size (States)")
ax_hardness_size.set_ylabel("Hardness (log scale)")
ax_hardness_size.set_yscale("log")

# Add merged legend
legend1 = ax_hardness_size.legend(handles=cluster_handles, title="Clusters", loc="upper left", bbox_to_anchor=(1.02, 1))
legend2 = ax_hardness_size.legend(handles=protocol_handles, title="Protocols", loc="upper left", bbox_to_anchor=(1.02, 0.5))
ax_hardness_size.add_artist(legend1)  # Keep the cluster legend

plt.tight_layout()
plt.savefig("hardness_vs_size.pdf", format="pdf")
plt.show()
