import numpy as np
import csv
import math
import os

# PART 3 - SEGMENTATION AND DIMENSIONALITY REDUCTION

# Load dataset
with open("data/records.csv", "r", newline="") as file:
    reader = csv.DictReader(file)
    records = list(reader)

n = len(records)

# Use continuous predictors X1 to X5
columns = [
    "customer_tenure_months",
    "earlier_ticket_count_90d",
    "sla_consumed_ratio",
    "account_value_inr_thousands",
    "satisfaction_score"
]

data = [
    [float(row[col]) for col in columns]
    for row in records
]

# Task 3.1: K-means from scratch

# Calculate column means
means = [
    sum(row[j] for row in data) / n
    for j in range(5)
]

# Calculate sample standard deviations
sds = []

for j in range(5):
    variance = sum(
        (row[j] - means[j]) ** 2
        for row in data
    ) / (n - 1)

    sds.append(math.sqrt(variance))

# Standardise data
standardised = [
    [
        (row[j] - means[j]) / sds[j]
        for j in range(5)
    ]
    for row in data
]


def kmeans(k):

    # Pinned starting centroid indices
    step = n // k

    starting_indices = [
        i * step for i in range(k)
    ]

    centroids = [
        standardised[i][:]
        for i in starting_indices
    ]

    assignments = None
    iterations = 0

    while True:

        new_assignments = []

        # Assign every row to nearest centroid
        for row in standardised:

            distances = []

            for centroid in centroids:

                distance = sum(
                    (row[j] - centroid[j]) ** 2
                    for j in range(5)
                )

                distances.append(distance)

            # First centroid wins ties
            cluster = distances.index(min(distances))

            new_assignments.append(cluster)

        iterations += 1

        # Stop when no assignment changes
        if assignments == new_assignments:
            assignments = new_assignments
            break

        assignments = new_assignments

        # Update centroids
        for cluster in range(k):

            members = [
                standardised[i]
                for i in range(n)
                if assignments[i] == cluster
            ]

            if members:

                centroids[cluster] = [
                    sum(row[j] for row in members) / len(members)
                    for j in range(5)
                ]

    # Calculate WCSS
    wcss = 0.0

    for i in range(n):

        cluster = assignments[i]

        distance = sum(
            (
                standardised[i][j] -
                centroids[cluster][j]
            ) ** 2
            for j in range(5)
        )

        wcss += distance

    return assignments, centroids, wcss, iterations


# Run K-means for K = 2, 3, 4, 5, 6
kmeans_results = {}

for k in range(2, 7):

    assignments, centroids, wcss, iterations = kmeans(k)

    kmeans_results[k] = (
        assignments,
        centroids,
        wcss,
        iterations
    )


# Task 3.2: Elbow method

wcss_values = {
    k: kmeans_results[k][2]
    for k in range(2, 7)
}

elbow_candidates = [3, 4, 5]

elbow_k = None

for k in elbow_candidates:

    previous_drop = wcss_values[k - 1] - wcss_values[k]

    next_drop = wcss_values[k] - wcss_values[k + 1]

    if next_drop < 0.5 * previous_drop:
        elbow_k = k
        break

if elbow_k is None:
    elbow_k = 5


# Save Task 3.1 and Task 3.2 results

os.makedirs("outputs", exist_ok=True)

with open("outputs/part3_results.txt", "w") as file:

    for k in range(2, 7):
        file.write(
            f"wcss_k{k} = {wcss_values[k]:.4f}\n"
        )

    file.write(f"elbow_k = {elbow_k}\n")


# Display results
print("Task 3.1 completed")

for k in range(2, 7):

    print(
        f"WCSS for K={k}: "
        f"{wcss_values[k]:.4f}"
    )

    print(
        f"Iterations for K={k}: "
        f"{kmeans_results[k][3]}"
    )

print("\nTask 3.2 completed")
print("Selected elbow K:", elbow_k)


# Task 3.3: Profile the K=3 segments

# Use K=3 assignments
assignments_k3 = kmeans_results[3][0]

# Group original rows by cluster
cluster_members = {}

for cluster in range(3):
    cluster_members[cluster] = [
        i for i in range(n)
        if assignments_k3[i] == cluster
    ]

# Order clusters by ascending mean X3
cluster_order = sorted(
    range(3),
    key=lambda c: (
        sum(data[i][2] for i in cluster_members[c])
        / len(cluster_members[c])
    )
)

segment_names = {
    cluster_order[0]: "A",
    cluster_order[1]: "B",
    cluster_order[2]: "C"
}

segment_stats = {}

for cluster in range(3):

    members = cluster_members[cluster]
    size = len(members)

    feature_means_original = [
        sum(data[i][j] for i in members) / size
        for j in range(5)
    ]

    outcome_rate = (
        sum(int(records[i]["escalated"]) for i in members)
        / size
    )

    mean_value = (
        sum(
            float(records[i]["resolution_minutes"])
            for i in members
        ) / size
    )

    segment_stats[cluster] = {
        "size": size,
        "means": feature_means_original,
        "outcome_rate": outcome_rate,
        "mean_value": mean_value
    }


# Save segment CSV
with open(
    "outputs/part3_segments.csv",
    "w",
    newline=""
) as file:

    writer = csv.writer(file)

    writer.writerow([
        "segment", "size",
        "mean_X1", "mean_X2", "mean_X3",
        "mean_X4", "mean_X5",
        "outcome_rate", "mean_value"
    ])

    for cluster in cluster_order:

        stats = segment_stats[cluster]

        writer.writerow([
            segment_names[cluster],
            stats["size"],
            *[f"{v:.4f}" for v in stats["means"]],
            f"{stats['outcome_rate']:.4f}",
            f"{stats['mean_value']:.4f}"
        ])


# Task 3.4: Hierarchical clustering

names = ["E1", "E2", "E3", "E4", "E5", "E6"]

distance_matrix = [
    [0.0, 7.9, 7.4, 2.9, 2.9, 6.1],
    [7.9, 0.0, 1.4, 5.9, 5.3, 2.0],
    [7.4, 1.4, 0.0, 5.8, 4.6, 1.3],
    [2.9, 5.9, 5.8, 0.0, 2.8, 4.5],
    [2.9, 5.3, 4.6, 2.8, 0.0, 3.4],
    [6.1, 2.0, 1.3, 4.5, 3.4, 0.0]
]


def hierarchical_clustering(linkage):

    clusters = [
        [name] for name in names
    ]

    merge_records = []
    snapshots = []

    for merge_no in range(1, 6):

        candidates = []

        # Examine every pair of active clusters
        for a in range(len(clusters)):

            for b in range(a + 1, len(clusters)):

                distances = []

                for name_a in clusters[a]:

                    for name_b in clusters[b]:

                        i = names.index(name_a)
                        j = names.index(name_b)

                        distances.append(
                            distance_matrix[i][j]
                        )

                if linkage == "single":
                    height = min(distances)

                elif linkage == "complete":
                    height = max(distances)

                else:
                    height = sum(distances) / len(distances)

                candidates.append((
                    height,
                    a,
                    b
                ))

        # Find smallest linkage distance
        candidates.sort(
            key=lambda item: (item[0], item[1], item[2])
        )

        height, a, b = candidates[0]

        cluster_a = sorted(clusters[a])
        cluster_b = sorted(clusters[b])

        # Ensure cluster_a has the smaller member
        if names.index(cluster_a[0]) > names.index(cluster_b[0]):
            cluster_a, cluster_b = cluster_b, cluster_a

        merge_records.append([
            linkage,
            merge_no,
            "+".join(cluster_a),
            "+".join(cluster_b),
            height
        ])

        # Merge clusters
        merged = sorted(cluster_a + cluster_b)

        clusters = [
            cluster for i, cluster in enumerate(clusters)
            if i not in (a, b)
        ]

        clusters.append(merged)

        # Save the clustering after this merge
        snapshots.append(
            [sorted(c) for c in clusters]
        )

        # Keep active clusters in deterministic order
        clusters.sort(
            key=lambda c: names.index(c[0])
        )

    return merge_records, snapshots


all_hierarchical_rows = []
cut_results = {}

for linkage in ["single", "complete", "average"]:

    merge_records, snapshots = hierarchical_clustering(linkage)

    all_hierarchical_rows.extend(merge_records)

    heights = [
        row[4] for row in merge_records
    ]

    # Calculate four consecutive height gaps
    gaps = [
        heights[i + 1] - heights[i]
        for i in range(4)
    ]

    largest_gap_index = max(
        range(4),
        key=lambda i: gaps[i]
    )

    merges_below_cut = largest_gap_index + 1

    clusters_at_cut = snapshots[
        merges_below_cut - 1
    ]

    cut_results[linkage] = clusters_at_cut


# Save hierarchical clustering CSV
with open(
    "outputs/part3_hierarchical.csv",
    "w",
    newline=""
) as file:

    writer = csv.writer(file)

    writer.writerow([
        "linkage",
        "merge_no",
        "cluster_a",
        "cluster_b",
        "height"
    ])

    for row in all_hierarchical_rows:

        writer.writerow([
            row[0],
            row[1],
            row[2],
            row[3],
            f"{row[4]:.4f}"
        ])


# Task 3.5: Principal Component Analysis

# Standardised 800 x 5 matrix
X = np.array(standardised, dtype=float)

# Centre the data
X_mean = np.mean(X, axis=0)

X_centered = X - X_mean

# Sample covariance matrix
covariance_matrix = (
    X_centered.T @ X_centered
) / (n - 1)

# Eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eigh(
    covariance_matrix
)

# Sort eigenvalues in descending order
order = np.argsort(eigenvalues)[::-1]

eigenvalues = eigenvalues[order]
eigenvectors = eigenvectors[:, order]

# Apply sign convention
for j in range(5):

    largest_index = np.argmax(
        np.abs(eigenvectors[:, j])
    )

    if eigenvectors[largest_index, j] < 0:
        eigenvectors[:, j] *= -1


# Explained variance ratios
eigenvalue_sum = float(np.sum(eigenvalues))

explained_ratios = (
    eigenvalues / eigenvalue_sum
)

cumulative_ratios = np.cumsum(
    explained_ratios
)

# Smallest number of components reaching 80%
components_for_80pct = next(
    i + 1
    for i, ratio in enumerate(cumulative_ratios)
    if ratio >= 0.80
)

# Keep top two eigenvectors
W = eigenvectors[:, :2]

# Project all records
Z = X_centered @ W

# Segment mean PC1 values
mean_pc1_segments = {}

for cluster in cluster_order:

    members = cluster_members[cluster]

    segment_name = segment_names[cluster]

    mean_pc1_segments[segment_name] = float(
        np.mean(Z[members, 0])
    )


# Save Task 3.3, 3.4 and 3.5 results

with open("outputs/part3_results.txt", "a") as file:

    # Task 3.3
    for cluster in cluster_order:

        segment = segment_names[cluster]
        stats = segment_stats[cluster]

        file.write(
            f"size_segment_{segment} = {stats['size']}\n"
        )

    for cluster in cluster_order:

        segment = segment_names[cluster]
        stats = segment_stats[cluster]

        for j in range(5):
            file.write(
                f"mean_X{j+1}_segment_{segment} = "
                f"{stats['means'][j]:.4f}\n"
            )

        file.write(
            f"outcome_rate_segment_{segment} = "
            f"{stats['outcome_rate']:.4f}\n"
        )

        file.write(
            f"mean_value_segment_{segment} = "
            f"{stats['mean_value']:.4f}\n"
        )

    # Task 3.4
    for linkage in ["single", "complete", "average"]:

        clusters_text = ";".join(
            "+".join(cluster)
            for cluster in cut_results[linkage]
        )

        file.write(
            f"{linkage}_clusters_at_cut = "
            f"{clusters_text}\n"
        )

    # Task 3.5
    file.write(
        f"eigenvalue_sum = {eigenvalue_sum:.4f}\n"
    )

    for j in range(5):

        file.write(
            f"eigenvalue_{j+1} = {eigenvalues[j]:.4f}\n"
        )

        file.write(
            f"explained_ratio_{j+1} = "
            f"{explained_ratios[j]:.4f}\n"
        )

        file.write(
            f"cumulative_ratio_{j+1} = "
            f"{cumulative_ratios[j]:.4f}\n"
        )

    file.write(
        f"components_for_80pct = {components_for_80pct}\n"
    )

    for j in range(5):

        file.write(
            f"pc1_loading_X{j+1} = "
            f"{W[j, 0]:.4f}\n"
        )

        file.write(
            f"pc2_loading_X{j+1} = "
            f"{W[j, 1]:.4f}\n"
        )

    # First three projected rows
    for i in range(3):

        file.write(
            f"row{i+1}_pc1 = {Z[i, 0]:.4f}\n"
        )

        file.write(
            f"row{i+1}_pc2 = {Z[i, 1]:.4f}\n"
        )

    for segment in ["A", "B", "C"]:

        file.write(
            f"mean_pc1_segment_{segment} = "
            f"{mean_pc1_segments[segment]:.4f}\n"
        )


# Display results

print("\nTask 3.3 completed")

for cluster in cluster_order:

    segment = segment_names[cluster]
    stats = segment_stats[cluster]

    print(
        f"Segment {segment}: "
        f"Size = {stats['size']}, "
        f"Outcome rate = {stats['outcome_rate']:.4f}"
    )

print("\nTask 3.4 completed")

for linkage in ["single", "complete", "average"]:

    print(
        f"{linkage} clusters at cut:",
        cut_results[linkage]
    )

print("\nTask 3.5 completed")

print("Eigenvalue sum:", round(eigenvalue_sum, 4))

print(
    "Components for 80% variance:",
    components_for_80pct
)

print("PC1 loadings:", W[:, 0])
print("PC2 loadings:", W[:, 1])

for segment in ["A", "B", "C"]:

    print(
        f"Mean PC1 for segment {segment}:",
        round(mean_pc1_segments[segment], 4)
    )

print("\nPart 3 completed")
