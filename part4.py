
from pathlib import Path
import csv
import math
import os

# Load data and recreate Task 2.1 training split

with open("data/records.csv", "r", newline="") as file:
    reader = csv.DictReader(file)
    records = list(reader)

# Separate records by outcome, preserving file order
negative_rows = [
    row for row in records
    if int(row["escalated"]) == 0
]

positive_rows = [
    row for row in records
    if int(row["escalated"]) == 1
]

# First 70% of each outcome goes to training
negative_cut = (7 * len(negative_rows)) // 10
positive_cut = (7 * len(positive_rows)) // 10

train_rows = (
    negative_rows[:negative_cut]
    + positive_rows[:positive_cut]
)

# Restore original file order
train_rows.sort(
    key=lambda row: int(row["record_id"][1:])
)

# Task 4.1: Entropy and Gini split evaluation

def entropy(positives, total):

    if total == 0:
        return 0.0

    p = positives / total

    if p == 0 or p == 1:
        return 0.0

    return (
        -p * math.log2(p)
        -(1 - p) * math.log2(1 - p)
    )


def gini(positives, total):

    if total == 0:
        return 0.0

    p = positives / total

    return 1 - (p ** 2 + (1 - p) ** 2)


# Root node statistics
total_rows = len(train_rows)

root_positives = sum(
    int(row["escalated"])
    for row in train_rows
)

root_entropy = entropy(
    root_positives,
    total_rows
)

root_gini = gini(
    root_positives,
    total_rows
)

# Four candidate splits
split_conditions = {
    "S1": lambda row: int(row["reopened_flag"]) == 1,
    "S2": lambda row: int(row["self_service_first_flag"]) == 1,
    "S3": lambda row: float(row["sla_consumed_ratio"]) > 0.5,
    "S4": lambda row: float(row["customer_tenure_months"]) < 24
}

split_results = []

for split_name, condition in split_conditions.items():

    yes_rows = [
        row for row in train_rows
        if condition(row)
    ]

    no_rows = [
        row for row in train_rows
        if not condition(row)
    ]

    n_yes = len(yes_rows)
    n_no = len(no_rows)

    positives_yes = sum(
        int(row["escalated"])
        for row in yes_rows
    )

    positives_no = sum(
        int(row["escalated"])
        for row in no_rows
    )

    # Weighted child entropy
    weighted_entropy = (
        (n_yes / total_rows) * entropy(
            positives_yes, n_yes
        )
        +
        (n_no / total_rows) * entropy(
            positives_no, n_no
        )
    )

    # Weighted child Gini
    weighted_gini = (
        (n_yes / total_rows) * gini(
            positives_yes, n_yes
        )
        +
        (n_no / total_rows) * gini(
            positives_no, n_no
        )
    )

    info_gain = root_entropy - weighted_entropy

    gini_gain = root_gini - weighted_gini

    split_results.append({
        "split": split_name,
        "n_yes": n_yes,
        "positives_yes": positives_yes,
        "n_no": n_no,
        "positives_no": positives_no,
        "info_gain": info_gain,
        "gini_gain": gini_gain
    })


# Find best splits
best_info = max(
    split_results,
    key=lambda x: x["info_gain"]
)

best_gini = max(
    split_results,
    key=lambda x: x["gini_gain"]
)

# Save Task 4.1 results

os.makedirs("outputs", exist_ok=True)

with open(
    "outputs/part4_splits.csv",
    "w",
    newline=""
) as file:

    writer = csv.writer(file)

    writer.writerow([
        "split",
        "n_yes",
        "positives_yes",
        "n_no",
        "positives_no",
        "info_gain",
        "gini_gain"
    ])

    for result in split_results:

        writer.writerow([
            result["split"],
            result["n_yes"],
            result["positives_yes"],
            result["n_no"],
            result["positives_no"],
            f"{result['info_gain']:.4f}",
            f"{result['gini_gain']:.4f}"
        ])


# Save results text
with open(
    "outputs/part4_results.txt",
    "w"
) as file:

    file.write(
        f"root_entropy = {root_entropy:.4f}\n"
    )

    file.write(
        f"root_gini = {root_gini:.4f}\n"
    )

    file.write(
        f"best_split_by_info_gain = {best_info['split']}\n"
    )

    file.write(
        f"best_split_by_gini = {best_gini['split']}\n"
    )


# Display results

print("Task 4.1 completed")

print("Training rows:", total_rows)
print("Training positives:", root_positives)

print("Root entropy:", round(root_entropy, 4))
print("Root Gini:", round(root_gini, 4))

print("\nSplit results:")

for result in split_results:

    print(
        result["split"],
        "Information Gain:",
        round(result["info_gain"], 4),
        "Gini Gain:",
        round(result["gini_gain"], 4)
    )

print("\nBest split by information gain:",
      best_info["split"])

print("Best split by Gini gain:",
      best_gini["split"])


# Task 4.2: One-question model and voting

# Recreate the exact stratified training/test split
zero_indices = [
    i for i, row in enumerate(records)
    if int(row["escalated"]) == 0
]

one_indices = [
    i for i, row in enumerate(records)
    if int(row["escalated"]) == 1
]

zero_cut = (7 * len(zero_indices)) // 10
one_cut = (7 * len(one_indices)) // 10

train_indices = (
    zero_indices[:zero_cut] +
    one_indices[:one_cut]
)

test_indices = (
    zero_indices[zero_cut:] +
    one_indices[one_cut:]
)

train_indices.sort()
test_indices.sort()

train_data = [records[i] for i in train_indices]
test_data = [records[i] for i in test_indices]

# Predictor columns in the exact required order
predictor_columns = [
    "customer_tenure_months",
    "earlier_ticket_count_90d",
    "sla_consumed_ratio",
    "account_value_inr_thousands",
    "satisfaction_score",
    "reopened_flag",
    "self_service_first_flag"
]

# Standardise using training statistics

raw_train = [
    [float(row[col]) for col in predictor_columns]
    for row in train_data
]

raw_test = [
    [float(row[col]) for col in predictor_columns]
    for row in test_data
]

train_count = len(raw_train)

feature_means = [
    sum(row[j] for row in raw_train) / train_count
    for j in range(7)
]

feature_sds = []

for j in range(7):

    variance = sum(
        (row[j] - feature_means[j]) ** 2
        for row in raw_train
    ) / (train_count - 1)

    feature_sds.append(math.sqrt(variance))


X_train = [
    [
        (row[j] - feature_means[j]) / feature_sds[j]
        for j in range(7)
    ]
    for row in raw_train
]

X_test = [
    [
        (row[j] - feature_means[j]) / feature_sds[j]
        for j in range(7)
    ]
    for row in raw_test
]

y_train = [
    int(row["escalated"])
    for row in train_data
]

y_test = [
    int(row["escalated"])
    for row in test_data
]


# Model 1: Logistic Regression

def sigmoid(z):

    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))

    exp_z = math.exp(z)

    return exp_z / (1.0 + exp_z)


logistic_weights = [0.0] * 7
logistic_bias = 0.0
learning_rate = 0.1

# Full-batch gradient descent: 1000 updates
for epoch in range(1000):

    gradient_weights = [0.0] * 7
    gradient_bias = 0.0

    for i in range(train_count):

        z = sum(
            logistic_weights[j] * X_train[i][j]
            for j in range(7)
        ) + logistic_bias

        probability = sigmoid(z)

        error = probability - y_train[i]

        for j in range(7):
            gradient_weights[j] += (
                error * X_train[i][j]
            )

        gradient_bias += error

    for j in range(7):

        gradient_weights[j] /= train_count

        logistic_weights[j] -= (
            learning_rate * gradient_weights[j]
        )

    gradient_bias /= train_count

    logistic_bias -= learning_rate * gradient_bias


logistic_probabilities = []

for row in X_test:

    z = sum(
        logistic_weights[j] * row[j]
        for j in range(7)
    ) + logistic_bias

    logistic_probabilities.append(sigmoid(z))

logistic_predictions = [
    1 if p >= 0.3 else 0
    for p in logistic_probabilities
]


# Model 2: K-Nearest Neighbours

best_k = 5

knn_probabilities = []
knn_predictions = []

for query in X_test:

    distances = []

    for i, reference in enumerate(X_train):

        distance = math.sqrt(
            sum(
                (query[j] - reference[j]) ** 2
                for j in range(7)
            )
        )

        distances.append((distance, i))

    # Stable ordering by distance and training position
    distances.sort(key=lambda item: (item[0], item[1]))

    neighbours = distances[:best_k]

    positive_count = sum(
        y_train[index]
        for distance, index in neighbours
    )

    probability = positive_count / best_k

    knn_probabilities.append(probability)

    knn_predictions.append(
        1 if positive_count >= (best_k + 1) // 2 else 0
    )


# Model 3: Decision Stump

best_split_name = best_info["split"]

best_condition = split_conditions[best_split_name]

yes_train = [
    row for row in train_data
    if best_condition(row)
]

no_train = [
    row for row in train_data
    if not best_condition(row)
]

yes_probability = (
    sum(int(row["escalated"]) for row in yes_train)
    / len(yes_train)
)

no_probability = (
    sum(int(row["escalated"]) for row in no_train)
    / len(no_train)
)

stump_probabilities = []

for row in test_data:

    if best_condition(row):
        stump_probabilities.append(yes_probability)
    else:
        stump_probabilities.append(no_probability)

stump_predictions = [
    1 if p >= 0.3 else 0
    for p in stump_probabilities
]


# Model 4: Hard Voting

hard_predictions = []

for i in range(len(test_data)):

    votes = (
        logistic_predictions[i]
        + knn_predictions[i]
        + stump_predictions[i]
    )

    hard_predictions.append(
        1 if votes >= 2 else 0
    )


# Model 5: Soft Voting

soft_probabilities = []

for i in range(len(test_data)):

    average_probability = (
        logistic_probabilities[i]
        + knn_probabilities[i]
        + stump_probabilities[i]
    ) / 3

    soft_probabilities.append(average_probability)


soft_predictions = [
    1 if p >= 0.3 else 0
    for p in soft_probabilities
]


# Evaluation metrics

def evaluate_model(predictions):

    tp = sum(
        1 for actual, predicted in zip(y_test, predictions)
        if actual == 1 and predicted == 1
    )

    fp = sum(
        1 for actual, predicted in zip(y_test, predictions)
        if actual == 0 and predicted == 1
    )

    tn = sum(
        1 for actual, predicted in zip(y_test, predictions)
        if actual == 0 and predicted == 0
    )

    fn = sum(
        1 for actual, predicted in zip(y_test, predictions)
        if actual == 1 and predicted == 0
    )

    precision = tp / (tp + fp) if tp + fp else 0.0

    recall = tp / (tp + fn) if tp + fn else 0.0

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall else 0.0
    )

    return {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }


models = [
    ("logistic", logistic_predictions),
    ("knn", knn_predictions),
    ("stump", stump_predictions),
    ("hard_vote", hard_predictions),
    ("soft_vote", soft_predictions)
]

voting_results = []

for model_name, predictions in models:

    result = evaluate_model(predictions)

    voting_results.append((model_name, result))


# Save voting CSV

with open(
    "outputs/part4_voting.csv",
    "w",
    newline=""
) as file:

    writer = csv.writer(file)

    writer.writerow([
        "model",
        "tp",
        "fp",
        "tn",
        "fn",
        "precision",
        "recall",
        "f1"
    ])

    for model_name, result in voting_results:

        writer.writerow([
            model_name,
            result["tp"],
            result["fp"],
            result["tn"],
            result["fn"],
            f"{result['precision']:.4f}",
            f"{result['recall']:.4f}",
            f"{result['f1']:.4f}"
        ])


# Save Task 4.2 text results

with open(
    "outputs/part4_results.txt",
    "a"
) as file:

    file.write("\n")

    for model_name, result in voting_results:

        for key in [
            "tp", "fp", "tn", "fn",
            "precision", "recall", "f1"
        ]:

            file.write(
                f"{model_name}_{key} = "
                f"{result[key]:.4f}\n"
            )


# Display results

print("\nTask 4.2 completed")

print("Best stump split:", best_split_name)

print("Test rows:", len(test_data))

print("\nVoting results:")

for model_name, result in voting_results:

    print("\nModel:", model_name)

    for key, value in result.items():

        print(f"{key}: {value:.4f}")



# Task 4.3: AdaBoost by hand

# Actual classes of 8 training points
actual = [
    1, 1, 1, -1, -1, -1, 1, -1
]

# Predictions of weak learner h1
h1 = [
    1, 1, -1, -1, -1, -1, -1, -1
]

# Predictions of weak learner h2
h2 = [
    -1, 1, 1, -1, 1, -1, 1, -1
]

number_of_points = 8

# Initially equal weights
weights = [
    1 / number_of_points
    for _ in range(number_of_points)
]

adaboost_rows = []


def calculate_adaboost_round(
    round_number,
    predictions,
    current_weights
):

    # Save weights before the update
    weights_before = current_weights.copy()

    # Calculate weighted classification error
    weighted_error = sum(
        current_weights[i]
        for i in range(number_of_points)
        if predictions[i] != actual[i]
    )

    # Calculate learner importance
    alpha = 0.5 * math.log(
        (1 - weighted_error) / weighted_error
    )

    # Calculate unnormalised weights
    unnormalised = []

    for i in range(number_of_points):

        new_weight = (
            current_weights[i]
            * math.exp(
                -alpha * actual[i] * predictions[i]
            )
        )

        unnormalised.append(new_weight)

    # Normalise weights
    normalising_sum = sum(unnormalised)

    updated_weights = [
        weight / normalising_sum
        for weight in unnormalised
    ]

    # Save all point-level calculations
    for i in range(number_of_points):

        adaboost_rows.append([
            round_number,
            i + 1,
            weights_before[i],
            unnormalised[i],
            updated_weights[i]
        ])

    return weighted_error, alpha, updated_weights


# Round 1: Use h1

round1_error, round1_alpha, weights = (
    calculate_adaboost_round(
        1,
        h1,
        weights
    )
)


# Round 2: Use h2

round2_error, round2_alpha, weights = (
    calculate_adaboost_round(
        2,
        h2,
        weights
    )
)


# Combined ensemble prediction

ensemble_predictions = []

for i in range(number_of_points):

    combined_score = (
        round1_alpha * h1[i]
        + round2_alpha * h2[i]
    )

    prediction = (
        1 if combined_score >= 0 else -1
    )

    ensemble_predictions.append(prediction)


# Calculate ensemble accuracy
correct_predictions = sum(
    1
    for i in range(number_of_points)
    if ensemble_predictions[i] == actual[i]
)

ensemble_accuracy = (
    correct_predictions / number_of_points
)


# Save AdaBoost CSV

with open(
    "outputs/part4_adaboost.csv",
    "w",
    newline=""
) as file:

    writer = csv.writer(file)

    writer.writerow([
        "round",
        "point",
        "weight_before",
        "weight_unnormalised",
        "weight_after"
    ])

    for row in adaboost_rows:

        writer.writerow([
            row[0],
            row[1],
            f"{row[2]:.4f}",
            f"{row[3]:.4f}",
            f"{row[4]:.4f}"
        ])


# Save results

with open(
    "outputs/part4_results.txt",
    "a"
) as file:

    file.write("\n")

    file.write(
        f"round1_error = {round1_error:.4f}\n"
    )

    file.write(
        f"round1_alpha = {round1_alpha:.4f}\n"
    )

    file.write(
        f"round2_error = {round2_error:.4f}\n"
    )

    file.write(
        f"round2_alpha = {round2_alpha:.4f}\n"
    )

    file.write(
        f"ensemble_accuracy = {ensemble_accuracy:.4f}\n"
    )


# Display results

print("\nTask 4.3 completed")

print("Round 1 error:", round(round1_error, 4))
print("Round 1 alpha:", round(round1_alpha, 4))

print("Round 2 error:", round(round2_error, 4))
print("Round 2 alpha:", round(round2_alpha, 4))

print(
    "Ensemble accuracy:",
    round(ensemble_accuracy, 4)
)

print("\nFinal ensemble predictions:")
print(ensemble_predictions)


# Task 4.4: Out-of-Bag Fraction

sample_size = 560

oob_fraction = (1 - 1 / sample_size) ** sample_size

# Save result
with open(
    "outputs/part4_results.txt",
    "a"
) as file:

    file.write("\n")
    file.write(
        f"oob_fraction = {oob_fraction:.4f}\n"
    )

# Display result
print("\nTask 4.4 completed")

print(
    "Out-of-bag fraction:",
    round(oob_fraction, 4)
)


# Task 4.5 - Neural Network Architecture

# Create output folder if it does not exist
Path("outputs").mkdir(exist_ok=True)

# Model A - Option 1: Fully Connected Network

mlp_layers = [
    ("mlp_hidden1", "dense", "(16,)", 784 * 16 + 16),
    ("mlp_hidden2", "dense", "(16,)", 16 * 16 + 16),
    ("mlp_output", "dense", "(10,)", 16 * 10 + 10)
]

mlp_total = sum(layer[3] for layer in mlp_layers)


# Model A - Option 2: CNN

cnn_layers = [
    ("cnn_conv1", "convolution", "(24, 24, 5)",
     5 * 5 * 3 * 5 + 5),

    ("cnn_conv2", "convolution", "(8, 8, 5)",
     5 * 5 * 5 * 5 + 5),

    ("cnn_dense1", "dense", "(32,)",
     80 * 32 + 32),

    ("cnn_dense2", "dense", "(10,)",
     32 * 10 + 10)
]

cnn_total = sum(layer[3] for layer in cnn_layers)


# Model B - Recurrent Neural Network (RNN)

rnn_wxh = 64 * 128
rnn_whh = 128 * 128
rnn_bh = 128
rnn_why = 128 * 16
rnn_by = 16

rnn_total = (
    rnn_wxh + rnn_whh + rnn_bh +
    rnn_why + rnn_by
)


# Save Model A architecture CSV

architecture_rows = mlp_layers + cnn_layers

with open("outputs/part4_architecture.csv",
          "w", newline="") as file:

    writer = csv.writer(file)

    writer.writerow([
        "layer",
        "layer_type",
        "output_shape",
        "parameters"
    ])

    for layer in architecture_rows:
        writer.writerow(layer)


# Display results

print("\nTask 4.5 completed")

print("\nModel A - Fully Connected Network")

for layer in mlp_layers:
    print(f"{layer[0]}: Shape={layer[2]}, "
          f"Parameters={layer[3]}")

print("MLP total parameters:", mlp_total)


print("\nModel A - Convolutional Neural Network")

for layer in cnn_layers:
    print(f"{layer[0]}: Shape={layer[2]}, "
          f"Parameters={layer[3]}")

print("CNN total parameters:", cnn_total)


print("\nModel B - Recurrent Neural Network")

print("W_xh parameters:", rnn_wxh)
print("W_hh parameters:", rnn_whh)
print("b_h parameters:", rnn_bh)
print("W_hy parameters:", rnn_why)
print("b_y parameters:", rnn_by)

print("RNN total parameters:", rnn_total)

print("\nArchitecture CSV saved successfully.")
