
import csv
import math
import os

# PART 2 - CLASSIFICATION AND EVALUATION

# Load dataset
with open("data/records.csv", "r", newline="") as file:
    reader = csv.DictReader(file)
    records = list(reader)

n = len(records)

# Predictor columns X1 to X7
predictor_columns = [
    "customer_tenure_months",
    "earlier_ticket_count_90d",
    "sla_consumed_ratio",
    "account_value_inr_thousands",
    "satisfaction_score",
    "reopened_flag",
    "self_service_first_flag"
]

X = [
    [float(row[col]) for col in predictor_columns]
    for row in records
]

y = [int(row["escalated"]) for row in records]


# Task 2.1: Stratified 70/30 split

# Separate row indices by outcome
zero_indices = [i for i in range(n) if y[i] == 0]
one_indices = [i for i in range(n) if y[i] == 1]

# First 70% of each class goes to training
zero_train_count = (7 * len(zero_indices)) // 10
one_train_count = (7 * len(one_indices)) // 10

train_indices = (
    zero_indices[:zero_train_count] +
    one_indices[:one_train_count]
)

test_indices = (
    zero_indices[zero_train_count:] +
    one_indices[one_train_count:]
)

# Restore original file order
train_indices.sort()
test_indices.sort()

X_train = [X[i] for i in train_indices]
y_train = [y[i] for i in train_indices]

X_test = [X[i] for i in test_indices]
y_test = [y[i] for i in test_indices]

train_rows = len(train_indices)
train_positives = sum(y_train)

test_rows = len(test_indices)
test_positives = sum(y_test)

# Training means
feature_means = [
    sum(row[j] for row in X_train) / train_rows
    for j in range(7)
]

# Training sample standard deviations
feature_sds = []

for j in range(7):
    variance = sum(
        (row[j] - feature_means[j]) ** 2
        for row in X_train
    ) / (train_rows - 1)

    feature_sds.append(math.sqrt(variance))

# Standardise using training statistics
X_train_std = [
    [
        (row[j] - feature_means[j]) / feature_sds[j]
        for j in range(7)
    ]
    for row in X_train
]

X_test_std = [
    [
        (row[j] - feature_means[j]) / feature_sds[j]
        for j in range(7)
    ]
    for row in X_test
]


# Task 2.2: Logistic regression from scratch

def sigmoid(z):
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))

    exp_z = math.exp(z)
    return exp_z / (1.0 + exp_z)


def binary_cross_entropy(X_data, y_data, weights, bias):
    total_loss = 0.0

    for i in range(len(X_data)):
        z = bias + sum(
            weights[j] * X_data[i][j]
            for j in range(7)
        )

        p = sigmoid(z)

        p = max(1e-15, min(1 - 1e-15, p))

        total_loss += -(
            y_data[i] * math.log(p) +
            (1 - y_data[i]) * math.log(1 - p)
        )

    return total_loss / len(X_data)


# Initial weights and bias
weights = [0.0] * 7
bias = 0.0

learning_rate = 0.1

# Initial loss before updates
loss_epoch_0 = binary_cross_entropy(
    X_train_std, y_train, weights, bias
)

loss_history = {}

# Perform 1000 gradient descent updates
for epoch in range(1, 1001):

    weight_gradients = [0.0] * 7
    bias_gradient = 0.0

    for i in range(train_rows):

        z = bias + sum(
            weights[j] * X_train_std[i][j]
            for j in range(7)
        )

        p = sigmoid(z)
        error = p - y_train[i]

        for j in range(7):
            weight_gradients[j] += (
                error * X_train_std[i][j]
            )

        bias_gradient += error

    # Average gradients and update parameters
    for j in range(7):
        weight_gradients[j] /= train_rows

        weights[j] -= (
            learning_rate * weight_gradients[j]
        )

    bias_gradient /= train_rows

    bias -= learning_rate * bias_gradient

    # Record loss after specified updates
    if epoch in [1, 10, 100, 1000]:
        loss_history[epoch] = binary_cross_entropy(
            X_train_std, y_train, weights, bias
        )

# Final test loss
test_bce = binary_cross_entropy(
    X_test_std, y_test, weights, bias
)


# Save Task 2.1 and Task 2.2 results

os.makedirs("outputs", exist_ok=True)

with open("outputs/part2_results.txt", "w") as file:

    file.write(f"train_rows = {train_rows}\n")
    file.write(f"train_positives = {train_positives}\n")
    file.write(f"test_rows = {test_rows}\n")
    file.write(f"test_positives = {test_positives}\n")

    file.write(f"loss_epoch_0 = {loss_epoch_0:.4f}\n")
    file.write(f"loss_epoch_1 = {loss_history[1]:.4f}\n")
    file.write(f"loss_epoch_10 = {loss_history[10]:.4f}\n")
    file.write(f"loss_epoch_100 = {loss_history[100]:.4f}\n")
    file.write(f"loss_epoch_1000 = {loss_history[1000]:.4f}\n")

    for j in range(7):
        file.write(f"weight_x{j+1} = {weights[j]:.4f}\n")

    file.write(f"bias = {bias:.4f}\n")
    file.write(f"test_bce = {test_bce:.4f}\n")


# Display results
print("Task 2.1 completed")
print("Training rows:", train_rows)
print("Training positives:", train_positives)
print("Testing rows:", test_rows)
print("Testing positives:", test_positives)

print("\nTask 2.2 completed")
print("Loss before training:", round(loss_epoch_0, 4))
print("Loss after 1 epoch:", round(loss_history[1], 4))
print("Loss after 10 epochs:", round(loss_history[10], 4))
print("Loss after 100 epochs:", round(loss_history[100], 4))
print("Loss after 1000 epochs:", round(loss_history[1000], 4))

for j in range(7):
    print(f"Weight X{j+1}:", round(weights[j], 4))

print("Bias:", round(bias, 4))
print("Test BCE:", round(test_bce, 4))


# Task 2.3: Learning-rate effect

learning_rates = [0.01, 0.1, 1.0]

learning_rate_losses = {}

for eta in learning_rates:

    # Restart from zero for every learning rate
    w = [0.0] * 7
    b = 0.0

    # Run 200 gradient descent updates
    for epoch in range(200):

        gradients = [0.0] * 7
        bias_grad = 0.0

        for i in range(train_rows):

            z = b + sum(
                w[j] * X_train_std[i][j]
                for j in range(7)
            )

            probability = sigmoid(z)
            error = probability - y_train[i]

            for j in range(7):
                gradients[j] += (
                    error * X_train_std[i][j]
                )

            bias_grad += error

        # Average gradients
        for j in range(7):
            gradients[j] /= train_rows

            w[j] -= eta * gradients[j]

        bias_grad /= train_rows
        b -= eta * bias_grad

    # Calculate loss after 200 updates
    final_loss = binary_cross_entropy(
        X_train_std, y_train, w, b
    )

    learning_rate_losses[eta] = final_loss


# Task 2.4: Odds ratios

odds_ratios = [
    math.exp(weight)
    for weight in weights
]


# Save Task 2.3 and Task 2.4 results

with open("outputs/part2_results.txt", "a") as file:

    file.write(
        f"loss_epoch_200_eta_0.01 = "
        f"{learning_rate_losses[0.01]:.4f}\n"
    )

    file.write(
        f"loss_epoch_200_eta_0.1 = "
        f"{learning_rate_losses[0.1]:.4f}\n"
    )

    file.write(
        f"loss_epoch_200_eta_1.0 = "
        f"{learning_rate_losses[1.0]:.4f}\n"
    )

    for j in range(7):
        file.write(
            f"odds_ratio_x{j+1} = "
            f"{odds_ratios[j]:.4f}\n"
        )


# Display results
print("\nTask 2.3 completed")

for eta in learning_rates:
    print(
        f"Loss at learning rate {eta}:",
        round(learning_rate_losses[eta], 4)
    )

print("\nTask 2.4 completed")

for j in range(7):
    print(
        f"Odds ratio X{j+1}:",
        round(odds_ratios[j], 4)
    )


# Task 2.5: Evaluation at two thresholds

# Predict probabilities for test records
test_probabilities = []

for row in X_test_std:

    z = bias + sum(
        weights[j] * row[j]
        for j in range(7)
    )

    test_probabilities.append(sigmoid(z))


# Evaluation function
def evaluate_threshold(actual, probabilities, threshold):

    tp = 0
    fp = 0
    tn = 0
    fn = 0

    for i in range(len(actual)):

        predicted = 1 if probabilities[i] >= threshold else 0

        if predicted == 1 and actual[i] == 1:
            tp += 1

        elif predicted == 1 and actual[i] == 0:
            fp += 1

        elif predicted == 0 and actual[i] == 0:
            tn += 1

        else:
            fn += 1

    accuracy = (tp + tn) / len(actual)

    precision = tp / (tp + fp) if (tp + fp) else 0.0

    recall = tp / (tp + fn) if (tp + fn) else 0.0

    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) else 0.0
    )

    specificity = tn / (tn + fp) if (tn + fp) else 0.0

    return {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "specificity": specificity
    }


# Evaluate at threshold 0.5
metrics_05 = evaluate_threshold(
    y_test,
    test_probabilities,
    0.5
)

# Evaluate at threshold 0.3
metrics_03 = evaluate_threshold(
    y_test,
    test_probabilities,
    0.3
)

# Blind accuracy: predict every record as negative
blind_accuracy = sum(
    1 for actual in y_test if actual == 0
) / len(y_test)


# Save Task 2.5 results

with open("outputs/part2_results.txt", "a") as file:

    for key, value in metrics_05.items():
        file.write(f"tau_0.5_{key} = {value:.4f}\n")

    for key, value in metrics_03.items():
        file.write(f"tau_0.3_{key} = {value:.4f}\n")

    file.write(f"blind_accuracy = {blind_accuracy:.4f}\n")


# Display results
print("\nTask 2.5 completed")

print("\nThreshold 0.5")

for key, value in metrics_05.items():
    print(f"{key}: {value:.4f}")

print("\nThreshold 0.3")

for key, value in metrics_03.items():
    print(f"{key}: {value:.4f}")

print("\nBlind accuracy:", round(blind_accuracy, 4))


# Task 2.6: Threshold sweep

threshold_rows = []

for tau in [0.1, 0.2, 0.3, 0.4, 0.5,
            0.6, 0.7, 0.8, 0.9]:

    result = evaluate_threshold(
        y_test,
        test_probabilities,
        tau
    )

    tp = result["tp"]
    fp = result["fp"]
    tn = result["tn"]
    fn = result["fn"]

    tpr = tp / (tp + fn) if (tp + fn) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0

    threshold_rows.append([
        tau, tp, fp, tn, fn, tpr, fpr
    ])


# Save threshold sweep CSV
with open(
    "outputs/part2_thresholds.csv",
    "w",
    newline=""
) as file:

    writer = csv.writer(file)

    writer.writerow([
        "tau", "tp", "fp", "tn", "fn", "tpr", "fpr"
    ])

    writer.writerows(threshold_rows)


print("\nTask 2.6 completed")
print("Threshold sweep saved successfully.")


# Task 2.7: KNN with 5-fold cross-validation

k_values = list(range(1, 22, 2))

# Store F1 scores for each fold
fold_f1_scores = {
    k: [] for k in k_values
}

# Five-fold cross-validation
for fold in range(5):

    validation_positions = [
        i for i in range(train_rows)
        if i % 5 == fold
    ]

    reference_positions = [
        i for i in range(train_rows)
        if i % 5 != fold
    ]

    # Store predictions for every K
    fold_predictions = {
        k: [] for k in k_values
    }

    fold_actual = [
        y_train[i] for i in validation_positions
    ]

    for query_position in validation_positions:

        query = X_train_std[query_position]

        distances = []

        # Calculate Euclidean distances
        for ref_position in reference_positions:

            reference = X_train_std[ref_position]

            distance_squared = sum(
                (query[j] - reference[j]) ** 2
                for j in range(7)
            )

            distance = math.sqrt(distance_squared)

            distances.append((
                distance,
                ref_position,
                y_train[ref_position]
            ))

        # Stable sorting: distance, then file-order position
        distances.sort(key=lambda item: (item[0], item[1]))

        positive_count = 0

        # Calculate predictions for K = 1, 3, ..., 21
        for rank, item in enumerate(distances[:21], start=1):

            positive_count += item[2]

            if rank in k_values:

                prediction = (
                    1 if positive_count >= (rank + 1) // 2
                    else 0
                )

                fold_predictions[rank].append(prediction)

    # Calculate F1 separately for each fold
    for k in k_values:

        predictions = fold_predictions[k]

        tp = sum(
            1 for i in range(len(fold_actual))
            if predictions[i] == 1 and fold_actual[i] == 1
        )

        fp = sum(
            1 for i in range(len(fold_actual))
            if predictions[i] == 1 and fold_actual[i] == 0
        )

        fn = sum(
            1 for i in range(len(fold_actual))
            if predictions[i] == 0 and fold_actual[i] == 1
        )

        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0

        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall else 0.0
        )

        fold_f1_scores[k].append(f1)


# Calculate mean cross-validation F1
mean_cv_f1 = {
    k: sum(fold_f1_scores[k]) / 5
    for k in k_values
}

# Choose highest F1, ties go to smaller K
knn_best_k = min(
    k_values,
    key=lambda k: (-mean_cv_f1[k], k)
)


# Save KNN cross-validation results
with open(
    "outputs/part2_knn_cv.csv",
    "w",
    newline=""
) as file:

    writer = csv.writer(file)

    writer.writerow(["K", "mean_cv_f1"])

    for k in k_values:
        writer.writerow([k, f"{mean_cv_f1[k]:.4f}"])


# Refit KNN using all 560 training rows

knn_predictions = []

for query in X_test_std:

    distances = []

    for i in range(train_rows):

        reference = X_train_std[i]

        distance_squared = sum(
            (query[j] - reference[j]) ** 2
            for j in range(7)
        )

        distances.append((
            math.sqrt(distance_squared),
            i,
            y_train[i]
        ))

    distances.sort(key=lambda item: (item[0], item[1]))

    nearest = distances[:knn_best_k]

    positive_count = sum(
        item[2] for item in nearest
    )

    prediction = (
        1 if positive_count >= (knn_best_k + 1) // 2
        else 0
    )

    knn_predictions.append(prediction)


# Calculate final test metrics
knn_tp = sum(
    1 for i in range(test_rows)
    if knn_predictions[i] == 1 and y_test[i] == 1
)

knn_fp = sum(
    1 for i in range(test_rows)
    if knn_predictions[i] == 1 and y_test[i] == 0
)

knn_tn = sum(
    1 for i in range(test_rows)
    if knn_predictions[i] == 0 and y_test[i] == 0
)

knn_fn = sum(
    1 for i in range(test_rows)
    if knn_predictions[i] == 0 and y_test[i] == 1
)

knn_accuracy = (knn_tp + knn_tn) / test_rows

knn_precision = (
    knn_tp / (knn_tp + knn_fp)
    if knn_tp + knn_fp else 0.0
)

knn_recall = (
    knn_tp / (knn_tp + knn_fn)
    if knn_tp + knn_fn else 0.0
)

knn_f1 = (
    2 * knn_precision * knn_recall /
    (knn_precision + knn_recall)
    if knn_precision + knn_recall else 0.0
)


# Save Task 2.7 results

with open("outputs/part2_results.txt", "a") as file:

    file.write(f"knn_best_k = {knn_best_k}\n")
    file.write(f"knn_tp = {knn_tp}\n")
    file.write(f"knn_fp = {knn_fp}\n")
    file.write(f"knn_tn = {knn_tn}\n")
    file.write(f"knn_fn = {knn_fn}\n")
    file.write(f"knn_accuracy = {knn_accuracy:.4f}\n")
    file.write(f"knn_precision = {knn_precision:.4f}\n")
    file.write(f"knn_recall = {knn_recall:.4f}\n")
    file.write(f"knn_f1 = {knn_f1:.4f}\n")


# Display results
print("\nTask 2.7 completed")

print("Mean cross-validation F1:")

for k in k_values:
    print(f"K = {k}: {mean_cv_f1[k]:.4f}")

print("\nBest K:", knn_best_k)

print("\nKNN test results:")
print("TP:", knn_tp)
print("FP:", knn_fp)
print("TN:", knn_tn)
print("FN:", knn_fn)
print("Accuracy:", round(knn_accuracy, 4))
print("Precision:", round(knn_precision, 4))
print("Recall:", round(knn_recall, 4))
print("F1:", round(knn_f1, 4))


# Task 2.8: Choose alert threshold with money

# Calculate training probabilities using Task 2.2 model
train_probabilities = []

for row in X_train_std:

    z = bias + sum(
        weights[j] * row[j]
        for j in range(7)
    )

    train_probabilities.append(sigmoid(z))


# Test thresholds from 0.05 to 0.95
candidate_thresholds = [
    k / 20 for k in range(1, 20)
]

training_costs = []

for tau in candidate_thresholds:

    result = evaluate_threshold(
        y_train,
        train_probabilities,
        tau
    )

    cost = (
        50 * result["fp"] +
        400 * result["fn"]
    )

    training_costs.append((cost, tau))


# Choose minimum cost; ties go to smaller threshold
cost_train_at_tau, cost_tau = min(
    training_costs,
    key=lambda item: (item[0], item[1])
)


# Evaluate selected threshold on test data
test_result_at_tau = evaluate_threshold(
    y_test,
    test_probabilities,
    cost_tau
)

cost_test_at_tau = (
    50 * test_result_at_tau["fp"] +
    400 * test_result_at_tau["fn"]
)


# Evaluate test data at threshold 0.5
test_result_at_05 = evaluate_threshold(
    y_test,
    test_probabilities,
    0.5
)

cost_test_at_05 = (
    50 * test_result_at_05["fp"] +
    400 * test_result_at_05["fn"]
)


# Cost of flagging nobody
cost_test_flag_nobody = 400 * sum(y_test)


# Save Task 2.8 results

with open("outputs/part2_results.txt", "a") as file:

    file.write(f"cost_tau = {cost_tau:.4f}\n")
    file.write(f"cost_train_at_tau = {cost_train_at_tau}\n")
    file.write(f"cost_test_at_tau = {cost_test_at_tau}\n")
    file.write(f"cost_test_at_0.5 = {cost_test_at_05}\n")
    file.write(
        f"cost_test_flag_nobody = {cost_test_flag_nobody}\n"
    )


# Display results
print("\nTask 2.8 completed")

print("Selected threshold:", round(cost_tau, 4))
print("Training cost:", cost_train_at_tau)
print("Test cost at selected threshold:", cost_test_at_tau)
print("Test cost at threshold 0.5:", cost_test_at_05)
print("Test cost when flagging nobody:", cost_test_flag_nobody)

print("\nPart 2 completed")
