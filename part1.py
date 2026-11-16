import numpy as np
import csv
import math
import os

# Load the dataset
with open("data/records.csv", "r", newline="") as file:
    reader = csv.DictReader(file)
    records = list(reader)

# Extract resolution times
values = [float(row["resolution_minutes"]) for row in records]

# Task 1.1: Mean
n = len(values)
mean_value = sum(values) / n

# Sample standard deviation
squared_differences = [(x - mean_value) ** 2 for x in values]
variance = sum(squared_differences) / (n - 1)
sd_value = math.sqrt(variance)

# Standard error
se_value = sd_value / math.sqrt(n)

# 95% confidence interval
ci95_low = mean_value - 1.96 * se_value
ci95_high = mean_value + 1.96 * se_value

# Create outputs folder if needed
os.makedirs("outputs", exist_ok=True)

# Save results
with open("outputs/part1_results.txt", "w") as file:
    file.write(f"mean_value = {mean_value:.4f}\n")
    file.write(f"sd_value = {sd_value:.4f}\n")
    file.write(f"se_value = {se_value:.4f}\n")
    file.write(f"ci95_low = {ci95_low:.4f}\n")
    file.write(f"ci95_high = {ci95_high:.4f}\n")

print("Task 1.1 completed")
print("Number of records:", n)
print("Mean resolution time:", round(mean_value, 4))
print("Sample standard deviation:", round(sd_value, 4))
print("Standard error:", round(se_value, 4))
print("95% confidence interval:", round(ci95_low, 4), "to", round(ci95_high, 4))


# Task 1.2: Test the planning claim

z_stat = (mean_value - 8.0) / se_value

reject_no = "yes" if abs(z_stat) > 1.96 else "no"


# Task 1.3: Required future sample size

margin_error = 0.5

n_required = math.ceil((1.96 * sd_value / margin_error) ** 2)


# Append results to the output file

with open("outputs/part1_results.txt", "a") as file:
    file.write(f"z_stat = {z_stat:.4f}\n")
    file.write(f"reject_no = {reject_no}\n")
    file.write(f"n_required = {n_required}\n")


print("\nTask 1.2 completed")
print("Z statistic:", round(z_stat, 4))
print("Reject H0:", reject_no)

print("\nTask 1.3 completed")
print("Required sample size:", n_required)


# Task 1.4: Simple Linear Regression

# Extract X4 and resolution values
x4 = [float(row["account_value_inr_thousands"]) for row in records]
y = values

mean_x4 = sum(x4) / n
mean_y = sum(y) / n

# Sample variance and covariance
var_x4 = sum((x - mean_x4) ** 2 for x in x4) / (n - 1)

cov_x4_y = sum(
    (x4[i] - mean_x4) * (y[i] - mean_y)
    for i in range(n)
) / (n - 1)

# Regression slope and intercept
reg_slope = cov_x4_y / var_x4
reg_intercept = mean_y - reg_slope * mean_x4

# Fitted values
fitted = [
    reg_intercept + reg_slope * x
    for x in x4
]

# Calculate regression metrics
tss = sum((yi - mean_y) ** 2 for yi in y)
rss = sum((y[i] - fitted[i]) ** 2 for i in range(n))
ess = sum((fi - mean_y) ** 2 for fi in fitted)

r2 = 1 - rss / tss
rmse = math.sqrt(rss / n)
mae = sum(abs(y[i] - fitted[i]) for i in range(n)) / n


# Task 1.5: Gradient Descent

# Standardise X4 using sample standard deviation
sd_x4 = math.sqrt(var_x4)

z = [(x - mean_x4) / sd_x4 for x in x4]

# Initial parameters
m = 0.0
c = 0.0

learning_rate = 0.1

# Initial cost
gd_0 = sum(
    yi ** 2 for yi in y
) / n

gd_1 = None
gd_10 = None
gd_100 = None

# Run 500 gradient descent updates
for step in range(1, 501):

    errors = [
        (m * z[i] + c) - y[i]
        for i in range(n)
    ]

    dm = (2 / n) * sum(
        errors[i] * z[i]
        for i in range(n)
    )

    dc = (2 / n) * sum(errors)

    m = m - learning_rate * dm
    c = c - learning_rate * dc

    # Record cost after each specified update
    if step in [1, 10, 100, 500]:
        cost = sum(
            (m * z[i] + c - y[i]) ** 2
            for i in range(n)
        ) / n

        if step == 1:
            gd_1 = cost
        elif step == 10:
            gd_10 = cost
        elif step == 100:
            gd_100 = cost
        elif step == 500:
            gd_500 = cost

# Convert parameters to original X4 scale
gd_slope = m / sd_x4
gd_intercept = c - m * mean_x4 / sd_x4

gd_slope_gap = abs(gd_slope - reg_slope)


# Save Task 1.4 and 1.5 results

with open("outputs/part1_results.txt", "a") as file:

    file.write(f"reg_slope = {reg_slope:.4f}\n")
    file.write(f"reg_intercept = {reg_intercept:.4f}\n")
    file.write(f"tss = {tss:.4f}\n")
    file.write(f"rss = {rss:.4f}\n")
    file.write(f"ess = {ess:.4f}\n")
    file.write(f"r2 = {r2:.4f}\n")
    file.write(f"rmse = {rmse:.4f}\n")
    file.write(f"mae = {mae:.4f}\n")

    file.write(f"gd_0 = {gd_0:.4f}\n")
    file.write(f"gd_1 = {gd_1:.4f}\n")
    file.write(f"gd_10 = {gd_10:.4f}\n")
    file.write(f"gd_100 = {gd_100:.4f}\n")
    file.write(f"gd_500 = {gd_500:.4f}\n")

    file.write(f"gd_slope = {gd_slope:.4f}\n")
    file.write(f"gd_intercept = {gd_intercept:.4f}\n")
    file.write(f"gd_slope_gap = {gd_slope_gap:.4f}\n")


print("\nTask 1.4 completed")
print("Regression slope:", round(reg_slope, 4))
print("Regression intercept:", round(reg_intercept, 4))
print("R-squared:", round(r2, 4))
print("RMSE:", round(rmse, 4))
print("MAE:", round(mae, 4))

print("\nTask 1.5 completed")
print("Gradient descent slope:", round(gd_slope, 4))
print("Gradient descent intercept:", round(gd_intercept, 4))
print("Gradient descent slope gap:", round(gd_slope_gap, 4))
print("Gradient descent cost after 500 updates:", round(gd_500, 4))


# Task 1.6: Residual funnel and log remedy

# Calculate sample standard deviation
def sample_sd(data):
    avg = sum(data) / len(data)
    return math.sqrt(
        sum((v - avg) ** 2 for v in data) / (len(data) - 1)
    )


# Sort rows by fitted values, breaking ties by row index
sorted_indices = sorted(
    range(n),
    key=lambda i: (fitted[i], i)
)

# Raw residuals
raw_residuals = [
    y[i] - fitted[i] for i in range(n)
]

# Divide into lower and upper halves
lower_raw = [
    raw_residuals[i] for i in sorted_indices[:400]
]

upper_raw = [
    raw_residuals[i] for i in sorted_indices[400:]
]

resid_sd_lower_raw = sample_sd(lower_raw)
resid_sd_upper_raw = sample_sd(upper_raw)

resid_sd_ratio_raw = (
    resid_sd_upper_raw / resid_sd_lower_raw
)


# Log-log regression: ln(value) against ln(X4)

log_x4 = [math.log(x) for x in x4]
log_y = [math.log(v) for v in y]

mean_log_x = sum(log_x4) / n
mean_log_y = sum(log_y) / n

log_var_x = sum(
    (v - mean_log_x) ** 2 for v in log_x4
)

log_cov_xy = sum(
    (log_x4[i] - mean_log_x) *
    (log_y[i] - mean_log_y)
    for i in range(n)
)

loglog_slope = log_cov_xy / log_var_x

loglog_intercept = (
    mean_log_y - loglog_slope * mean_log_x
)

log_fitted = [
    loglog_intercept + loglog_slope * v
    for v in log_x4
]

log_residuals = [
    log_y[i] - log_fitted[i]
    for i in range(n)
]

log_tss = sum(
    (v - mean_log_y) ** 2 for v in log_y
)

log_rss = sum(
    r ** 2 for r in log_residuals
)

loglog_r2 = 1 - log_rss / log_tss

loglog_rmse = math.sqrt(log_rss / n)


# Sort using fitted log values
log_sorted_indices = sorted(
    range(n),
    key=lambda i: (log_fitted[i], i)
)

lower_log = [
    log_residuals[i]
    for i in log_sorted_indices[:400]
]

upper_log = [
    log_residuals[i]
    for i in log_sorted_indices[400:]
]

resid_sd_lower_log = sample_sd(lower_log)
resid_sd_upper_log = sample_sd(upper_log)

resid_sd_ratio_log = (
    resid_sd_upper_log / resid_sd_lower_log
)

# Task 1.7: Multiple regression normal equation

# Predictor order:
# Constant, ln(X4), X1, X2, X3, X5, X6, X7

X1 = [float(row["customer_tenure_months"]) for row in records]
X2 = [float(row["earlier_ticket_count_90d"]) for row in records]
X3 = [float(row["sla_consumed_ratio"]) for row in records]
X5 = [float(row["satisfaction_score"]) for row in records]
X6 = [float(row["reopened_flag"]) for row in records]
X7 = [float(row["self_service_first_flag"]) for row in records]

# Construct design matrix
design_matrix = np.column_stack([
    np.ones(n),
    np.array(log_x4),
    np.array(X1),
    np.array(X2),
    np.array(X3),
    np.array(X5),
    np.array(X6),
    np.array(X7)
])

target = np.array(log_y)

# Normal equation using linear solve
mr_beta = np.linalg.solve(
    design_matrix.T @ design_matrix,
    design_matrix.T @ target
)

# Predicted log values
mr_fitted = design_matrix @ mr_beta

# Calculate multiple regression R-squared
mr_residuals = target - mr_fitted

mr_rss = float(np.sum(mr_residuals ** 2))

mr_tss = float(
    np.sum((target - np.mean(target)) ** 2)
)

mr_r2 = 1 - mr_rss / mr_tss

# Adjusted R-squared (7 predictors)
p = 7

mr_adj_r2 = (
    1 - (1 - mr_r2) * (n - 1) / (n - p - 1)
)

# Adjusted R-squared for Task 1.6 log-log model (1 predictor)
loglog_adj_r2 = (
    1 - (1 - loglog_r2) * (n - 1) / (n - 2)
)


# Save Task 1.6 and Task 1.7 results

with open("outputs/part1_results.txt", "a") as file:

    file.write(f"resid_sd_lower_raw = {resid_sd_lower_raw:.4f}\n")
    file.write(f"resid_sd_upper_raw = {resid_sd_upper_raw:.4f}\n")
    file.write(f"resid_sd_ratio_raw = {resid_sd_ratio_raw:.4f}\n")

    file.write(f"loglog_intercept = {loglog_intercept:.4f}\n")
    file.write(f"loglog_slope = {loglog_slope:.4f}\n")
    file.write(f"loglog_r2 = {loglog_r2:.4f}\n")
    file.write(f"loglog_rmse = {loglog_rmse:.4f}\n")

    file.write(f"resid_sd_lower_log = {resid_sd_lower_log:.4f}\n")
    file.write(f"resid_sd_upper_log = {resid_sd_upper_log:.4f}\n")
    file.write(f"resid_sd_ratio_log = {resid_sd_ratio_log:.4f}\n")

    for i, beta in enumerate(mr_beta):
        file.write(f"mr_beta_{i} = {beta:.4f}\n")

    file.write(f"mr_r2 = {mr_r2:.4f}\n")
    file.write(f"mr_adj_r2 = {mr_adj_r2:.4f}\n")
    file.write(f"loglog_adj_r2 = {loglog_adj_r2:.4f}\n")


print("\nTask 1.6 completed")
print("Raw residual SD ratio:", round(resid_sd_ratio_raw, 4))
print("Log-log intercept:", round(loglog_intercept, 4))
print("Log-log slope:", round(loglog_slope, 4))
print("Log-log R-squared:", round(loglog_r2, 4))
print("Log residual SD ratio:", round(resid_sd_ratio_log, 4))

print("\nTask 1.7 completed")

for i, beta in enumerate(mr_beta):
    print(f"mr_beta_{i}:", round(float(beta), 4))

print("Multiple regression R-squared:", round(mr_r2, 4))
print("Multiple regression adjusted R-squared:", round(mr_adj_r2, 4))
print("Log-log adjusted R-squared:", round(loglog_adj_r2, 4))
