Domain: Support

## How to run

Follow these commands from the project root directory.

### 1. Create and activate a virtual environment

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 2. Install the required libraries

```powershell
pip install -r requirements.txt
```

### 3. Generate the dataset

```powershell
python generate_data.py
```

### 4. Run Part 1

```powershell
python part1.py
```

### 5. Run Part 2

```powershell
python part2.py
```

### 6. Run Part 3

```powershell
python part3.py
```

### 7. Run Part 4

```powershell
python part4.py
```


## Part 1 findings

### Task 1.1 – Descriptive Statistics

- Number of records: 800
- Mean resolution time: 8.9382
- Sample standard deviation: 9.1723
- Standard error: 0.3243
- 95% confidence interval: 8.3026 to 9.5738

### Task 1.2 – Hypothesis Testing

- Z statistic: 2.893
- Decision: Reject the null hypothesis (H0).

### Task 1.3 – Sample Size Calculation

- Required sample size: 1293

### Task 1.4 – Linear Regression

- Regression slope: 0.6701
- Regression intercept: -3.6598
- R-squared: 0.5196
- RMSE: 6.3537
- MAE: 3.7636

### Task 1.5 – Gradient Descent

- Gradient descent slope: 0.6701
- Gradient descent intercept: -3.6598
- Slope gap: 0.0000
- Cost after 500 updates: 40.3698

### Task 1.6 – Log-Log Regression

- Log-log intercept: -2.4370
- Log-log slope: 1.5030
- Log-log R-squared: 0.6565
- Log residual SD ratio: 1.0396

### Task 1.7 – Multiple Regression

- Multiple regression R-squared: 0.7425
- Adjusted R-squared: 0.7403
- Log-log adjusted R-squared: 0.6561

Regression coefficients:

- Beta 0: -1.8613
- Beta 1: 1.1576
- Beta 2: -0.0074
- Beta 3: 0.0940
- Beta 4: 0.1067
- Beta 5: 0.0028
- Beta 6: 0.1518
- Beta 7: 0.0172



## Part 2 findings

### Task 2.1 – Training and Testing Split

- Training rows: 560
- Training positive cases: 119
- Testing rows: 240
- Testing positive cases: 51

### Task 2.2 – Logistic Regression

- Initial loss: 0.6931
- Loss after 1 epoch: 0.6753
- Loss after 10 epochs: 0.5685
- Loss after 100 epochs: 0.3952
- Loss after 1000 epochs: 0.3704
- Test binary cross-entropy: 0.3818

Learned weights:

- Weight X1: -0.5554
- Weight X2: 0.5505
- Weight X3: 0.9013
- Weight X4: -0.1459
- Weight X5: -0.2162
- Weight X6: 0.2348
- Weight X7: -0.0120
- Bias: -2.0696

### Task 2.3 – Learning Rate Comparison

- Learning rate 0.01: Loss = 0.5091
- Learning rate 0.1: Loss = 0.3776
- Learning rate 1.0: Loss = 0.3703

### Task 2.4 – Odds Ratios

- X1: 0.5739
- X2: 1.7341
- X3: 2.4629
- X4: 0.8642
- X5: 0.8056
- X6: 1.2646
- X7: 0.9881

### Task 2.5 – Classification Threshold Comparison

**Threshold: 0.5**

- True positives: 24
- False positives: 20
- True negatives: 169
- False negatives: 27
- Accuracy: 0.8042
- Precision: 0.5455
- Recall: 0.4706
- F1-score: 0.5053
- Specificity: 0.8942

**Threshold: 0.3**

- True positives: 38
- False positives: 44
- True negatives: 145
- False negatives: 13
- Accuracy: 0.7625
- Precision: 0.4634
- Recall: 0.7451
- F1-score: 0.5714
- Specificity: 0.7672

- Blind accuracy: 0.7875

### Task 2.6 – Threshold Sweep

- Threshold sweep completed.
- Results saved successfully.

### Task 2.7 – K-Nearest Neighbours (KNN)

Mean cross-validation F1-scores:

| K | Mean F1-score |
|---|---:|
| 1 | 0.4249 |
| 3 | 0.4249 |
| 5 | 0.4541 |
| 7 | 0.4165 |
| 9 | 0.4212 |
| 11 | 0.4382 |
| 13 | 0.3993 |
| 15 | 0.3965 |
| 17 | 0.3655 |
| 19 | 0.3736 |
| 21 | 0.3853 |

- Best K: 5

KNN test results:

- True positives: 17
- False positives: 24
- True negatives: 165
- False negatives: 34
- Accuracy: 0.7583
- Precision: 0.4146
- Recall: 0.3333
- F1-score: 0.3696

### Task 2.8 – Cost-Sensitive Decision

- Selected threshold: 0.15
- Training cost: INR 12,150
- Test cost at selected threshold: INR 6,050
- Test cost at threshold 0.5: INR 11,800
- Test cost when flagging nobody: INR 20,400

**Conclusion:** Lowering the classification threshold increases recall and reduces the cost of missed positive cases in this analysis.



## Part 3 findings

### Task 3.1 – K-Means Clustering

Within-Cluster Sum of Squares (WCSS):

| K | WCSS | Iterations |
|---|---:|---:|
| 2 | 2262.6753 | 6 |
| 3 | 1602.8471 | 7 |
| 4 | 1494.7732 | 21 |
| 5 | 1323.7157 | 15 |
| 6 | 1278.1691 | 27 |

### Task 3.2 – Elbow Method

- Selected optimal number of clusters (K): 3
- The elbow method indicates three customer segments.

### Task 3.3 – Customer Segmentation

| Segment | Size | Outcome Rate |
|---|---:|---:|
| A | 322 | 0.0186 |
| B | 255 | 0.2353 |
| C | 223 | 0.4664 |

Segment characteristics:

**Segment A**
- Mean X1: 47.5028
- Mean X2: 1.9503
- Mean X3: 0.2250
- Mean X4: 12.5301
- Mean X5: 62.1584
- Mean value: 3.5976

**Segment B**
- Mean X1: 28.6071
- Mean X2: 6.0706
- Mean X3: 0.4534
- Mean X4: 28.7192
- Mean X5: 55.0082
- Mean value: 17.1248

**Segment C**
- Mean X1: 14.3892
- Mean X2: 4.2287
- Mean X3: 0.7118
- Mean X4: 16.5099
- Mean X5: 38.5215
- Mean value: 7.2883

### Task 3.4 – Hierarchical Clustering

Clusters at the selected cut:

**Single linkage**
- E1
- E4
- E5
- E2, E3, E6

**Complete linkage**
- E2, E3, E6
- E1, E4, E5

**Average linkage**
- E2, E3, E6
- E1, E4, E5

### Task 3.5 – Principal Component Analysis (PCA)

- Sum of eigenvalues: 5.0000
- Components required for 80% variance: 3

Explained variance:

| Component | Eigenvalue | Explained Variance | Cumulative Variance |
|---|---:|---:|---:|
| PC1 | 2.6136 | 0.5227 | 0.5227 |
| PC2 | 1.1828 | 0.2366 | 0.7593 |
| PC3 | 0.5114 | 0.1023 | 0.8616 |
| PC4 | 0.4055 | 0.0811 | 0.9427 |
| PC5 | 0.2867 | 0.0573 | 1.0000 |

Principal component loadings:

| Feature | PC1 Loading | PC2 Loading |
|---|---:|---:|
| X1 | 0.5317 | 0.1773 |
| X2 | -0.4100 | 0.5056 |
| X3 | -0.5187 | -0.2648 |
| X4 | -0.2916 | 0.6934 |
| X5 | 0.4418 | 0.4025 |

Mean PC1 scores by segment:

- Segment A: 1.7773
- Segment B: -0.7831
- Segment C: -1.6709

**Conclusion:** Three principal components explain approximately 86.16% of the total variance, reducing the dimensionality while retaining most of the information.



## Part 4 findings

### Task 4.1 – Decision Tree Split Evaluation

- Training rows: 560
- Training positive cases: 119
- Root entropy: 0.7462
- Root Gini impurity: 0.3347

Information gain:

| Split | Information Gain |
|---|---:|
| S1 | 0.0111 |
| S2 | 0.0033 |
| S3 | 0.1046 |
| S4 | 0.1195 |

- Best split by information gain: S4
- Best split by Gini gain: S4

### Task 4.2 – Model Voting and Comparison

| Model | TP | FP | TN | FN | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Logistic | 38 | 44 | 145 | 13 | 0.4634 | 0.7451 | 0.5714 |
| KNN | 17 | 24 | 165 | 34 | 0.4146 | 0.3333 | 0.3696 |
| Stump | 39 | 51 | 138 | 12 | 0.4333 | 0.7647 | 0.5532 |
| Hard Vote | 35 | 45 | 144 | 16 | 0.4375 | 0.6863 | 0.5344 |
| Soft Vote | 38 | 49 | 140 | 13 | 0.4368 | 0.7451 | 0.5507 |

- Best F1-score: Logistic Regression (0.5714)
- All models evaluated on 240 test records.

### Task 4.3 – AdaBoost Calculations

**Round 1**
- Weighted error: 0.2500
- Alpha: 0.5493

**Round 2**
- Weighted error: 0.1667
- Alpha: 0.8047

- Ensemble accuracy: 0.7500
- Correct predictions: 6 out of 8

### Task 4.4 – Out-of-Bag Fraction

- Sample size: 560
- Out-of-bag fraction: 0.3676

### Task 4.5 – Neural Network Architecture

**Model A – Option 1: Fully Connected Network**

| Layer | Output Shape | Parameters |
|---|---|---:|
| Hidden Layer 1 | 16 | 12,560 |
| Hidden Layer 2 | 16 | 272 |
| Output Layer | 10 | 170 |
| Total | | 13,002 |

**Model A – Option 2: Convolutional Neural Network**

| Layer | Output Shape | Parameters |
|---|---|---:|
| Conv1 | 24 × 24 × 5 | 380 |
| MaxPool1 | 12 × 12 × 5 | 0 |
| Conv2 | 8 × 8 × 5 | 630 |
| MaxPool2 | 4 × 4 × 5 | 0 |
| Flatten | 80 | 0 |
| Dense1 | 32 | 2,592 |
| Output | 10 | 330 |
| Total | | 3,932 |

**Model B – Recurrent Neural Network**

| Parameter | Shape | Parameters |
|---|---|---:|
| W_xh | 64 × 128 | 8,192 |
| W_hh | 128 × 128 | 16,384 |
| b_h | 128 | 128 |
| W_hy | 128 × 16 | 2,048 |
| b_y | 16 | 16 |
| Total | | 26,768 |

### Overall Part 4 Conclusion

Logistic regression achieved the highest F1-score among the five evaluated models. The CNN architecture requires fewer parameters than the fully connected network and RNN architectures. These results provide a basis for selecting suitable models for future development.



## Model-choice memo

**To:** Customer-Support Operations Lead

**Subject:** Recommendation for Customer-Support Decision Intelligence

I recommend deploying the logistic regression model as the first structured-data model for production. Among the five models evaluated in Part 4, logistic regression achieved the highest F1-score of 0.5714, with a recall of 0.7451. This makes it a reasonable choice for identifying potentially unresolved customer-support cases.

The logistic regression model performed better than KNN, which achieved an F1-score of 0.3696 and recall of 0.3333. The decision stump achieved a slightly higher recall of 0.7647 but a lower F1-score of 0.5532. Combining models did not improve the F1-score over logistic regression.

The cost analysis in Task 2.8 identified a threshold of 0.15, with a test cost of INR 6,050, compared with INR 11,800 at a threshold of 0.5 and INR 20,400 when no cases are flagged. These results support evaluating a lower threshold to reduce missed cases.

However, the analysis has limitations. The dataset is synthetic, the test set contains only 240 records, and the evaluation uses a single train-test split. Real customer data and further validation are necessary before production deployment.

I recommend building Model A (the image-based CNN) before Model B (the text-based RNN) as the next development step, while keeping logistic regression as the initial production baseline.


## Limitations

- The dataset used in this project is synthetic and may not fully represent real-world customer-support data.
- The test dataset contains only 240 records, which limits the reliability of model evaluation.
- The models were evaluated using a single train-test split, so performance may vary with different data splits.
- The classification results depend on the selected decision threshold.
- The neural network architectures were analysed theoretically and were not trained or evaluated on actual image or text datasets.
- Real customer data and further testing are required before deploying the recommended model in a production environment.