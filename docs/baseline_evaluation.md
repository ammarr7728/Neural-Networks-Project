# Baseline Evaluation

## 1. Evaluation Overview

The purpose of baseline evaluation is to establish initial performance benchmarks for the Chest X-Ray Pneumonia Classification project.

The baseline models will provide reference results that can be compared with more advanced neural network architectures.

The evaluation focuses on binary classification:

- NORMAL (0)
- PNEUMONIA (1)

## 2. Baseline Models

Two baseline approaches are planned.

### 2.1 Traditional Machine Learning Baseline

A traditional machine learning classifier will be implemented using extracted image features.

The evaluation will investigate how well a conventional machine learning approach can distinguish between NORMAL and PNEUMONIA images.

The final algorithm and feature extraction method will be documented after implementation.

### 2.2 Simple CNN Baseline

A basic Convolutional Neural Network will be trained directly on chest X-ray images.

The initial architecture consists of:

- Three convolutional layers.
- Max pooling layers.
- Global Average Pooling.
- A dense layer.
- Dropout.
- A sigmoid output layer.

The CNN baseline will serve as a reference for evaluating improved neural network architectures.

## 3. Evaluation Dataset

The cleaned dataset will be used with the following splits:

| Split | NORMAL | PNEUMONIA | Total |
|---|---:|---:|---:|
| Training | 1,078 | 3,086 | 4,164 |
| Validation | 270 | 772 | 1,042 |
| Testing | 231 | 387 | 618 |

Training data will be used to fit the models.

Validation data will be used for baseline comparison and model selection.

The test set will remain untouched until final evaluation.

## 4. Evaluation Metrics

The baseline models will be evaluated using the following metrics:

### Accuracy

Measures the proportion of correctly classified images.

\[
Accuracy = \frac{TP+TN}{TP+TN+FP+FN}
\]

### Precision

Measures how many predicted PNEUMONIA cases are actually PNEUMONIA.

\[
Precision = \frac{TP}{TP+FP}
\]

### Recall

Measures how many actual PNEUMONIA cases are correctly identified.

\[
Recall = \frac{TP}{TP+FN}
\]

### F1-Score

Balances Precision and Recall.

\[
F1 = 2 \times \frac{Precision \times Recall}{Precision+Recall}
\]

### ROC-AUC

Measures the ability of the model to distinguish between the two classes across different classification thresholds.

## 5. Primary Evaluation Metric

The primary metric is **Pneumonia Recall**.

The project prioritizes identifying pneumonia cases and reducing False Negative predictions.

However, Recall must be analyzed alongside Precision, F1-Score, ROC-AUC, and the Confusion Matrix.

A model that predicts every image as PNEUMONIA will not be considered a successful baseline simply because it achieves high Recall.

## 6. Baseline Results

The following table will be completed after running both baseline models.

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---|---|---|---|---|
| Traditional ML | TBD | TBD | TBD | TBD | TBD |
| Simple CNN | TBD | TBD | TBD | TBD | TBD |

Actual validation results will be recorded in `experiments/baseline_results.csv`.

## 7. Confusion Matrix Analysis

A confusion matrix will be generated for each baseline model.

The analysis will focus on:

- True Positives: Correctly identified PNEUMONIA cases.
- True Negatives: Correctly identified NORMAL cases.
- False Positives: NORMAL cases incorrectly classified as PNEUMONIA.
- False Negatives: PNEUMONIA cases incorrectly classified as NORMAL.

False Negative predictions will receive particular attention because they represent missed pneumonia cases.

## 8. Baseline Comparison

The comparison will consider:

1. Pneumonia Recall.
2. Precision and F1-Score.
3. ROC-AUC.
4. Confusion Matrix.
5. Training time.
6. Model complexity.

The results will help determine whether CNN-based feature learning provides an advantage over traditional machine learning approaches.

## 9. Limitations

Baseline performance may be affected by:

- Class imbalance.
- Image quality variations.
- Dataset size.
- Feature extraction choices.
- Model architecture and hyperparameters.

These limitations will be considered when interpreting the results.

## 10. Conclusion

The baseline evaluation will establish initial performance benchmarks for the project.

The results will be used to guide further experimentation, architecture improvements, and hyperparameter tuning.

**Current Status:** Evaluation criteria and comparison structure are defined. Actual results will be added after both baseline models have been implemented and evaluated.
