# Evaluation Plan

## 1. Evaluation Objective

The objective of this evaluation plan is to define how the machine learning and neural network models will be assessed in the Chest X-Ray Pneumonia Classification project.

The evaluation process aims to:

- Measure the classification performance of each model.
- Compare different models using consistent evaluation metrics.
- Identify false positive and false negative predictions.
- Evaluate the model's ability to detect pneumonia cases.
- Select the most suitable final model.
- Analyze the model's generalization performance on unseen data.

## 2. Dataset Splits

The cleaned dataset is divided into three subsets:

| Dataset Split | NORMAL | PNEUMONIA | Total |
|---|---:|---:|---:|
| Training | 1,078 | 3,086 | 4,164 |
| Validation | 270 | 772 | 1,042 |
| Testing | 231 | 387 | 618 |
| **Total** | **1,579** | **4,245** | **5,824** |

### Training Set
Used to train the models and learn image features.

### Validation Set
Used to compare models, tune hyperparameters, and select the best configuration.

### Testing Set
Used only for final evaluation after model selection.

The test set must not be used for training or hyperparameter tuning.

## 3. Primary Evaluation Metric

### Pneumonia Recall

The primary metric is **Recall for the PNEUMONIA class**.

Recall measures the proportion of actual pneumonia cases correctly identified by the model.

\[
Recall = \frac{TP}{TP + FN}
\]

Where:

- TP (True Positive): Pneumonia cases correctly classified as PNEUMONIA.
- FN (False Negative): Pneumonia cases incorrectly classified as NORMAL.

Recall is important because false negative predictions may result in pneumonia cases being missed.

## 4. Additional Evaluation Metrics

The following metrics will also be used:

### 4.1 Accuracy

Measures the percentage of correctly classified images.

\[
Accuracy = \frac{TP + TN}{TP + TN + FP + FN}
\]

### 4.2 Precision

Measures how many images predicted as PNEUMONIA actually belong to that class.

\[
Precision = \frac{TP}{TP + FP}
\]

### 4.3 F1-Score

The harmonic mean of Precision and Recall.

\[
F1 = 2 \times \frac{Precision \times Recall}{Precision + Recall}
\]

### 4.4 ROC-AUC

Measures the model's ability to distinguish between NORMAL and PNEUMONIA across different classification thresholds.

A higher ROC-AUC generally indicates better class discrimination.

## 5. Confusion Matrix Analysis

A confusion matrix will be generated for each evaluated model.

It contains four possible outcomes:

| Outcome | Meaning |
|---|---|
| True Positive (TP) | PNEUMONIA correctly predicted |
| True Negative (TN) | NORMAL correctly predicted |
| False Positive (FP) | NORMAL incorrectly predicted as PNEUMONIA |
| False Negative (FN) | PNEUMONIA incorrectly predicted as NORMAL |

Special attention will be given to False Negative cases because the primary objective is to identify pneumonia cases.

## 6. Model Comparison

All models will be evaluated using the same validation dataset and evaluation metrics.

The comparison will include:

- Traditional Machine Learning Baseline.
- Simple CNN.
- Improved CNN Architecture.

The comparison table will contain:

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---|---|---|---|---|
| Traditional Baseline | TBD | TBD | TBD | TBD | TBD |
| Simple CNN | TBD | TBD | TBD | TBD | TBD |
| Improved CNN | TBD | TBD | TBD | TBD | TBD |

Actual values will be added after the experiments are completed.

## 7. Error Analysis

The evaluation process will investigate incorrectly classified images.

The analysis will focus on:

- False Negative cases.
- False Positive cases.
- Images with unclear visual patterns.
- Images with unusual contrast or quality.
- Cases where the model produces uncertain predictions.

Sample misclassified images may be visualized to understand possible reasons for model errors.

## 8. Threshold Selection

The default classification threshold is 0.5.

The predicted probability will be interpreted as follows:

- Probability below 0.5: NORMAL.
- Probability equal to or above 0.5: PNEUMONIA.

Different thresholds may be investigated using validation data to understand the trade-off between recall and precision.

The final threshold must be selected before evaluating the test set.

## 9. Reproducibility and Stability

To evaluate the consistency of the models, experiments will use three random seeds:

- 42
- 123
- 2026

Performance metrics will be recorded for each seed.

Where appropriate, the mean and standard deviation of the evaluation metrics will be calculated to understand performance stability.

## 10. Final Model Selection

The final model will be selected based on:

1. High Pneumonia Recall.
2. Good ROC-AUC performance.
3. Balanced Precision and F1-Score.
4. A reasonable number of False Negative predictions.
5. Consistent performance across random seeds.
6. Good generalization without significant overfitting.

A model that achieves high recall by classifying every image as PNEUMONIA will not be considered successful.

## 11. Final Test Evaluation

After selecting the final architecture, hyperparameters, and classification threshold, the model will be evaluated on the untouched test set.

The final evaluation will include:

- Accuracy.
- Precision.
- Pneumonia Recall.
- F1-Score.
- ROC-AUC.
- Confusion Matrix.
- Error Analysis.

The final test results will be reported as the project's main performance results.

## 12. Expected Deliverables

The evaluation phase is expected to produce:

- Model comparison results.
- Confusion matrices.
- ROC curves.
- Training and validation learning curves.
- Error analysis examples.
- Final test metrics.
- A documented explanation of the selected model.

## 13. Limitations

The project is based on a specific chest X-ray dataset and may not generalize to all hospitals, imaging devices, or patient populations.

The model is developed for academic research and educational purposes only.

It is not intended to replace professional medical judgment or clinical diagnosis.

## 14. Current Status

The evaluation strategy and metrics have been defined.

Actual evaluation results, plots, and error analysis will be added after model training and final testing are completed.
