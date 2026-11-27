# Experiment Plan

## 1. Experimentation Objective

The objective of the experimentation phase is to evaluate different machine learning and neural network configurations for Chest X-Ray Pneumonia Classification.

The experiments aim to:

- Compare traditional machine learning baselines with neural network models.
- Evaluate different CNN architectures.
- Investigate the effect of hyperparameters on model performance.
- Reduce overfitting and improve generalization.
- Select the best-performing model based on validation results.

## 2. Dataset and Evaluation Setup

The experiments will use the cleaned Chest X-Ray dataset.

| Dataset Split | Number of Images |
|---|---:|
| Training | 4,164 |
| Validation | 1,042 |
| Testing | 618 |
| **Total** | **5,824** |

The dataset contains two classes:

- NORMAL (0)
- PNEUMONIA (1)

The test set will remain separate and will only be used for final evaluation after model selection.

## 3. Baseline Experiments

Before evaluating more advanced neural network architectures, baseline models will be implemented.

### Experiment 1: Traditional Machine Learning Baseline

A traditional machine learning approach will be trained using extracted image features.

The experiment will evaluate how well a conventional classifier performs on the classification task.

### Experiment 2: Simple CNN Baseline

A simple CNN will be trained using the prepared chest X-ray images.

The initial architecture includes:

- Three convolutional layers.
- Max pooling layers.
- Global Average Pooling.
- A dense layer.
- Dropout.
- A sigmoid output layer.

The results will be used as a reference for evaluating improved architectures.

## 4. Hyperparameter Experiments

The following hyperparameters will be investigated:

| Hyperparameter | Initial Value | Values to Explore |
|---|---|---|
| Learning Rate | 0.001 | 0.0001, 0.0005, 0.001 |
| Batch Size | 32 | 16, 32, 64 |
| Dropout | 0.5 | 0.3, 0.5 |
| Filters | 32 | 16, 32, 64 |
| Optimizer | Adam | Adam |
| Epochs | 30 | Up to 30 |
| Early Stopping Patience | 5 | 5 |

The experiments will change selected parameters while keeping the remaining configuration consistent whenever possible.

## 5. Data Augmentation Experiments

Data augmentation will be investigated to understand its effect on model generalization.

The planned augmentation techniques are:

- Random Rotation.
- Random Translation.

Two configurations will be compared:

1. Training without data augmentation.
2. Training with data augmentation.

Augmentation will only be applied to training images.

## 6. Class Imbalance Experiments

The dataset contains more PNEUMONIA images than NORMAL images.

The experiments will investigate the use of class weights to reduce the effect of class imbalance.

The following configurations may be compared:

- Training without class weights.
- Training with calculated class weights.

All comparisons will use the same data splits to ensure fair evaluation.

## 7. Reproducibility

To improve reproducibility, experiments will use the following random seeds:

- 42
- 123
- 2026

Each experiment will record its seed and configuration.

The final selected model will be evaluated across the planned seeds to examine performance consistency.

## 8. Evaluation Metrics

The following metrics will be recorded for each experiment:

- Accuracy.
- Precision.
- Recall.
- F1-Score.
- ROC-AUC.
- Confusion Matrix.

### Primary Metric

**Pneumonia Recall** is the primary metric.

It measures the proportion of actual pneumonia cases correctly identified by the model.

However, recall will not be considered alone. ROC-AUC, precision, F1-Score, and confusion matrix results will also be reviewed to avoid selecting a model that predicts every image as PNEUMONIA.

## 9. Experiment Tracking

Each experiment will be documented with:

| Field | Description |
|---|---|
| Experiment ID | Unique experiment identifier |
| Model | Model architecture or baseline |
| Learning Rate | Learning rate used |
| Batch Size | Batch size used |
| Dropout | Dropout rate |
| Filters | Number of convolutional filters |
| Augmentation | Whether augmentation was enabled |
| Class Weights | Whether class weights were enabled |
| Random Seed | Seed used |
| Best Epoch | Epoch with the best monitored validation result |
| Validation Recall | Pneumonia recall on validation data |
| Validation AUC | ROC-AUC on validation data |
| Validation F1 | F1-Score on validation data |
| Training Time | Total training duration |

Results will be stored in CSV files under the `experiments/` directory.

## 10. Model Selection Strategy

The best model will be selected using validation results.

The selection process will consider:

1. Pneumonia Recall.
2. ROC-AUC.
3. F1-Score.
4. Confusion Matrix.
5. Consistency across random seeds.
6. Evidence of overfitting.

The final test evaluation will be performed only after the model and its configuration have been selected.

## 11. Expected Outcomes

The experimentation phase is expected to provide:

- A comparison between baseline models and CNN architectures.
- An understanding of how hyperparameters affect model performance.
- An evaluation of data augmentation and class weighting.
- Training and validation learning curves.
- A reproducible experiment log.
- A selected final model supported by experimental results.

## 12. Current Status

The project has an initial CNN architecture and a data preprocessing pipeline.

The planned experiments will be executed, tracked, and documented as implementation progresses.

Actual experiment results will be added after the corresponding runs are completed.
