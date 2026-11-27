# Chest X-Ray Pneumonia Classification

## Project Overview
This project focuses on developing a deep learning system to classify chest X-ray images into two categories:

- **NORMAL:** No pneumonia detected.
- **PNEUMONIA:** Pneumonia detected.

The project uses supervised learning and neural network architectures to analyze chest X-ray images and evaluate their classification performance.

## Dataset
- **Dataset:** Chest X-Ray Pneumonia
- **Source:** [Hugging Face – hf-vision/chest-xray-pneumonia](https://huggingface.co/datasets/hf-vision/chest-xray-pneumonia)
- **Task:** Binary Image Classification
- **Classes:** NORMAL and PNEUMONIA

The dataset will be loaded and processed through the project data pipeline. Raw dataset files will not be stored directly in this repository.

## Technologies
- Python
- TensorFlow / Keras
- NumPy
- Pandas
- Scikit-learn
- Matplotlib

## Project Structure
```text
Neural-Networks-Project/
├── configs/
├── data/
├── notebooks/
├── src/
├── experiments/
├── models/
├── app/
├── report/
├── README.md
└── requirements.txt
```

## Evaluation
The project will evaluate the models using classification metrics, including:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix
- ROC Curve and AUC

**Primary Metric:** To be finalized by the team.

## Team
This project is developed by a team of six students as part of the AIE231 Neural Networks course at Alamein International University.

## Project Status
Currently working on data exploration, preprocessing, and baseline development.
