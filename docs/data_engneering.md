# Dataset Information

## 1. Dataset Name
**Chest X-Ray Pneumonia Dataset**

## 2. Dataset Source
The dataset is obtained from Hugging Face:

https://huggingface.co/datasets/hf-vision/chest-xray-pneumonia

It contains chest X-ray images used for binary classification between normal lungs and pneumonia cases.

## 3. Dataset Description
The dataset consists of grayscale chest X-ray images divided into two classes:

| Class | Label | Description |
|---|---:|---|
| NORMAL | 0 | Chest X-ray images without pneumonia |
| PNEUMONIA | 1 | Chest X-ray images showing pneumonia |

The task is to train a neural network to recognize visual patterns that distinguish the two classes.

## 4. Dataset Structure
The cleaned dataset is organized into three main splits:

```text
chest_xray_clean/
├── train/
│   ├── NORMAL/
│   └── PNEUMONIA/
├── val/
│   ├── NORMAL/
│   └── PNEUMONIA/
└── test/
    ├── NORMAL/
    └── PNEUMONIA/
```

## 5. Dataset Distribution After Cleaning

After cleaning and splitting, the dataset contains **5,824 images**.

| Dataset Split | NORMAL | PNEUMONIA | Total |
|---|---:|---:|---:|
| Training | 1,078 | 3,086 | 4,164 |
| Validation | 270 | 772 | 1,042 |
| Testing | 231 | 387 | 618 |
| **Total** | **1,579** | **4,245** | **5,824** |

## 6. Class Distribution Analysis

The dataset is imbalanced because the number of PNEUMONIA images is significantly higher than the number of NORMAL images.

- NORMAL: 1,579 images (approximately 27.1%).
- PNEUMONIA: 4,245 images (approximately 72.9%).

To address class imbalance, class weights are calculated from the training data and can be used during model training.

## 7. Image Properties and Preprocessing

The images are preprocessed using the following steps:

1. Load images from their respective directories.
2. Check image readability and remove corrupted images.
3. Convert images to grayscale.
4. Resize images to 224 × 224 pixels.
5. Normalize pixel values to the range [0, 1].
6. Expand image dimensions to obtain the shape (224, 224, 1).

### Final Image Format

| Property | Value |
|---|---|
| Image type | Grayscale |
| Image size | 224 × 224 |
| Number of channels | 1 |
| Data type | float32 |
| Pixel value range | [0, 1] |
| Input shape | (224, 224, 1) |

## 8. Data Cleaning

The data cleaning process includes:

- Removing unreadable or corrupted images.
- Calculating SHA-256 hashes to identify duplicate images.
- Detecting duplicate images with conflicting class labels.
- Removing duplicate copies while prioritizing the test split, followed by validation and training.
- Extracting patient identifiers from filenames when possible.

These steps help improve dataset quality and reduce the risk of duplicate images affecting model evaluation.

## 9. Data Splitting and Leakage Prevention

The dataset is divided into:

- Training set: used to train the models.
- Validation set: used for model selection and hyperparameter tuning.
- Test set: reserved for final model evaluation.

Patient-based grouping is used when splitting training and validation data to reduce the possibility of images from the same patient appearing in both sets.

Image hash overlap is also checked to detect duplicate images across splits.

The test set is kept separate from training and hyperparameter tuning.

## 10. Data Augmentation

Data augmentation is applied to training images only to increase data diversity and help reduce overfitting.

The current augmentation pipeline includes:

- Random Rotation.
- Random Translation.

Validation and test images are not augmented.

## 11. Dataset Limitations

Some limitations of the dataset include:

- Class imbalance between NORMAL and PNEUMONIA images.
- Possible variations in image quality and contrast.
- Limited dataset size compared with large-scale medical imaging datasets.
- Patient-level leakage must be carefully checked when splitting the data.

These limitations are considered during preprocessing, model training, and evaluation.

## 12. Summary

The cleaned dataset contains 5,824 grayscale chest X-ray images across two classes: NORMAL and PNEUMONIA.

Images are resized to 224 × 224 and normalized before being provided to the neural network.

The dataset is divided into training, validation, and testing sets, with cleaning, duplicate detection, and leakage checks applied to improve the reliability of the experiments.
