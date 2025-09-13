# Salient Object Detection with Decision Transformer

## Overview
This notebook explores **Salient Object Detection (SOD)** using the **DUTS dataset**, a widely recognized benchmark in computer vision. The DUTS dataset contains **10,553 training images** and **5,019 test images**, with pixel-level annotations for evaluating SOD models. The goal is to detect and segment the most prominent objects in an image.

## Contents
1. **Data Analysis** – Understanding the dataset’s characteristics and quality.
2. **Preprocessing** – Data cleaning, normalization, and augmentation.
3. **Feature Extraction** – Identifying key attributes for SOD.
4. **Model Implementation** – Applying **Decision Trees (DT)** for classification.
5. **Evaluation** – Measuring performance using accuracy, precision, recall, and F1-score.

## Usage
- Run each cell in sequence to preprocess the data, train the model, and evaluate its performance.
- Modify hyperparameters to optimize the decision tree’s accuracy.

## References
- [DUTS Dataset Paper](https://openaccess.thecvf.com/content_cvpr_2017/html/Wang_Learning_to_Detect_CVPR_2017_paper.html)
- [Decision Transformer](https://arxiv.org/abs/2106.01345)
