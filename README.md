# Data-Science-for-Business-Real-world-Case-Studies — HR Department Code

## Overview
This repository contains the HR module of my six-department business practice project.  
The code builds a full machine-learning pipeline to predict employee attrition using data sourced from the links referenced in my notebook.

## Technical Summary

### Data Loading & Preparation
- Load the HR dataset into a Pandas DataFrame.
- Convert categorical yes/no fields (Attrition, OverTime, Over18) into binary values.
- Drop non-informative or constant columns (`EmployeeCount`, `StandardHours`, `Over18`, `EmployeeNumber`).
- Split the dataset into employees who stayed vs. employees who left for comparison.
- Visualize distributions, correlations, and attrition patterns using Seaborn and Matplotlib.

### Feature Engineering
- One-hot encode categorical variables (`BusinessTravel`, `Department`, `EducationField`, `Gender`, `JobRole`, `MaritalStatus`).
- Select numerical features (e.g., Age, MonthlyIncome, JobLevel, TotalWorkingYears, etc.).
- Combine encoded categorical and numerical features into a unified feature matrix.
- Normalize all features with `MinMaxScaler`.

### Model Training
The code trains three different classification models:

1. **Logistic Regression**  
   - Baseline binary classifier using scikit-learn.  
   - Evaluates accuracy, confusion matrix, and classification report.

2. **Random Forest Classifier**  
   - Ensemble tree-based model for non-linear relationships.  
   - Evaluates model performance using the same metrics as above.

3. **Deep Learning Model**  
   - TensorFlow sequential neural network.  
   - Architecture: three hidden layers (500 units, ReLU) and a sigmoid output layer.  
   - Trained for 100 epochs with batch size 50.  
   - Predicts attrition using a probability threshold of 0.5.

### Evaluation
- Confusion matrices visualized with Seaborn heatmaps.
- Classification reports for precision, recall, f1-score, and accuracy.
- Training-loss and training-accuracy curves for the neural network.

## Purpose
This HR module demonstrates a complete, code-driven machine-learning workflow for employee-attrition prediction as part of a broader multi-department business case-study project.

## Note from Me
The data was provided by the HR Department section of a Udemy course called "Data Science for Business | 6 Real-world Case Studies." 

In the process I managed to build the full machine-learning pipeline on my own, relying primarily on my independent understanding rather than the course videos or slides. My primary purpose of using Udemy was as a validation to my final work as well as a way to get useable clean data.
