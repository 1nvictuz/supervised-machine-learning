# Supervised Machine Learning and Model Evaluation

## Overview
This project is part of my AI Developer Boot Camp. I used Python and Logistic Regression to train a machine learning model and evaluate its predictions.

## What I Did
- Loaded a CSV dataset using Pandas.
- Split the data into 80% training and 20% testing.
- Trained a Logistic Regression model.
- Compared its accuracy with a baseline model.
- Calculated precision, recall, and a confusion matrix.
- Reviewed incorrect predictions.

## Tools Used
- Python
- Pandas
- Scikit-learn
- Visual Studio Code

## How to Run

```bash
pip install pandas scikit-learn
python supervised_ml.py
```

## Results
The program calculates accuracy, baseline accuracy, precision, and recall. It also displays a confusion matrix and saves incorrect predictions to a CSV file.

## Limitations
Model performance depends on the quality and amount of data. False positives and false negatives may have different impacts depending on the problem.

The dataset was provided as part of the course. Redistribution permissions have not been verified.

## What I Learned
I learned how to train a classification model, evaluate its accuracy, and compare its predictions with a simple baseline. I also learned why testing with unseen data is important.
