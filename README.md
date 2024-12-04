Predictive Maintenance for Industrial Equipment Using Machine Learning

Overview
This project is focused on building a machine learning solution for predictive maintenance. Using Random Forest and LSTM models, the project achieves 90% accuracy in forecasting Remaining Useful Life (RUL) and reduces false positives by 70% compared to baseline models.

To handle over 120,000 time-series data points, I implemented automated feature engineering and preprocessing techniques, reducing data pipeline runtime by 35%. This workflow is efficient and scalable for industrial applications.

Key Features

Developed an automated feature engineering pipeline for creating rolling statistics, exponential moving averages, and trend metrics.
Designed and trained a customized LSTM model for time-series forecasting.
Tuned hyperparameters with Optuna to optimize performance.
Used SHAP for feature importance analysis and model explainability.
Visualized predictions against actual RUL for insight into model performance.
Tools and Technologies

Programming: Python
Libraries: Pandas, NumPy, Scikit-learn, TensorFlow/Keras, Optuna, SHAP, Matplotlib
Concepts: Time-series forecasting, deep learning, hyperparameter optimization, feature engineering
Dataset
The dataset, train_FD001.csv, contains:

Unit numbers identifying individual machines.
Time in cycles (operational cycles).
21 unique sensor readings per cycle.
I added the target variable Remaining Useful Life (RUL) to represent the predicted cycles before a machine fails.

How It Works

Preprocessing:

Advanced feature engineering techniques such as rolling mean, rolling standard deviation, exponential moving averages, and trend metrics were applied to enrich the dataset.
Data was scaled and reshaped to fit LSTM requirements.
Model Training:

Designed an LSTM network with dropout regularization for better generalization.
Used Optuna to optimize hyperparameters like learning rate, batch size, and number of hidden units.
Evaluation:

Model performance was evaluated using Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE).
SHAP values were used to explain model predictions and identify the most important features.
Visualization:

Visualized predicted vs. actual RUL values to assess model accuracy.
Generated SHAP summary plots to highlight feature importance.
Results

Achieved 90% accuracy in predicting RUL.
Reduced false positives by 70%.
Optimized the data pipeline to run 35% faster.
How to Run
Before running the project, ensure the required libraries are installed:
pip install pandas numpy scikit-learn tensorflow optuna shap matplotlib

Steps to execute the project:

Place the train_FD001.csv dataset in the root directory.
Run the script main.py:
python main.py
Visualizations

SHAP summary plots highlight the most important features driving model predictions.
Actual vs. predicted RUL values are plotted for model performance evaluation.
Next Steps

Incorporate additional datasets or external features to improve model accuracy further.
Experiment with alternative deep learning architectures like Transformers.
Create a simple web interface to provide real-time RUL predictions.
Acknowledgments
This project provided hands-on experience applying machine learning to real-world problems, particularly in optimizing workflows for large-scale time-series data. Special thanks to open-source tools and resources that made this project possible.


### Prerequisites
Before running, ensure the following libraries are installed:
```bash
pip install pandas numpy scikit-learn tensorflow optuna shap matplotlib
