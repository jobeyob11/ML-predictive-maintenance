import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import optuna
import shap
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv('train_FD001.csv', header=None)
columns = ['unit_number', 'time_in_cycles'] + [f'sensor_{i}' for i in range(1, 22)]
data.columns = columns

# Add Remaining Useful Life (RUL)
def add_rul(df):
    max_cycle = df.groupby('unit_number')['time_in_cycles'].max()
    df['RUL'] = df['unit_number'].map(max_cycle) - df['time_in_cycles']
    return df

data = add_rul(data)

# Feature Engineering
def generate_advanced_features(df, window=5):
    for sensor in [col for col in df.columns if 'sensor_' in col]:
        df[f'{sensor}_rolling_mean'] = df.groupby('unit_number')[sensor].transform(lambda x: x.rolling(window).mean())
        df[f'{sensor}_rolling_std'] = df.groupby('unit_number')[sensor].transform(lambda x: x.rolling(window).std())
        df[f'{sensor}_exp_mean'] = df.groupby('unit_number')[sensor].transform(lambda x: x.ewm(span=window).mean())
        df[f'{sensor}_trend'] = df.groupby('unit_number')[sensor].transform(lambda x: x.diff())
    df = df.dropna()
    return df

data = generate_advanced_features(data)

# Define feature set and target
feature_columns = [col for col in data.columns if 'sensor_' in col and ('rolling' in col or 'exp_mean' in col or 'trend' in col)]
X = data[feature_columns]
y = data['RUL']

# Scale data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Reshape data for LSTM
def reshape_for_lstm(X, y, time_steps=30):
    X_reshaped, y_reshaped = [], []
    for i in range(time_steps, len(X)):
        X_reshaped.append(X[i-time_steps:i])
        y_reshaped.append(y[i])
    return np.array(X_reshaped), np.array(y_reshaped)

time_steps = 30
X_lstm, y_lstm = reshape_for_lstm(X_scaled, y)

# Train-test split
train_size = int(len(X_lstm) * 0.8)
X_train, X_test = X_lstm[:train_size], X_lstm[train_size:]
y_train, y_test = y_lstm[:train_size], y_lstm[train_size:]

# Define LSTM model
def create_lstm_model(trial):
    model = Sequential()
    model.add(LSTM(trial.suggest_int("units", 50, 200), activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dropout(trial.suggest_float("dropout", 0.1, 0.5)))
    model.add(Dense(1))
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=trial.suggest_float("lr", 1e-4, 1e-2, log=True)),
                  loss='mse', metrics=['mae'])
    return model

# Hyperparameter optimization with Optuna
def objective(trial):
    model = create_lstm_model(trial)
    early_stop = EarlyStopping(monitor='val_loss', patience=5)
    history = model.fit(X_train, y_train, validation_split=0.2, epochs=50, batch_size=trial.suggest_int("batch_size", 16, 128),
                        callbacks=[early_stop], verbose=0)
    val_loss = min(history.history['val_loss'])
    return val_loss

study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=20)

# Train the best model
best_params = study.best_params
print(f"Best Parameters: {best_params}")

best_model = create_lstm_model(study.best_trial)
early_stop = EarlyStopping(monitor='val_loss', patience=5)
best_model.fit(X_train, y_train, validation_split=0.2, epochs=50, batch_size=best_params["batch_size"],
               callbacks=[early_stop], verbose=1)

# Evaluate the model
y_pred = best_model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"Test Mean Absolute Error: {mae}")
print(f"Test Root Mean Squared Error: {rmse}")

# SHAP for Explainability
explainer = shap.KernelExplainer(best_model.predict, X_test)
shap_values = explainer.shap_values(X_test[:100])

# SHAP Summary Plot
shap.summary_plot(shap_values, X_test[:100], feature_names=feature_columns)

# Visualize Actual vs Predicted
plt.figure(figsize=(12, 8))
plt.plot(range(len(y_test)), y_test, label='Actual RUL')
plt.plot(range(len(y_pred)), y_pred, label='Predicted RUL', alpha=0.7)
plt.xlabel('Instance')
plt.ylabel('RUL')
plt.title('Actual vs Predicted RUL')
plt.legend()
plt.show()
