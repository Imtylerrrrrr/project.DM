import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping


df = pd.read_excel('프로젝트.xlsx', sheet_name=0)
df.columns = df.columns.str.strip()

df['시간'] = pd.to_datetime(df['시간'], format='%H:%M:%S')
df.sort_values('시간', inplace=True)
df.reset_index(drop=True, inplace=True)

feature_cols = [col for col in df.columns if col != '시간']
data = df[feature_cols].copy()
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

def create_sequences_multi_step(dataset, seq_length, horizon):
    X, Y = [], []
    for i in range(len(dataset) - seq_length - horizon + 1):
        X.append(dataset[i: i + seq_length])
        Y.append(dataset[i + seq_length: i + seq_length + horizon])
    return np.array(X), np.array(Y)

sequence_length = 6
forecast_horizon = 6
X_all, Y_all = create_sequences_multi_step(data_scaled, sequence_length, forecast_horizon)
num_features = data_scaled.shape[1]

Y_all_2d = Y_all.reshape((Y_all.shape[0], forecast_horizon * num_features))

num_samples = X_all.shape[0]
train_size = int(num_samples * 0.7)
val_size = int(num_samples * 0.15)

X_train = X_all[:train_size]
Y_train = Y_all_2d[:train_size]
X_val = X_all[train_size:train_size + val_size]
Y_val = Y_all_2d[train_size:train_size + val_size]
X_test = X_all[train_size + val_size:]
Y_test = Y_all_2d[train_size + val_size:]

model = Sequential()
model.add(LSTM(64, activation='relu', return_sequences=True, input_shape=(sequence_length, num_features)))
model.add(LSTM(64, activation='relu'))
model.add(Dense(forecast_horizon * num_features))
model.compile(optimizer='adam', loss='mse')

early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
model.fit(X_train, Y_train, epochs=100, batch_size=32, validation_data=(X_val, Y_val),
          callbacks=[early_stop], verbose=0)

_ = model.evaluate(X_test, Y_test, verbose=0)
pred_2d = model.predict(X_test[0:1], verbose=0)
pred_multi_step = pred_2d.reshape((forecast_horizon, num_features))
pred_multi_step_rescaled = scaler.inverse_transform(pred_multi_step)

true_multi_step = Y_test[0].reshape((forecast_horizon, num_features))
true_multi_step_rescaled = scaler.inverse_transform(true_multi_step)

print("예측값 (30분 예측):")
print(pred_multi_step_rescaled[:6])
print("실제값 (30분 예측):")
print(true_multi_step_rescaled[:6])

mse = model.evaluate(X_test, Y_test, verbose=0)
print(f"MSE: {mse:.4f}")

