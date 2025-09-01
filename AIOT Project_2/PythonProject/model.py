import pandas as pd
import numpy as np
import os
import joblib
import tensorflow as tf

from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, LSTM, Dense, Dropout, Flatten, Input, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.losses import Huber

print("📌 GPU 상태:", tf.config.list_physical_devices('GPU'))

# ✅ 경로 설정
save_dir = r"C:\Users\rud22\PycharmProjects\PythonProject\model"
data_path = r"C:\Users\rud22\PycharmProjects\PythonProject\data\preprocessed_smp.csv"

# ✅ 데이터 불러오기 및 파생
df = pd.read_csv(data_path, parse_dates=["일시"])
df["일시"] = pd.to_datetime(df["일시"])
df["hour"] = df["일시"].dt.hour
df["dayofweek"] = df["일시"].dt.dayofweek
df["month"] = df["일시"].dt.month

# ⏳ shift(-3) 기준 미래 예측
df["prev_SMP"] = df["SMP"].shift(-1)
df["prev_SMP_t2"] = df["SMP"].shift(-2)
df["prev_SMP_t3"] = df["SMP"].shift(-3)
df["prev_demand"] = df["전력수요(MWh)"].shift(-1)
df["prev_demand_t2"] = df["전력수요(MWh)"].shift(-2)
df["prev_demand_t3"] = df["전력수요(MWh)"].shift(-3)

df["delta_prev_smp"] = df["prev_SMP"] - df["prev_SMP_t2"]
df["rolling_smp"] = df["prev_SMP"].rolling(window=3).mean()
df["rate_prev_smp"] = df["delta_prev_smp"] / (df["prev_SMP_t2"] + 1e-5)
df["delta_demand"] = df["전력수요(MWh)"].diff(-1)
df["rate_demand"] = df["delta_demand"] / (df["prev_demand_t2"] + 1e-5)
df["rolling_mean_smp_7"] = df["SMP"].shift(-1).rolling(window=7).mean()
df["diff_prev_smp"] = df["prev_SMP"].diff(-1)
df["pct_change_smp"] = df["prev_SMP"].pct_change(-1)
df["rolling_max_smp_5"] = df["SMP"].rolling(window=5).max()
df["rolling_std_smp_3"] = df["SMP"].rolling(window=3).std()
df["smp_zscore"] = (df["SMP"] - df["SMP"].rolling(5).mean()) / (df["SMP"].rolling(5).std() + 1e-5)

df["target_log_SMP"] = df["log_SMP"].shift(-3)  # 미래 예측

df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(inplace=True)

# ✅ 피처 및 타깃 설정
features = [
    "기온(°C)", "습도(%)", "전력수요(MWh)", "hour", "dayofweek", "month",
    "delta_prev_smp", "rolling_smp", "rate_prev_smp",
    "delta_demand", "rate_demand", "rolling_mean_smp_7",
    "diff_prev_smp", "pct_change_smp",
    "prev_SMP", "prev_SMP_t2", "prev_SMP_t3",
    "prev_demand", "prev_demand_t2", "prev_demand_t3",
    "rolling_max_smp_5", "rolling_std_smp_3", "smp_zscore"
]

X = df[features].copy()
y = df["target_log_SMP"].copy()

# ✅ 스케일링 및 reshape
scaler = RobustScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = X_scaled.reshape((X_scaled.shape[0], X_scaled.shape[1], 1))

# ✅ 모델 정의 (Conv1D + LSTM 하이브리드 구조)
model = Sequential([
    Input(shape=(X_scaled.shape[1], 1)),
    Conv1D(64, kernel_size=3, padding='causal', activation='relu'),
    BatchNormalization(),
    Dropout(0.2),
    LSTM(64, return_sequences=True),
    LSTM(32),
    Dense(64, activation='relu'),
    Dense(1)
])

model.compile(optimizer='adam', loss=Huber(delta=10.0))
early_stop = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)

# ✅ 모델 학습
print("\n🚀 모델 학습 시작...")
history = model.fit(
    X_scaled, y,
    epochs=200,
    batch_size=32,
    validation_split=0.15,
    callbacks=[early_stop],
    verbose=2
)

# ✅ 저장
os.makedirs(save_dir, exist_ok=True)
model.save(os.path.join(save_dir, "model.keras"))
joblib.dump(scaler, os.path.join(save_dir, "scaler.pkl"))
joblib.dump(features, os.path.join(save_dir, "features.pkl"))
print("✅ 모델 및 스케일러 저장 완료!")

# ✅ 테스트셋 평가
test_date = "2024-12-30"
df_test = df[(df["일시"] >= f"{test_date} 01:00:00") & (df["일시"] <= f"{test_date} 23:00:00")].copy()
X_test = df_test[features]
X_test_scaled = scaler.transform(X_test).reshape((X_test.shape[0], X_test.shape[1], 1))
y_test = np.expm1(df_test["target_log_SMP"])
y_pred = np.expm1(model.predict(X_test_scaled).flatten())

# ✅ 성능 평가
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\n📊 최종 테스트셋 성능")
print("=" * 40)
print(f"MAE : {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R²  : {r2:.4f}")
print("=" * 40)

# ✅ 급등 구간 출력 (시간 단위)
df_test["예측_SMP"] = y_pred
df_test["변화량"] = df_test["예측_SMP"].diff()
df_test["급등_여부"] = df_test["변화량"] > 15
peak_idx = df_test["예측_SMP"].idxmax()

print("\n⏱️ 시간별 예측 및 급등 정보")
print("=" * 60)
for i in range(len(df_test)):
    row = df_test.iloc[i]
    time = row["일시"]
    smp = row["예측_SMP"]
    real = np.expm1(row["target_log_SMP"])
    delta = row["변화량"]
    spike = row["급등_여부"]

    note = ""
    if i == df_test.index.get_loc(peak_idx):
        note += "🔺 피크 구간"
    elif spike:
        note += "⚠️ 급등 경고"
        if i == df_test[df_test["급등_여부"]].index[0]:

            note += " (⏳ 급등 시작)"

    delta_str = f"(+{delta:.2f})" if pd.notnull(delta) and delta > 0 else f"({delta:.2f})" if pd.notnull(delta) else ""
    print(f"{time.strftime('%H:%M')} | 예측 SMP: {smp:.2f}₩ | 실제 SMP: {real:.2f}₩ {delta_str:<10} {note}")

print("=" * 60)
