import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import Huber
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ✅ 경로 설정
model_dir = r"C:\Users\rud22\PycharmProjects\PythonProject\model"
data_path = r"C:\Users\rud22\PycharmProjects\PythonProject\data\preprocessed_smp.csv"

# ✅ 모델 및 전처리기 로드
model = load_model(f"{model_dir}/model.keras", custom_objects={"Huber": Huber})
scaler = joblib.load(f"{model_dir}/scaler.pkl")
features = joblib.load(f"{model_dir}/features.pkl")

# ✅ 데이터 로드 및 파생변수 생성
df = pd.read_csv(data_path, parse_dates=["일시"])
df["일시"] = pd.to_datetime(df["일시"])
df["hour"] = df["일시"].dt.hour
df["dayofweek"] = df["일시"].dt.dayofweek
df["month"] = df["일시"].dt.month

# 파생 변수 (학습 시와 동일하게)
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
df["target_log_SMP"] = df["log_SMP"].shift(-2)

df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(inplace=True)

# ✅ 예측 대상 날짜 설정
test_date = "2024-12-30"
df_test = df[(df["일시"] >= f"{test_date} 01:00:00") & (df["일시"] <= f"{test_date} 23:00:00")].copy()

# ✅ 입력값 구성
X_test = df_test[features]
X_test_scaled = scaler.transform(X_test).reshape((X_test.shape[0], X_test.shape[1], 1))

# ✅ 예측
y_pred_log = model.predict(X_test_scaled).flatten()
y_pred = np.expm1(y_pred_log)
y_true = np.expm1(df_test["target_log_SMP"])

df_test["예측_SMP"] = y_pred
df_test["실제_SMP"] = y_true
df_test["변화량"] = df_test["예측_SMP"].diff()
df_test["급등_여부"] = df_test["변화량"] > 15

# ✅ 피크 시점 탐지
peak_idx = df_test["예측_SMP"].idxmax()

# ✅ 설명력 (R²)
r2 = r2_score(y_true, y_pred)

# ✅ 시간별 상세 출력
print("\n📊 시간별 예측 결과 및 급등 탐지")
print("=" * 90)
for i in range(len(df_test)):
    row = df_test.iloc[i]
    time = row["일시"].strftime('%H:%M')
    pred = row["예측_SMP"]
    actual = row["실제_SMP"]
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
    print(f"{time} | 예측: {pred:7.2f}₩ | 실제: {actual:7.2f}₩ {delta_str:<10} {note}")

print("=" * 90)
print(f"\n📈 설명력 (R² 결정계수): {r2:.4f}")
