import pandas as pd
import numpy as np

# ✅ 데이터 로딩
X_path = r"C:\Users\rud22\PycharmProjects\PythonProject\data\SMP_시계열_2024까지.csv"
y_path = r"C:\Users\rud22\PycharmProjects\PythonProject\data\전처리_SMP제외_일시살림.csv"

df_X = pd.read_csv(X_path, parse_dates=["일시"])
df_y = pd.read_csv(y_path, parse_dates=["일시"])
df = pd.merge(df_X, df_y, on="일시", how="inner")
df = df.sort_values("일시").reset_index(drop=True)

# ✅ 날짜 기반 파생변수
df["hour"] = df["일시"].dt.hour
df["dayofweek"] = df["일시"].dt.dayofweek  # 월=0, 일=6
df["month"] = df["일시"].dt.month

# ✅ 변화량 파생변수
df["delta_temp"] = df["기온(°C)"].diff().fillna(0)
df["delta_humidity"] = df["습도(%)"].diff().fillna(0)
df["delta_demand"] = df["전력수요(MWh)"].diff().fillna(0)

# ✅ 이전 시점 변수
df["prev_SMP"] = df["SMP"].shift(1).bfill()
df["prev_demand"] = df["전력수요(MWh)"].shift(1).bfill()

# ✅ 로그 변환된 SMP (선택)
df["log_SMP"] = np.log1p(df["SMP"])  # 모델 입력용

# ✅ 피크 라벨 (선택)
df["is_peak"] = (df["SMP"] > 250).astype(int)

# ✅ 결과 확인
print(df[["일시", "기온(°C)", "습도(%)", "전력수요(MWh)", "SMP", "log_SMP",
          "hour", "dayofweek", "month",
          "delta_temp", "delta_humidity", "delta_demand",
          "prev_SMP", "prev_demand", "is_peak"]].head(5))

# ✅ 저장
save_path = r"C:\Users\rud22\PycharmProjects\PythonProject\data\preprocessed_smp.csv"
df.to_csv(save_path, index=False)
print(f"✅ 전처리 완료 → {save_path}")