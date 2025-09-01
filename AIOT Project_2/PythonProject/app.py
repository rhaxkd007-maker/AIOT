from flask import Flask, render_template, jsonify, request
from flask_socketio import SocketIO
from flask_sqlalchemy import SQLAlchemy
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import Huber
import joblib
import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
import matplotlib
import threading
import time

auto_thread_running = False
matplotlib.use('Agg')

# -------------------- 앱 및 DB 초기화 --------------------
app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+pymysql://energy_user:supersecret123!@localhost/energy_system'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

socketio = SocketIO(app, async_mode='threading')
db = SQLAlchemy(app)

# -------------------- DB 모델 정의 --------------------
class SolarSwitchLog(db.Model):
    __tablename__ = 'solar_switch_log'
    id = db.Column(db.Integer, primary_key=True)
    switched_at = db.Column(db.DateTime, nullable=False)
    trigger_reason = db.Column(db.String(255))

class SolarEstimate(db.Model):
    __tablename__ = 'solar_estimate'
    id = db.Column(db.Integer, primary_key=True)
    date = db.Column(db.Date, unique=True, nullable=False)
    radiation_mj = db.Column(db.Float, nullable=False)
    estimated_kwh = db.Column(db.Float, nullable=False)
    sunshine = db.Column(db.Float, nullable=True)

with app.app_context():
    db.create_all()
# -------------------- 기본 라우트 --------------------
@app.route('/')
def index():
    return render_template('index.html')

@socketio.on('connect')
def on_connect():
    print('클라이언트 연결됨')

# -------------------- 태양광 전환 테스트 --------------------
@app.route('/switch')
def trigger_solar_switch():
    now = datetime.now()
    now_str = now.strftime('%H:%M:%S')
    socketio.emit('realtime_data', {'time': now_str, 'source': 'solar'})
    log = SolarSwitchLog(switched_at=now, trigger_reason='테스트 전환')
    db.session.add(log)
    db.session.commit()
    return f'태양광 으로 전환됨 ({now_str})'

# -------------------- 일사량 API --------------------
@app.route('/api/kma_radiation')
def kma_radiation():
    date_str = request.args.get('date')
    if not date_str:
        return jsonify({'error': '날짜를 지정해주세요'}), 400
    try:
        tm = date_str.replace('-', '') + "1200"
        res = requests.get(
            'https://apihub.kma.go.kr/api/typ01/url/kma_sfctm2.php',
            params={
                'tm': tm,
                'stn': '108',
                'authKey': 'LMBJkhTqTmeASZIU6j5nkw'
            },
            headers={'User-Agent': 'Mozilla/5.0'}
        )
        if res.status_code != 200:
            return jsonify({'error': '기상청 일사량 API 요청 실패'}), 500

        soup = BeautifulSoup(res.content, 'html.parser')
        rows = soup.find_all('tr')[2:]
        for row in rows:
            cols = row.find_all('td')
            if len(cols) >= 13:
                stn_number = cols[0].text.strip()
                if stn_number == '108':
                    radiation_mj = float(cols[11].text.strip())
                    sunshine = float(cols[12].text.strip())
                    estimated_kwh = round(radiation_mj * 0.15, 2)
                    return jsonify({
                        'radiation_mj': radiation_mj,
                        'sunshine': sunshine,
                        'estimated_kwh': estimated_kwh
                    })
        return jsonify({'error': '서울 데이터 없음'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# -------------------- 예측 피크 위험도 --------------------
@app.route('/api/predicted_peak')
def predicted_peak():
    try:
        peak_risk_index = 0.87
        is_predicted_peak = peak_risk_index >= 0.8
        return jsonify({
            "peak_risk_index": round(peak_risk_index, 2),
            "is_predicted_peak": is_predicted_peak
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# -------------------- 종합 기상 API --------------------
@app.route('/api/kma_full_weather')
def full_weather_combined():
    date_str = request.args.get('date')
    if not date_str:
        return jsonify({'error': '날짜를 지정해주세요'}), 400
    try:
        start_dt = datetime.strptime(date_str + " 00:00", "%Y-%m-%d %H:%M")
        end_dt = start_dt + timedelta(hours=23)
        tm1 = start_dt.strftime("%Y%m%d%H%M")
        tm2 = end_dt.strftime("%Y%m%d%H%M")

        res_weather = requests.get(
            'https://apihub.kma.go.kr/api/typ01/url/kma_sfctm3.php',
            params={
                'tm1': tm1,
                'tm2': tm2,
                'stn': '108',
                'help': '1',
                'authKey': 'LMBJkhTqTmeASZIU6j5nkw'
            },
            headers={'User-Agent': 'Mozilla/5.0'}
        )
        if res_weather.status_code != 200:
            return jsonify({'error': '기온/습도 API 요청 실패'}), 500

        lines = res_weather.text.strip().split('\n')
        data_lines = [line for line in lines if line and line[0].isdigit()]
        weather_result = []
        for line in data_lines:
            fields = line.split()
            if len(fields) > 13:
                tm = fields[0]
                ta = fields[11]
                hm = fields[13]
                ss = fields[15]
                hour = datetime.strptime(tm, "%Y%m%d%H%M").strftime("%H:00")
                weather_result.append({
                    "time": hour,
                    "temperature": ta,
                    "humidity": hm,
                    "sunshine": ss
                })

        res_sun = requests.get(
            'https://apihub.kma.go.kr/api/typ01/url/kma_sfctm2.php',
            params={
                'tm': date_str.replace('-', '') + '1200',
                'stn': '108',
                'authKey': 'LMBJkhTqTmeASZIU6j5nkw'
            },
            headers={'User-Agent': 'Mozilla/5.0'}
        )

        sunshine = None
        if res_sun.status_code == 200:
            soup = BeautifulSoup(res_sun.content, 'html.parser')
            rows = soup.find_all('tr')[2:]
            for row in rows:
                cols = row.find_all('td')
                if len(cols) >= 13 and cols[0].text.strip() == '108':
                    sunshine = cols[12].text.strip()
                    break

        return jsonify({
            "date": date_str,
            "sunshine_hour": sunshine,
            "weather": weather_result
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# -------------------- 머신러닝 예측 API --------------------
@app.route('/api/ml_predict')
def api_ml_predict():
    date_str = request.args.get('date')
    if not date_str:
        return jsonify({'error': '날짜를 지정해주세요'}), 400

    global auto_thread_running
    if not auto_thread_running:
        t = threading.Thread(target=auto_predict_simulator, args=(date_str,), daemon=True)
        t.start()

    try:
        model_dir = r"C:\\Users\\rud22\\Desktop\\완자Projects\\PycharmProjects\\PythonProject\\model"
        data_path = r"C:\\Users\\rud22\\Desktop\\완자Projects\\PycharmProjects\\PythonProject\\data\\preprocessed_smp.csv"

        df = pd.read_csv(data_path, parse_dates=["일시"])
        df["일시"] = pd.to_datetime(df["일시"])

        # 파생변수 생성 (모델과 일치해야 함)
        df["hour"] = df["일시"].dt.hour
        df["dayofweek"] = df["일시"].dt.dayofweek
        df["month"] = df["일시"].dt.month
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
        df["pct_change_smp"] = df["prev_SMP"].pct_change(periods=-1)
        df["rolling_max_smp_5"] = df["SMP"].rolling(window=5).max()
        df["rolling_std_smp_3"] = df["SMP"].rolling(window=3).std()
        df["smp_zscore"] = (df["SMP"] - df["SMP"].rolling(5).mean()) / (df["SMP"].rolling(5).std() + 1e-5)
        df["log_SMP"] = np.log1p(df["SMP"])
        df["target_log_SMP"] = df["log_SMP"].shift(-2)

        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.dropna(inplace=True)

        df_test = df[df["일시"].dt.date == pd.to_datetime(date_str).date()].copy()
        if df_test.empty:
            return jsonify({'error': f"{date_str}에 해당하는 데이터가 없습니다."})

        scaler = joblib.load(f"{model_dir}/scaler.pkl")
        features = joblib.load(f"{model_dir}/features.pkl")
        model = load_model(f"{model_dir}/model.keras", custom_objects={"Huber": Huber})

        for f in features:
            if f not in df_test.columns:
                return jsonify({'error': f"파생 변수 '{f}'이(가) 누락되었습니다."})

        X = df_test[features]
        X_scaled = scaler.transform(X).reshape((X.shape[0], X.shape[1], 1))
        y_pred = np.expm1(model.predict(X_scaled).flatten())
        y_true = np.expm1(df_test["target_log_SMP"])
        df_test["예측"] = y_pred
        df_test["실제"] = y_true
        df_test["변화"] = df_test["예측"].diff()
        df_test["급등"] = df_test["변화"] > 15

        r2 = round(float(np.corrcoef(y_pred, y_true)[0, 1] ** 2), 4)

        try:
            top2 = df_test["예측"].nlargest(2)
            avg_top2 = float(top2.mean())

            if avg_top2 >= 170:
                led_action = "danger_on"
                trigger_reason = "예측 피크 위험(상)"
            elif avg_top2 >= 150:
                led_action = "warning_on"
                trigger_reason = "예측 피크 위험(중)"
            else:
                led_action = "safe_on"
                trigger_reason = "예측 피크 위험(하)"

            # 전환 로그 DB 저장
            log = SolarSwitchLog(switched_at=datetime.now(), trigger_reason=trigger_reason)
            db.session.add(log)
            db.session.commit()

            pi_url = "http://192.168.1.16:5000/led"
            res = requests.post(pi_url, json={"action": led_action})
            print(f"📡 LED sent: {led_action} | avg_top2 = {avg_top2:.2f} | status = {res.status_code}")
        except Exception as e:
            print(f"❌ LED send failed: {e}")
            led_action = "error"
            avg_top2 = None

        results = []
        for _, row in df_test.iterrows():
            results.append({
                "time": row["일시"].strftime("%H:%M"),
                "predicted": round(float(row["예측"]), 2),
                "actual": round(float(row["실제"]), 2),
                "note": "📈 급등" if row["급등"] else ""
            })

        return jsonify({
            "results": results,
            "r2": r2,
            "avg_top2_smp": round(avg_top2, 2) if avg_top2 else None,
            "led_signal": led_action
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

def auto_predict_simulator(start_date_str):
    global auto_thread_running
    if auto_thread_running:
        print("⏱ 이미 자동 실행 중")
        return

    auto_thread_running = True
    with app.app_context():
        current_date = datetime.strptime(start_date_str, "%Y-%m-%d")
        while True:
            date_str = current_date.strftime("%Y-%m-%d")
            print(f"📡 자동 예측 실행 중: {date_str}")

            try:
                with app.test_request_context(f'/api/ml_predict?date={date_str}'):
                    response = api_ml_predict()
                    print(f"✅ 예측 완료: {date_str}")
            except Exception as e:
                print(f"❌ 예측 실패: {date_str} - {e}")

            current_date += timedelta(days=1)
            time.sleep(5)
# -------------------- 실행부 --------------------
if __name__ == '__main__':
    # 자동 예측 쓰레드 시작
    t = threading.Thread(target=auto_predict_simulator, daemon=True)
    t.start()

    # Flask 서버 실행
    socketio.run(app, debug=True, allow_unsafe_werkzeug=True)
