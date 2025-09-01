from flask import Flask
import requests

app = Flask(__name__)

@app.route('/')
def get_weather_data():
    url = (
        "https://apihub.kma.go.kr/api/typ01/url/kma_sfctm3.php"
        "?tm1=201512110100"
        "&tm2=201512140000"
        "&stn=108"
        "&help=1"
        "&authKey=LMBJkhTqTmeASZIU6j5nkw"
    )

    headers = {
        'User-Agent': 'Mozilla/5.0'
    }

    response = requests.get(url, headers=headers)
    lines = response.text.strip().split('\n')

    # 관측값 라인만 필터링 (숫자로 시작하는 줄만)
    data_lines = [line for line in lines if line and line[0].isdigit()]

    print(">>> 관측 데이터 출력")
    for line in data_lines[:5]:  # 앞에서 5줄만 확인
        # 공백 기준으로 분리
        fields = line.split()
        if len(fields) > 14:
            tm = fields[0]
            pa = fields[7]   # 현지기압
            ta = fields[11]  # 기온
            hm = fields[13]  # 습도
            print(f"[{tm}] 기온: {ta}°C, 습도: {hm}%, 기압: {pa}hPa")

    return "✅ 텍스트 파싱 완료, 디버거에서 확인"

if __name__ == '__main__':
    app.run(debug=True)
