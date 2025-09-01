import requests
import xml.etree.ElementTree as ET
from urllib.parse import urlencode, quote_plus

# 1. API URL과 인증키 설정
base_url = 'https://openapi.kpx.or.kr/openapi/sumperfuel5m/getSumperfuel5m'
service_key = 'ISymxiVURQrXN+R+evMoXRTz6SrO7ceheURKf1azKMLmeYldHfTWOciRY6PUU9Lpk/i/4eF8c2ssiRLpzRfKpg=='

params = {
    'serviceKey': service_key
}

url = base_url + '?' + urlencode(params, quote_via=quote_plus)

# 2. 요청
response = requests.get(url)
response.encoding = 'utf-8'

# 3. 디버깅 출력
print("Status Code:", response.status_code)
print("응답 내용:", response.text[:300])  # 앞부분만 잘라 보기

# 4. 파싱 시도
try:
    root = ET.fromstring(response.text)
    print("XML 파싱 성공")
except ET.ParseError as e:
    print("XML 파싱 실패:", e)

print("사용한 키:", service_key)
print("인코딩 상태 확인:", urlencode({'serviceKey': service_key}))  # 디버깅용
