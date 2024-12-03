from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager  # 드라이버 자동 설치
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
import os
import time
from dotenv import load_dotenv
import re


# .env 파일 로드
load_dotenv()  # .env 파일을 로드하여 환경 변수 읽기

# Chrome 드라이버 설정
options = Options()
options.add_argument("--headless")  # 창을 띄우지 않음 (필요없으면 삭제 가능)
options.add_argument("--disable-gpu")  # GPU 비활성화
options.add_argument("--no-sandbox")  # 샌드박스 비활성화
options.add_argument("--disable-dev-shm-usage")  # /dev/shm 사용 비활성화
service = Service(ChromeDriverManager().install())

driver = webdriver.Chrome(service=service, options=options)

url_list=[]
txt_list=[]

# 환경변수에 저장된 URL 로드
# URL의 수만큼 설정을 달리 해야함 (DL: 16, ML:23, LLM:18, PYTHON:15, OPENSOURCE:6)
for i in range(1, 24):
    url = os.getenv(f"ML_URL_{i}")
    if url:  # 환경변수가 존재하면 추가
        url_list.append(url)

# 전처리 함수 정의
def preprocess_text(txt):
    # 1. '[스파르타코딩클럽]' 제거
    txt = re.sub(r'\[?스파르타코딩클럽\]?', '', txt)

    # 2. 저작권 문구 제거
    txt = re.sub(r'Copyright.*$', '', txt)

    # 3. \xa0를 공백으로 변환
    txt = txt.replace('\xa0', ' ')

    # 4. 정규식을 사용해 \\로 시작하는 LaTeX 명령어 제거
    txt = re.sub(r'\\[a-zA-Z]+\{.*?\}', '', txt)  

    # 5. \command{...} 형식 제거
    txt = re.sub(r'\\[a-zA-Z]+', '', txt)  

    return txt


# 웹페이지 요청
for url in url_list:
    driver.get(url)  # 페이지 로드

        # 페이지 로드 대기
    try:
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, '.notion-page-content'))
        )
        print("페이지가 완전히 로드되었습니다.")
    except Exception as e:
        print(f"페이지 로딩 실패: {url}. Error: {e}")
        continue

    # 토글이 닫혀 있으면 토글을 열기
    try:
        # 모든 토글 버튼을 찾음 (Ctrl+Alt+T에 해당하는 토글을 찾아서 열기)
        toggle_buttons = driver.find_elements(By.XPATH, "//div[@role='button' and (@aria-expanded='false')]")

        # 각 토글을 클릭하여 열기
        for button in toggle_buttons:
            button.click()
            time.sleep(1)  # 토글이 열리기 전에 잠깐 대기

    except Exception as e:
        print(f"토글을 여는 데 실패했습니다: {e}")

    # HTML 파싱 및 텍스트 추출
    html_content = driver.page_source
    soup = BeautifulSoup(html_content, 'html.parser')

    # 텍스트 추출 및 전처리
    raw_txt = soup.get_text()
    cleaned_txt = preprocess_text(raw_txt)

    # 텍스트 저장
    txt_list.append(cleaned_txt)

driver.quit()
print("드라이버 종료")

# 저장할 폴더 경로
output_folder = "cleaned_texts"  # 원하는 폴더 이름 지정

# 폴더가 없으면 생성
os.makedirs(output_folder, exist_ok=True)

# 각 텍스트를 개별 파일로 저장
for idx, txt in enumerate(txt_list, start=1):
    output_file = os.path.join(output_folder, f"cleaned_text_{idx}.txt")  # 폴더 경로 포함
    with open(output_file, 'w', encoding='utf-8') as file:
        file.write(txt)  # 전처리된 텍스트를 파일에 저장

    print(f"텍스트가 '{output_file}'에 저장되었습니다.")