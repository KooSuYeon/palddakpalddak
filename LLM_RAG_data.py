import re
import os
import time
import nltk
from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import ssl

from dotenv import load_dotenv


load_dotenv()  # .env 파일에서 환경 변수 로드


# NLTK에서 불용어와 토큰화에 필요한 리소스 다운로드

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download('punkt_tab')
nltk.download('punkt')
nltk.download('stopwords')

# 데이터 전처리 함수
def preprocess_text(text):
    # 1. HTML 태그 제거
    text = re.sub(r'<.*?>', '', text)  # HTML 태그 제거

    # 2. 소문자 변환
    text = text.lower()

    # 3. 구두점 및 특수 문자 제거
    text = re.sub(r'[^\w\s]', '', text)

    # 4. 불용어 제거
    stop_words = set(stopwords.words('english'))  # 영어 불용어 목록
    word_tokens = word_tokenize(text)
    filtered_text = [word for word in word_tokens if word not in stop_words]

    # 5. 전처리된 텍스트 반환
    return ' '.join(filtered_text)

# Selenium을 사용하여 여러 URL에서 텍스트 추출
def fetch_page_content(urls):
    # Selenium 웹드라이버 설정
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()))

    # 각 URL의 텍스트를 저장할 리스트
    all_texts = []

    # 각 URL을 순차적으로 처리
    for url in urls:
        driver.get(url)
        time.sleep(5)  # 페이지 로딩 대기 (필요에 따라 조정)

        # 페이지 내용 추출 (body 태그 내의 텍스트)
        page_content = driver.find_element(By.TAG_NAME, "body").text
        
        # 텍스트 전처리 후 저장
        preprocessed_text = preprocess_text(page_content)
        all_texts.append(preprocessed_text)  # ["url1 content", "url2 content"]

    # 브라우저 종료
    driver.quit()

    return all_texts

def web_scrawl_contents():
    # 여러 URL 리스트
    urls = [
        os.getenv('WEEK_1_LINK_1'),
        os.getenv('WEEK_1_LINK_2'),
        os.getenv('WEEK_1_LINK_3'),
        os.getenv('WEEK_2_LINK_1'),
        os.getenv('WEEK_2_LINK_2'),
        os.getenv('WEEK_2_LINK_3'),
        os.getenv('WEEK_3_LINK_1'),
        os.getenv('WEEK_3_LINK_2'),
        os.getenv('WEEK_3_LINK_3'),
        os.getenv('WEEK_4_LINK_1'),
        os.getenv('WEEK_4_LINK_2'),
        os.getenv('WEEK_5_LINK_1'),
        os.getenv('WEEK_5_LINK_2'),
        os.getenv('WEEK_5_LINK_3'),
        os.getenv('WEEK_5_LINK_4'),
        os.getenv('WEEK_5_LINK_5'),
        os.getenv('WEEK_5_LINK_6'),
        os.getenv('WEEK_5_LINK_7')

    ]

    # 텍스트 추출
    texts = fetch_page_content(urls)

    # RecursiveCharacterTextSplitter를 사용하여 텍스트 분할
    recursive_text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=100,  # 덩어리 크기
        chunk_overlap=10,  # 덩어리 겹침 크기
        length_function=len,
        is_separator_regex=False,
    )

    # 모든 텍스트를 Document 객체로 감싸서 분할
    documents = [Document(page_content=text) for text in texts]

    # 텍스트 분할
    splits = []
    for doc in documents:
        splits += recursive_text_splitter.split_documents([doc])  # Document 객체 리스트 전달

    # 분할된 텍스트 확인
    for idx, split in enumerate(splits[:10]):
        print(f"Chunk {idx+1}: {split.page_content[:10]}..")  # 앞부분 50자만 출력


if __name__ == '__main__':
    web_scrawl_contents()