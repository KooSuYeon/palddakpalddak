import re
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

# 여러 URL 리스트
urls = [
    # 1주차
    "https://teamsparta.notion.site/LLM-bc8f1fe5e9cb4033a684af9b6432501e",
    "https://teamsparta.notion.site/LLM-e9245bcecd8c4ffcb208de612425571d",
    "https://teamsparta.notion.site/OpenAI-Playground-1b359405637349b4b658c9cd428b8454",
    # 2주차
    "https://teamsparta.notion.site/3678e2f29c004649880ecf030b4ea6a9",
    "https://teamsparta.notion.site/85ee22b636bf402cb2603ae9c5ac2eab",
    "https://teamsparta.notion.site/ea1d0042e2874b2eb122b3ae3234303e",
    # 3주차
    "https://teamsparta.notion.site/Shot-e30ba599508d40d7afd0ea674c6d917b",
    "https://teamsparta.notion.site/Act-As-4fb832f337034a4ca462ba8fa6d74c34",
    "https://teamsparta.notion.site/2ef1e307bd5c4766ac1c4fe82a5dd742",
    # 4주차
    "https://teamsparta.notion.site/0a629faa1d44461eb5d1f3413e6c701a",
    "https://teamsparta.notion.site/32525b172df14fdfaf20f553e1be785e",
    # 5주차
    "https://teamsparta.notion.site/LLM-caa6bd3ca7af417d8c143384f7f61668",
    "https://teamsparta.notion.site/Vector-DB-RAG-Retrieval-Augmented-Generation-f0d0d151ad0b422ab9939b6012469be8",
    "https://teamsparta.notion.site/5b99d723739841e6b5010758801d5158",
    "https://teamsparta.notion.site/LangChain-fa67277f776b48d59aa7e25e1f79980e",
    "https://teamsparta.notion.site/Python-LangChain-FAISS-c8c90ebb26a34d59aa2b0cc8417c4dd2",
    "https://teamsparta.notion.site/Sentence-Transformer-Word2Vec-Transformer-635fca3ce0784db592994b297959fc01",
    "https://teamsparta.notion.site/a947d836cbae4770affc53b3bb8b3441"
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
