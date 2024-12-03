import openai
import streamlit as st
from dotenv import load_dotenv
import os
import pandas as pd
import uuid
import requests  # FastAPI와 통신
import logging
import subprocess
import atexit
import time
import deepl
from streamlit.runtime.scriptrunner import RerunException # 페이지 새로고침
from datetime import datetime

CHATLOG_SERVER_DIR = "./user_chatlog_server"
CHATLOG_CLIENT_DIR = "./user_chatlog_client"

load_dotenv() #환경변수 값 로드 (API 포함)

# 페이지 구성
st.set_page_config(
    page_title='학습 퀴즈 AI',
    page_icon='🤖',
    layout='centered',
    initial_sidebar_state='auto'
)

# CSV 파일 관련 로드/초기값 생성
CSV_FILE = "chat_history.csv"
try:
    chat_history_df = pd.read_csv(CSV_FILE)
except FileNotFoundError:
    chat_history_df = pd.DataFrame(columns=["ChatID", "Role", "Content"])

if os.path.exists(CSV_FILE):
    chat_history_df = pd.read_csv(CSV_FILE)
else:
    chat_history_df = pd.DataFrame(columns=["ChatID", "Role", "Content"])

########### FastAPI 서버 URL 선언 / 로그파일 생성 ##################
API_BASE_URL = "http://127.0.0.1:8002"  # FastAPI 서버 로컬 호스트 값
# API_BASE_URL = "http://0.0.0.0:8000"  # FastAPI 서버 외부 연결 시

logging.basicConfig(
    filename="Client_UI.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logging.info("Streamlit UI started.")

################# FastAPI 서버 실행/종료 관련 모듈 개선 #######################
# API 서버 실행
def start_api_server():
    process = subprocess.Popen(["uvicorn", "v1_API_server:app", "--reload", "--port", "8002"])
    return process

# API 서버 종료
def stop_api_server(process):
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
    print("API 서버가 종료되었습니다.")

# 세션 종료 시 API 서버 종료하도록 설정
def on_session_end():
    if 'api_server_process' in st.session_state:
        stop_api_server(st.session_state.api_server_process)

# 종료 시점에 호출될 함수 등록
atexit.register(on_session_end)

# Streamlit UI 실행
if 'api_server_process' not in st.session_state:
    st.session_state.api_server_process = start_api_server()
    print("API 서버가 시작되었습니다.")

def wait_for_api():
    for _ in range(10):
        try:
            response = requests.get(f"{API_BASE_URL}/server_check")  # health_check 엔드포인트를 통해 서버 상태 확인
            if response.status_code == 200:
                break
        except requests.exceptions.RequestException:
            time.sleep(1)  # 서버가 준비될 때까지 1초 간격으로 반복
    
wait_for_api()

########### session_state 전역변수 초기값 및 각종 변수 선언 ####################
if 'page' not in st.session_state:
    st.session_state.page = 'login'  # 초기 페이지 설정
if "selected_theme" not in st.session_state:
    st.session_state.selected_theme = '파이썬_라이브러리'
if "order_str" not in st.session_state:
    st.session_state.order_str = 'Pandas 설치 및 Jupyter Notebook 설정하기'
if 'user_id' not in st.session_state:
    st.session_state.user_id = 'None'
if 'session_no' not in st.session_state:
    st.session_state.session_no = 0
if 'type_' not in st.session_state:
    st.session_state.type_ = 'python'
if 'order' not in st.session_state:
    st.session_state.order = 1
if 'language' not in st.session_state:
    st.session_state.language = "KO"
if 'chat_log' not in st.session_state:
    st.session_state.chat_log = ""
if 'chat_session_to_str' not in st.session_state:
    st.session_state.chat_session_to_str = ""
if 'quiz_status_check' not in st.session_state:
    st.session_state.quiz_status_check = 0

### 수진님 코드 (CSV 저장 관련) #######
if "chat_session" not in st.session_state:
    st.session_state["chat_session"] = []
    st.session_state["current_chat_id"] = st.session_state.user_id # str(uuid.uuid4())[:8] 임시 아이디
if 'theme_selected' not in st.session_state:
    st.session_state['theme_selected'] = False

# chat_session 예시 형태 (딕셔너리로 구성된 리스트)
# st.session_state.chat_session = [
#     {"role": "assistant", "content": "Hello, how can I help you?"},
#     {"role": "user", "content": "What is the weather today?"}
# ]

# 초기화 함수 (세션 상태에 chat_history_df 추가)
def initialize_chat_history():
    if 'chat_history_df' not in st.session_state:
        st.session_state.chat_history_df = pd.DataFrame(columns=["ChatID", "Role", "Content"])

# 교재별 매핑 데이터
mapping_data = {
    "파이썬_라이브러리": {
        'Pandas 설치 및 Jupyter Notebook 설정하기': 1,
        'NumPy 소개 및 설치': 2,
        'NumPy 배열(array) 생성 및 기초 연산': 3,
        '배열 연산 및 브로드캐스팅': 4,
        '판다스 사용을 위해 데이터를 불러오기와 저장하기': 5,
        '불러온 데이터 미리보기 및 기본 정보 확인': 6,
        '데이터를 선택하는 기본 방법': 7,
        '조건부 필터링과 데이터 타입 변환': 8,
        '데이터 변형해보기: 데이터 정렬과 병합': 9,
        '데이터 변형해보기: 그룹화 및 집계, 피벗테이블': 10,
        '데이터 전처리: 결측치 탐지와 다양한 처리 방법': 11,
        '데이터 전처리: 이상치 탐지 및 처리': 12,
        '데이터 전처리: 데이터 정규화와 표준화 (비선형 변환 포함)': 13,
        '데이터 전처리: 인코딩 (Encoding)': 14,
        '판다스 심화: 멀티 인덱스와 복합 인덱스': 15
    },
    "머신러닝": {
        '강의 소개': 1,
        '머신러닝 개요와 구성요소': 2,
        'Anaconda 설치 및 라이브러리 소개': 3,
        'Jupyter Notebook 사용해보기': 4,
        '데이터셋 불러오기': 5,
        '데이터 전처리': 6,
        '데이터 전처리 실습': 7,
        '지도학습 : 회귀모델': 8,
        '지도학습 : 분류모델 - 로지스틱 회귀': 9,
        '지도학습 : 분류모델 - SVM': 10,
        '지도학습 : 분류모델 - KNN': 11,
        '지도학습 : 분류모델 - 나이브베이즈': 12,
        '지도학습 : 분류모델 - 의사결정나무': 13,
        '비지도학습 : 군집화모델 - k-means clustering': 14,
        '비지도학습 : 군집화모델 - 계층적 군집화': 15,
        '비지도학습 : 군집화모델 - DBSCAN': 16,
        '비지도학습 : 차원축소 - PCA': 17,
        '비지도학습 : 차원축소 - t-SNE': 18,
        '비지도학습 : 차원축소 - LDA': 19,
        '앙상블 학습 - 배깅과 부스팅': 20,
        '앙상블 학습 - 랜덤 포레스트': 21,
        '앙상블 학습 - 그래디언트 부스팅 머신 (GBM)': 22,
        '앙상블 학습 - XGBoost': 23
    },
    "딥러닝": {
        '딥러닝 개념을 잡아봅시다!': 1,
        '신경망의 기본 원리': 2,
        '딥러닝 실습 환경 구축': 3,
        '인공 신경망(ANN)': 4,
        '합성곱 신경망(CNN)': 5,
        '순환 신경망(RNN)': 6,
        '어텐션 (Attention) 메커니즘': 7,
        '자연어 처리(NLP) 모델': 8,
        'ResNet': 9,
        '이미지 처리 모델': 10,
        '오토인코더': 11,
        '생성형 모델': 12,
        '전이학습': 13,
        '과적합 방지 기법': 14,
        '하이퍼파라미터 튜닝': 15,
        '모델 평가와 검증 및 Pytorch 문법 정리': 16
    },
    "LLM_RAG": {
        'LLM이란? 강의소개!': 1,
        'LLM 시스템 형성을 위한 다양한 기법 및 요소 개념 익히기': 2,
        'OpenAI Playground 사용법 가이드': 3,
        '프롬프트 엔지니어링 개념잡기!': 4,
        '프롬프트 엔지니어링 맛보기': 5,
        '프롬프트 엔지니어링의 기본 원칙': 6,
        'Shot 계열의 프롬프팅 기법 배워보기': 7,
        'Act As 류의 프롬프팅 기법 배우기': 8,
        '논리적인 추론 강화하기': 9,
        '대화를 활용한 프롬프팅 기법': 10,
        '형식 지정 기법': 11,
        'LLM의 사용 준비하기': 12,
        'Vector DB 개념 및 RAG (Retrieval-Augmented Generation) 개념': 13,
        '텍스트 처리의 핵심 기법과 임베딩 활용하기': 14,
        'LangChain: 개념과 활용': 15,
        'Python LangChain과 FAISS': 16,
        'Sentence-Transformer, Word2Vec, 그리고 Transformer 기반 임베딩': 17,
        '문서 임베딩 실습하기': 18
    },
    "OPENSOURCE": {
        '서울시 공공 자전거 분석': 1,
        '무더위 쉼터 데이터': 2,
        'ETF 예측 모델 (다중선형회귀, XGBoost, ARIMA)': 3,
        'ResNet을 이용한 개 고양이 분류기': 4,
        'GAN을 이용한 MNIST 숫자 생성 모델': 5,
        '다양한 유형의 소스(PDF, YouTube 동영상) 로부터 데이터를 가공해 RAG 파이프 라인을 구현하는 예제의 컬럼': 6,
    }
}

# selectbox로 주제 선택
theme_to_type = {
    '파이썬_라이브러리': 'python',
    '머신러닝': 'ml',
    '딥러닝': 'dl',
    'LLM_RAG': 'llm',
    'OPENSOURCE': 'open_source'
}

# 초기 화면 (고정)
st.markdown(
    """
    <style>
    .custom-title {
        color: #008080;
        font-size: 30px;
        font-weight: bold;
    }
    </style>
    """,
    unsafe_allow_html=True,
)
st.markdown('<p class="custom-title">복습퀴즈 챗봇 ✨팔딱이✨</p>', unsafe_allow_html=True)


# 채팅기록 txt 파일 ---> chat_session 형식으로 변환
def parse_txt_to_chat(content):
    chat_session = []
    lines = content.splitlines()  # 텍스트를 줄 단위로 분리
    current_role = None
    current_content = []

    for line in lines:
        line = line.strip()  # 공백 제거

        # 역할 구분
        if line.startswith("👤"):
            # 이전 역할 저장
            if current_role:
                chat_session.append({"role": current_role, "content": "\n".join(current_content).strip()})
            # 새 역할 시작
            current_role = "👤"
            current_content = []
        elif line.startswith("🤖"):
            # 이전 역할 저장
            if current_role:
                chat_session.append({"role": current_role, "content": "\n".join(current_content).strip()})
            # 새 역할 시작
            current_role = "🤖"
            current_content = []
        else:
            # 현재 역할의 content에 줄 추가
            current_content.append(line)

    # 마지막 역할 저장
    if current_role:
        chat_session.append({"role": current_role, "content": "\n".join(current_content).strip()})

    return chat_session

# chat_session ---> 채팅기록 txt 파일 형식으로 변환
def parse_chat_session_to_txt(chat_session):
    
    result = []
    for chat in chat_session:
        # role과 content를 각각 추가
        result.append(chat["role"])  # 🤖 또는 👤
        result.append(chat["content"])  # 대화 내용
        result.append("")  # 줄 간격 추가
    
    # 모든 항목을 줄바꿈으로 연결하고 마지막 공백 제거
    return "\n".join(result).strip()

# 최근 대화목록 생성/갱신
def update_recent_chats():
    # 파일 목록 가져오기
    files = [
        f for f in os.listdir(CHATLOG_CLIENT_DIR)
        if f.startswith(st.session_state.user_id) and f.endswith(".txt")]
    
    # 파일 정렬: datetime 순으로 정렬
    files.sort(key=lambda x: datetime.strptime(x.split("_")[1] + "_" + x.split("_")[2], "%Y%m%d_%H%M%S"), reverse=True)
    # 가장 최근 파일 5개 선택
    recent_files = files[:5]
    # 각 파일에 대해 버튼 생성
    for i, file in enumerate(recent_files, start=1):
        file_path = os.path.join(CHATLOG_CLIENT_DIR, file)
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        # 파일 내용 출력 버튼 추가
        if st.sidebar.button(f"{st.session_state.user_id}님의 최근 대화 {i}"):
            st.session_state.chat_session = parse_txt_to_chat(content)
            st.session_state.quiz_status_check = 2
            reload_chattingBox()
            st.rerun()
            # time.sleep(1)
            # if len(st.session_state.chat_session) >= 10:
            #     st.text_area("채팅 내역", value=content, height=300)
            

# CSV 파일에 마지막 대화 갱신 (실시간 저장)
def append_newchat_to_CSV():
    chat_id = st.session_state.user_id
    new_rows = []
    content = st.session_state.chat_session[-1]
    new_rows.append({
        "ChatID": chat_id,
        "Role": content["role"],
        "Content": content["content"]
    })
    # 마지막 대화내용을 DataFrame으로 변환하여 저장
    new_data_df = pd.DataFrame(new_rows)
    st.session_state.chat_history_df = pd.concat([st.session_state.chat_history_df, new_data_df], ignore_index=True) # 기존 chat_history_df와 new_data_df를 합침
    st.session_state.chat_history_df.to_csv(CSV_FILE, index=False) # CSV 파일에 저장
    
    # 채팅 로그내역 구성
    loaded_chat = st.session_state.chat_history_df[st.session_state.chat_history_df["ChatID"] == chat_id]
    loaded_chat_string = "\n\n".join(f"{row['Role']}\n{row['Content']}" for _, row in loaded_chat.iterrows())
    st.session_state.chat_log = loaded_chat_string

# AI 언어 번역
def get_deepl_discription(content:str, language:str):
    load_dotenv()
    auth_key = os.getenv("DEEPL_API_KEY")
    translator = deepl.Translator(auth_key)
    result = translator.translate_text(content, target_lang=language)
    return result.text

# chat_session 기준으로 대화창 갱신
def reload_chattingBox():
    for msg in st.session_state.chat_session:
        with st.chat_message("ai" if msg["role"] == "🤖" else "user"):
            st.markdown(msg["content"])

################ 콜백 함수 선언 (API 서버에 요청) ######################
# 서버에 저장된 user_id의 최근 대화를 클라이언트 폴더에 저장
def get_recent_chats_fromServer():
    try:
        response = requests.get(f"{API_BASE_URL}/get_history/{st.session_state.user_id}") # 서버에 이전 대화내역 요청
        
        # 응답 상태 확인
        if response.status_code == 200:
            files = response.json()
            # CHATLOG_CLIENT_DIR 경로 폴더가 없으면 새로 생성
            os.makedirs(CHATLOG_CLIENT_DIR, exist_ok=True)
            # 서버로부터 받은 파일을 로컬 디렉토리에 저장
            for file in files:
                file_path = os.path.join(CHATLOG_CLIENT_DIR, file["file_name"])
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(file["content"])
        else:
            st.error(f"서버 요청 실패: {response.status_code} - {response.text}")
    except requests.exceptions.RequestException as e:
        st.error(f"서버와 통신하는 중 오류가 발생했습니다: {e}")

# 아이디 갱신 요청
def update_api_user_id():
    response = requests.post(f"{API_BASE_URL}/set_user_id", json={"requested_user_id": st.session_state.user_id})
    if response.status_code == 200:
        print(f"user_id 값 '{st.session_state.user_id}'으로 서버전송 성공!")
    else:
        print("user_id 값 서버전송 실패: " + response.text)
# 언어 갱신 요청
def update_language():
    selected_language = st.session_state.language
    response = requests.post(f"{API_BASE_URL}/set_language", json={"lang": selected_language})
    if response.status_code == 200:
        print(f"'{selected_language}'로 언어 변경 성공!")
    else:
        print("language값 서버전송 실패: " + response.text)
######################################################################

# 기존 채팅기록 표시
for msg in st.session_state.chat_session:
    with st.chat_message("ai" if msg["role"] == "🤖" else "user"):
        st.markdown(msg["content"])

# 전체 채팅 화면
def chat_page():
    # 초기화 함수 호출하여 chat_history_df 세션 상태에 추가
    initialize_chat_history()

    # 사이드바 구성하기
    st.sidebar.header('주제 선택')

    # 대주제 갱신 요청
    def update_api_type():
        st.session_state.type_ = theme_to_type.get(st.session_state.selected_theme)
        response = requests.post(f"{API_BASE_URL}/set_big_topic", json={"big_topic": st.session_state.type_})
        if response.status_code == 200:
            print(f"type_ 값 '{st.session_state.type_}'으로 서버전송 성공!")
        else:
            print("type_ 값 서버전송 실패: " + response.text)

    theme = st.sidebar.selectbox('주제를 선택하세요.', options=list(theme_to_type.keys()), key="selected_theme", on_change=update_api_type)

    # 소제목 갱신 요청
    def update_api_order():
        st.session_state.order = mapping_data[theme].get(st.session_state.order_str)
        response = requests.post(f"{API_BASE_URL}/set_small_topic", json={"small_topic_order": st.session_state.order})
        if response.status_code == 200:
            print(f"order 값 '{st.session_state.order}'으로 서버전송 성공!")
        else:
            print("order 값 서버전송 실패: " + response.text)

    if theme == '파이썬_라이브러리':
        textbook = st.sidebar.selectbox('어떤 교재를 선택할까요?', options=list(mapping_data[theme].keys()), key="order_str", on_change=update_api_order())
    elif theme == '머신러닝':
        textbook = st.sidebar.selectbox('어떤 교재를 선택할까요?', options=list(mapping_data[theme].keys()), key="order_str", on_change=update_api_order())
    elif theme == '딥러닝':
        textbook = st.sidebar.selectbox('어떤 교재를 선택할까요?', options=list(mapping_data[theme].keys()), key="order_str", on_change=update_api_order())
    elif theme == 'LLM_RAG':
        textbook = st.sidebar.selectbox('어떤 교재를 선택할까요?', options=list(mapping_data[theme].keys()), key="order_str", on_change=update_api_order())
    elif theme == 'OPENSOURCE':
        textbook = st.sidebar.selectbox('어떤 교재를 선택할까요?', options=list(mapping_data[theme].keys()), key="order_str", on_change=update_api_order())

    # 언어 선택
    language_list = ["KO", "EN-US", "JA"]
    st.sidebar.segmented_control("대화언어 선택", language_list, selection_mode="single", default="KO", key="language", on_change=update_language)
    if st.session_state.language == "KO" :
        st.sidebar.markdown(f"**한국어**가 선택되었습니다.")
    elif st.session_state.language == "EN-US" :
        st.sidebar.markdown(f"**영어**가 선택되었습니다.")
    elif st.session_state.language == "JA" :
        st.sidebar.markdown(f"**일본어**가 선택되었습니다.")
    # 녹음 기능
    audio_value = st.sidebar.audio_input("음성으로 대화해보세요.")
    
    if audio_value:
        st.sidebar.audio(audio_value)
    
    
    
    # 퀴즈 생성 함수
    def generate_quiz():
        # with st.chat_message("ai"):
        #     st.markdown("(팔딱이가 퀴즈를 작성중입니다...)")
        st.session_state.quiz_status_check = 1
        try:
            # 로그 코드
            # st.write(f'현재 selected_theme : {st.session_state.selected_theme}')
            # st.write(f'현재 user_id : {st.session_state.user_id}')
            # st.write(f'현재 session_no : {st.session_state.session_no}')
            # st.write(f'현재 type_ : {st.session_state.type_}')
            # st.write(f'현재 order : {st.session_state.order}')
            # st.write(f'현재 order_str : {st.session_state.order_str}')
            # st.write(f'현재 language : {st.session_state.language}')
            # st.write(f'현재 chat_session : {st.session_state.chat_session}')
            # st.write(f'현재 chat_history_df : {st.session_state.chat_history_df}')
            # st.write(f'현재 chat_log : {st.session_state.chat_log}')
            
            response = requests.post(f"{API_BASE_URL}/generate_quiz", json={"topic": st.session_state.type_})
            response.raise_for_status()  # HTTP 에러 발생 시 예외를 발생시킴
            quiz_data = response.json()  # JSON 데이터 추출
            st.session_state.quiz_data = quiz_data
            with st.chat_message("ai"):
                if st.session_state.language == "KO":
                    st.write(f'{theme}에 대한 퀴즈를 내보겠습니다!')
                elif st.session_state.language == "EN-US":
                    st.write(f"Let's take a quiz on {theme}!")
                elif st.session_state.language == "JA":
                    st.write(f'{theme}のクイズを書きましょう！')
                st.markdown(quiz_data["QUIZ"])

            # 퀴즈 내용을 채팅 기록에 추가
            st.session_state.chat_session.append({"role": "🤖", "content": quiz_data["QUIZ"]})
            append_newchat_to_CSV()

        except requests.exceptions.RequestException as e:
            logging.error(f"Error making API request: {e}")
            st.error(f"API 호출 실패: {e}")

    if prompt := st.chat_input("메시지를 입력하세요."):
        # 유저의 답변
        with st.chat_message("user"):
            st.markdown(prompt)
            # 사용자의 입력을 채팅 기록에 추가
            st.session_state.chat_session.append({"role": "👤" , "content": prompt})
            append_newchat_to_CSV()

        # 언어 변경에 따라 안내메세지 세분화
        if st.session_state.quiz_status_check == 0:
            with st.chat_message("ai"):
                if st.session_state.language == "KO":
                    st.markdown("QUIZ 시작 버튼을 눌러 퀴즈를 시작해주세요.")
                    st.session_state.chat_session.append({"role": "🤖", "content": "QUIZ 시작 버튼을 눌러 퀴즈를 시작해주세요."})
                    append_newchat_to_CSV()
                elif st.session_state.language == "EN-US":
                    st.markdown("Please click the QUIZ Start button to start the quiz.")
                    st.session_state.chat_session.append({"role": "🤖", "content": "Please click the QUIZ Start button to start the quiz."})
                    append_newchat_to_CSV()
                elif st.session_state.language == "JA":
                    st.markdown("QUIZスタートボタンを押してクイズを開始してください。")
                    st.session_state.chat_session.append({"role": "🤖", "content": "QUIZスタートボタンを押してクイズを開始してください。"})
                    append_newchat_to_CSV()

        elif st.session_state.quiz_status_check == 1:
            with st.chat_message("ai"):
                if st.session_state.quiz_status_check == 1 and st.session_state.language == "KO":
                    st.markdown("(팔딱이가 피드백을 작성중입니다...)")
                elif st.session_state.quiz_status_check == 1 and st.session_state.language == "EN-US":
                    st.markdown("(팔딱이 is writing feedback...)")
                elif st.session_state.quiz_status_check == 1 and st.session_state.language == "JA":
                    st.markdown("(팔딱이はフィードバックを書いています...)")

        # 실제 피드백/답변
        if st.session_state.quiz_status_check != 0:
            with st.chat_message("ai"):
                if st.session_state.quiz_status_check == 1 :
                    quiz_content = st.session_state.quiz_data.get("QUIZ", "내용 없음") # 딕셔너리 형태의 quiz_data 에서 실제 QUIZ 값만 추출 (str 형식)
                    response = requests.post(f"{API_BASE_URL}/check_answer", json={"quiz": quiz_content, "user_answer" : prompt})
                    response.raise_for_status()  # HTTP 에러 발생 시 예외를 발생시킴
                    feedback_data = response.json()
                    st.markdown(feedback_data["FeedBack"])
                    feedback_content = feedback_data.get("FeedBack","내용 없음")
                    # 응답을 채팅 기록에 추가
                    st.session_state.chat_session.append({"role": "🤖", "content": feedback_content})
                    append_newchat_to_CSV()
                    st.session_state.quiz_status_check += 1

                elif st.session_state.quiz_status_check > 1 :
                    try:
                        # GPT에게 메시지 전달
                        # 마지막 두 개의 딕셔너리 요소 추출
                        last_two_messages = st.session_state.chat_session[-2:]  # 마지막 2개 가져오기
                        # 문자열로 변환
                        formatted_messages_to_str = "\n".join(
                            [f"Role: {msg['role']}, Content: {msg['content']}" for msg in last_two_messages]
                        )
                        gpt_response = openai.chat.completions.create(
                            model="gpt-4o-mini",
                            messages=[
                                {"role": "system", "content": f"다음 대화내용을 참고해서 사용자의 추가적인 질문에 답변해주세요. {formatted_messages_to_str}"},
                                {"role": "user", "content": prompt},
                            ]
                        )
                        gpt_answer_str = gpt_response.choices[0].message.content  # GPT의 응답 내용 중 content 내용만 추출

                        # 대화언어 선택에 따라 팔딱이 언어 변경
                        if st.session_state.language == "KO":
                            st.markdown(gpt_answer_str)  # 응답 출력
                            st.session_state.chat_session.append({"role": "🤖", "content": gpt_answer_str})
                            append_newchat_to_CSV()
                        elif st.session_state.language == "EN-US":
                            trans_answer = get_deepl_discription(gpt_answer_str, "EN-US")
                            st.markdown(trans_answer)
                            st.session_state.chat_session.append({"role": "🤖", "content": trans_answer})
                            append_newchat_to_CSV()
                        elif st.session_state.language == "JA":
                            trans_answer = get_deepl_discription(gpt_answer_str, "JA")
                            st.markdown(trans_answer)
                            st.session_state.chat_session.append({"role": "🤖", "content": trans_answer})
                            append_newchat_to_CSV()

                    except openai.OpenAIError as e:
                        st.error(f"GPT 응답 생성 중 오류가 발생했습니다: {e}")
            
    
    if st.button("채팅기록 보기"):
        st.text_area("채팅 내역", value=parse_chat_session_to_txt(st.session_state.chat_session), height=300)
    if st.button('QUIZ 시작'):
        generate_quiz()
    # 대화 내역을 선택할 수 있는 버튼 추가
    # def get_button_label(chat_df, chat_id):
    #     # 가장 마지막 사용자 메시지를 가져옵니다.
    #     user_messages = chat_df[(chat_df["ChatID"] == chat_id) & (chat_df["Role"] == "user")]
    #     if not user_messages.empty:  # 'User' 메시지가 존재하는 경우
    #         last_user_message = user_messages.iloc[-1]["Content"]
    #         return f"{chat_id[0:7]} : {' '.join(last_user_message.split()[:10])}"  # 마지막 메시지의 첫 10단어를 표시
    #     else:
    #         return f"{chat_id[0:7]} : No User message found"  # 메시지가 없으면 안내 문구 표시

    # 새 대화 시작 버튼 (이전 대화저장 기능 포함)
    if st.sidebar.button('새 대화 시작'):
        # 서버에 현재 대화기록 저장
        try:
            response = requests.post(
                f"{API_BASE_URL}/save_conversation",
                json={"requested_user_id": st.session_state.user_id, "chatlog": st.session_state.chat_log}
            )
            response.raise_for_status()
            st.success("현재 대화가 서버에 저장되었습니다.")
        except requests.exceptions.RequestException as e:
            st.error(f"서버 요청 실패: {e}")

        # 대화세션 관련 정보 초기화
        st.session_state.chat_history_df = pd.DataFrame(columns=["ChatID", "Role", "Content"])
        st.session_state.chat_log =""
        st.session_state.chat_session = []
        st.session_state.quiz_status_check = 0

        get_recent_chats_fromServer() #서버에 저장된 대화기록 불러오기 + 클라 저장
        st.rerun() # 채팅창 동기화를 위해 화면 갱신

    # 현재 대화내용 버튼
    st.sidebar.header('현재 채팅내용 보기')
    if len(st.session_state.chat_history_df) > 0:
        for chat_id in st.session_state.chat_history_df["ChatID"].unique():
            # button_label = get_button_label(st.session_state.chat_history_df, chat_id)
            if st.sidebar.button(f"{st.session_state.user_id}님의 현재 대화"):
                # st.text_area("채팅 내역", value=st.session_state.chat_log, height=300)
                st.session_state.quiz_status_check = 2     
                st.session_state.chat_session = parse_txt_to_chat(st.session_state.chat_log)
                reload_chattingBox()
                st.rerun()
    else:
        st.sidebar.write("진행중인 대화가 없습니다.")
    
    # 사이드바에 저장된 대화 기록을 표시
    st.sidebar.header('최근 대화내역')
    if os.path.exists(CHATLOG_CLIENT_DIR):
    # 디렉토리 내 파일이나 폴더가 있는지 확인
        if os.listdir(CHATLOG_CLIENT_DIR):
            update_recent_chats()

# ID 입력 화면
def login_page():
    
    st.markdown("""
        <div style="text-align: center;">
            <h1>🤖팔딱팔딱 AI QUIZ🤖</h1>
            <img src="https://viralsolutions.net/wp-content/uploads/2019/06/shutterstock_749036344.jpg" width="1280" />
        </div>
        <div style="margin-top: 30px;">  <!-- ID 입력창과 이미지 사이에 30px의 여백 추가 -->
        </div>
    """, unsafe_allow_html=True)

    user_id = st.text_input("ID를 입력하세요.", key="custom_input", placeholder="ID 입력", label_visibility="visible", help="ID를 입력하세요.")
    
    st.markdown("""
        <style>
            .centered-button {
                display: flex;
                justify-content: center;
                width: 30%;
                height: 60px;  /* 버튼 높이 키우기 */
                font-size: 20px;  /* 버튼 글씨 크기 키우기 */
                background-color: #4CAF50;  /* 버튼 배경색 설정 */
                color: white;  /* 버튼 글자 색 */
                border: none;
                border-radius: 8px;
            }
        </style>
    """, unsafe_allow_html=True)

    if st.button('로그인', key='chat_button', use_container_width=True):
        if user_id:
            st.session_state.user_id = user_id

            update_api_user_id() # 유저 아이디 서버 전송
            get_recent_chats_fromServer() # 서버로부터 대화내역 로드 + 클라 저장

            # st.success(f"안녕하세요! {st.session_state['user_id']}님 반갑습니다! '로그인' 버튼을 한번 더 누르면 채팅이 시작됩니다.")
            st.session_state.page = 'chat'  # 페이지를 'chat'으로 설정
            st.rerun() # 로그인 동기화를 위해 화면 갱신
        else:
            st.error('채팅에 사용할 ID를 먼저 입력해주세요.')

# 앱 실행
if st.session_state.page == 'login':
    login_page()
elif st.session_state.page == 'chat':
    chat_page()
# elif st.session_state.page == 'chatting':
#     user_chatting()