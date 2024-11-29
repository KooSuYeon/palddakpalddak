import openai
import streamlit as st
from dotenv import load_dotenv
import os
import pandas as pd
import uuid
import requests  # FastAPI와 통신
import logging
import subprocess
import time

load_dotenv()

# .env 파일에서 api 키 가져오기
API_KEY = os.getenv('OPENAI_API_KEY')

# OpenAI API 키 설정
if API_KEY:
    openai.api_key = API_KEY
else:
    st.error("API 키가 설정되지 않았습니다. .env 파일을 확인하세요.")
    st.stop()

# CSV 파일 로드
CSV_FILE = "chat_history.csv"

# CSV 파일이 존재하면 불러오기, 없으면 새로 생성
try:
    chat_history_df = pd.read_csv(CSV_FILE)
except FileNotFoundError:
    chat_history_df = pd.DataFrame(columns=["ChatID", "Role", "Content"])


########### FastAPI 서버 URL 선언 / 로그파일 생성 ##################
API_BASE_URL = "http://127.0.0.1:8006"  # FastAPI 서버 로컬 호스트 값
# API_BASE_URL = "http://0.0.0.0:8000"  # FastAPI 서버 외부 연결 시

logging.basicConfig(
    filename="Client_UI.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logging.info("Streamlit UI started.")

################# FastAPI 서버 실행 및 대기 #######################
subprocess.Popen(["uvicorn", "v1_API_server:app", "--reload", "--port", "8006"])
def wait_for_api():
    for _ in range(10):
        try:
            response = requests.get(f"{API_BASE_URL}/server_check")  # health_check 엔드포인트를 통해 서버 상태 확인
            if response.status_code == 200:
                break
        except requests.exceptions.RequestException:
            time.sleep(1)  # 서버가 준비될 때까지 1초 간격으로 반복
    
wait_for_api()

########### session_state 전역변수 초기값 설정 ####################

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
    st.session_state.language = "한국어"

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
        'RAG 기반 비구조화된 데이터를 기반으로 질문에 답변하는 오픈 소스': 1,
        '다양한 유형의 소스(PDF, YouTube 동영상) 로부터 데이터를 가공해 RAG 파이프 라인을 구현하는 예제의 컬럼': 2,
        'ResNet을 이용한 개 고양이 분류기': 3,
        'GAN을 이용한 MNIST 숫자 생성 모델': 4,
        'ETF 예측 모델 (다중선형회귀, XGBoost, ARIMA)': 5,
        '서울시 공공 자전거 분석': 6,
        '무더위 쉼터 데이터': 7
    }
}

# 페이지 구성
st.set_page_config(
    page_title='복습 퀴즈 챗봇',
    page_icon='🤖',
    layout='wide',
    initial_sidebar_state='auto'
)

# 챗봇 이름 꾸미기
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
st.markdown('<p class="custom-title">복습 퀴즈 챗봇📖</p>', unsafe_allow_html=True)

# 사이드바 구성하기
st.sidebar.header('주제 선택')

# selectbox로 주제 선택
theme_to_type = {
    '파이썬_라이브러리': 'python',
    '머신러닝': 'ml',
    '딥러닝': 'dl',
    'LLM_RAG': 'llm',
    'OPENSOURCE': 'open_source'
}

def update_api_type():
    st.session_state.type_ = theme_to_type.get(st.session_state.selected_theme)
    response = requests.post(f"{API_BASE_URL}/set_big_topic", json={"big_topic": st.session_state.type_})
    if response.status_code == 200:
        st.success(f"type_ 값 '{st.session_state.type_}'으로 서버전송 성공!")
    else:
        st.error("type_ 값 서버전송 실패: Server code error.")

def update_api_order():
    st.write(f"현재 theme : {theme}") # 로그 기록
    st.session_state.order = mapping_data[theme].get(st.session_state.order_str)
    response = requests.post(f"{API_BASE_URL}/set_small_topic", json={"small_topic_order": st.session_state.order})
    if response.status_code == 200:
        st.success(f"order 값 '{st.session_state.order}'으로 서버전송 성공!")
    else:
        st.error("order 값 서버전송 실패: Server code error.")

theme = st.sidebar.selectbox(
    '주제를 선택하세요.',
    options=list(theme_to_type.keys()),
    key="selected_theme",  # 상태 저장 키
    on_change=update_api_type  # 값 변경 시 콜백 호출
)

################################# 소주제 선택 #####################################
if theme == '파이썬_라이브러리':
    textbook = st.sidebar.selectbox('어떤 교재를 선택할까요?', options=list(mapping_data[theme].keys()), key="order_str", on_change=update_api_order)
elif theme == '머신러닝':
    textbook = textbook = st.sidebar.selectbox('어떤 교재를 선택할까요?', options=list(mapping_data[theme].keys()), key="order_str", on_change=update_api_order)
elif theme == '딥러닝':
    textbook = st.sidebar.selectbox('어떤 교재를 선택할까요?', options=list(mapping_data[theme].keys()), key="order_str", on_change=update_api_order)
elif theme == 'LLM_RAG':
    textbook = st.sidebar.selectbox('어떤 교재를 선택할까요?', options=list(mapping_data[theme].keys()), key="order_str", on_change=update_api_order)
elif theme == 'OPENSOURCE':
    textbook = st.sidebar.selectbox('어떤 교재를 선택할까요?', options=list(mapping_data[theme].keys()), key="order_str", on_change=update_api_order)

# 언어 선택
language_list = ["한국어", "영어", "일본어"]
selection = st.sidebar.segmented_control(
    "언어", language_list, selection_mode="single", default="한국어"
)
st.sidebar.markdown(f"**{selection}**가 선택되었습니다.")

# 녹음 기능
audio_value = st.sidebar.audio_input("녹음해주세요.")

if audio_value:
    st.sidebar.audio(audio_value)
    
st.sidebar.header('대화 내역')

# 사이드바에 '대화 저장' 버튼 추가
if st.sidebar.button('대화 저장'):
    # 대화 내용을 TXT 파일로 저장 (탭으로 구분)
    chat_history_df.to_csv("chat_history.txt", sep="\t", index=False)
    st.sidebar.write("대화가 TXT 파일로 저장되었습니다.")

# CSV 파일이 존재하지 않으면 빈 DataFrame 생성
if os.path.exists(CSV_FILE):
    chat_history_df = pd.read_csv(CSV_FILE)
else:
    chat_history_df = pd.DataFrame(columns=["ChatID", "Role", "Content"])

# 새 대화 세션 시작
def start_chat_session():
    return []

if "chat_session" not in st.session_state:
    st.session_state["chat_session"] = start_chat_session()
    st.session_state["current_chat_id"] = str(uuid.uuid4())[:8]  # 새 대화가 시작되면 새로운 ChatID 생성

if 'user_id' not in st.session_state:
    st.session_state['user_id'] = None

if 'theme_selected' not in st.session_state:
    st.session_state['theme_selected'] = False

# 기존 채팅 기록 표시
for content in st.session_state.chat_session:
    with st.chat_message("ai" if content["role"] == "assistant" else "user"):
        st.markdown(content["content"])

# 초기화 함수 (세션 상태에 chat_history_df 추가)
def initialize_chat_history():
    if 'chat_history_df' not in st.session_state:
        st.session_state.chat_history_df = pd.DataFrame(columns=["ChatID", "Role", "Content"])

# ID 입력 화면
def login_page():
    st.title('🤖팔딱팔딱 AI QUIZ🤖')
    st.write("⬅️⬅️왼쪽에 있는 사이드바에서 원하는 주제와 교재를 선택해주세요.")
    st.image("https://viralsolutions.net/wp-content/uploads/2019/06/shutterstock_749036344.jpg")
    user_id = st.text_input("ID를 입력하세요:", key="custom_input", placeholder="ID 입력", label_visibility="visible", help="ID를 입력하세요")
    
    if st.button('저장'):
        if user_id:
            # ID를 입력하면 채팅 페이지로 이동
            st.session_state.user_id = user_id
            st.success(f"안녕하세요! {st.session_state['user_id']}님 반갑습니다! **저장** 버튼을 한 번 더 눌러주시면 채팅 페이지로 넘어갑니다.")
            st.session_state.page = 'chat'  # 페이지를 'chat'으로 설정
        else:
            st.error('ID를 입력해주세요.')

# 사용자 입력 받기
def chat_page():
    initialize_chat_history()  # 초기화 함수 호출하여 chat_history_df 세션 상태에 추가

    st.write(f'{theme}에 대한 퀴즈를 내보겠습니다!')
    try:
        st.write(f'현재 selected_theme : {st.session_state.selected_theme}')
        st.write(f'현재 user_id : {st.session_state.user_id}')
        st.write(f'현재 session_no : {st.session_state.session_no}')
        st.write(f'현재 type_ : {st.session_state.type_}')
        st.write(f'현재 order : {st.session_state.order}')
        st.write(f'현재 order_str : {st.session_state.order_str}')
        st.write(f'현재 language : {st.session_state.language}')
        response = requests.post(f"{API_BASE_URL}/generate_quiz", json={"topic": st.session_state.type_})
        response.raise_for_status()  # HTTP 에러 발생 시 예외를 발생시킴
        quiz_data = response.json()  # JSON 데이터 추출
        st.write(quiz_data)  # 퀴즈 내용을 출력
        
    except requests.exceptions.RequestException as e:
        logging.error(f"Error making API request: {e}")
        st.error(f"API 호출 실패: {e}")

    if prompt := st.chat_input("메시지를 입력하세요."):

        with st.chat_message("user"):
            st.markdown(prompt)
            # 사용자의 입력을 채팅 기록에 추가
            st.session_state.chat_session.append({"role": "user", "content": prompt})

        # GPT 모델로부터 응답 받기
        with st.chat_message("ai"):
            quiz_content = quiz_data.get("QUIZ", "내용 없음") # 딕셔너리 형태의 quiz_data 에서 실제 QUIZ 값만 추출 (str 형식)
            response = requests.post(f"{API_BASE_URL}/check_answer", json={"quiz": quiz_content, "user_answer" : prompt})
            response.raise_for_status()  # HTTP 에러 발생 시 예외를 발생시킴
            feedback_data = response.json()  # JSON 데이터 추출
            st.write(feedback_data)  # 퀴즈 내용을 출력
            feedback_content = feedback_data.get("FeedBack","내용 없음")
            # 응답을 채팅 기록에 추가
            st.session_state.chat_session.append({"role": "assistant", "content": feedback_content})

        # 대화 내역을 CSV에 저장
        chat_id = st.session_state["current_chat_id"]
        new_rows = []

        for content in st.session_state.chat_session:
            new_rows.append({
                "ChatID": chat_id,
                "Role": content["role"],
                "Content": content["content"]
            })

        # 새로운 데이터를 DataFrame으로 변환
        new_data_df = pd.DataFrame(new_rows)

        # 기존 chat_history_df와 new_data_df를 합침
        st.session_state.chat_history_df = pd.concat([st.session_state.chat_history_df, new_data_df], ignore_index=True)

        # CSV 파일에 저장
        st.session_state.chat_history_df.to_csv(CSV_FILE, index=False)

    # 대화 내역을 선택할 수 있는 버튼 추가
    def get_button_label(chat_df, chat_id):
        # 가장 마지막 사용자 메시지를 가져옵니다.
        user_messages = chat_df[(chat_df["ChatID"] == chat_id) & (chat_df["Role"] == "user")]
        if not user_messages.empty:  # 'User' 메시지가 존재하는 경우
            last_user_message = user_messages.iloc[-1]["Content"]
            return f"Chat {chat_id[0:7]}: {' '.join(last_user_message.split()[:5])}..."  # 마지막 메시지의 첫 5단어를 표시
        else:
            return f"Chat {chat_id[0:7]}: No User message found"  # 메시지가 없으면 안내 문구 표시

    # 사이드바에 저장된 대화 기록을 표시
    if len(st.session_state.chat_history_df) > 0:
        # 이미 버튼이 만들어져 있다면 대화 목록 표시
        for chat_id in st.session_state.chat_history_df["ChatID"].unique():
            button_label = get_button_label(st.session_state.chat_history_df, chat_id)
            if st.sidebar.button(button_label):
                current_chat_id = chat_id
                loaded_chat = st.session_state.chat_history_df[st.session_state.chat_history_df["ChatID"] == chat_id]
                loaded_chat_string = "\n".join(f"{row['Role']}: {row['Content']}" for _, row in loaded_chat.iterrows())
                st.text_area("Chat History", value=loaded_chat_string, height=300)
    else:
        st.sidebar.write("저장된 대화가 없습니다.")

# 앱 실행
if 'page' not in st.session_state:
    st.session_state.page = 'login'  # 초기 페이지 설정

if st.session_state.page == 'login':
    login_page()
elif st.session_state.page == 'chat':
    chat_page()