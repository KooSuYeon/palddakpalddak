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


########### FastAPI 서버 URL 선언 / 로그파일 생성 ###################
API_BASE_URL = "http://127.0.0.1:8003"  # FastAPI 서버 로컬 호스트 값
# API_BASE_URL = "http://0.0.0.0:8000"  # FastAPI 서버 외부 연결 시

logging.basicConfig(
    filename="Client_UI.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logging.info("Streamlit UI started.")

################# FastAPI 서버 실행 #################################
subprocess.Popen(["uvicorn", "v1_API_server:app", "--reload", "--port", "8003"])

def wait_for_api():
    while True:
        try:
            response = requests.get(f"{API_BASE_URL}/server_check")  # health_check 엔드포인트를 통해 서버 상태 확인
            if response.status_code == 200:
                break
        except requests.exceptions.RequestException:
            time.sleep(1)  # 서버가 준비될 때까지 1초 간격으로 반복
    
wait_for_api()
####################### OpenAI API키 호출 ###########################
# .env 파일에서 api 키 가져오기
load_dotenv()
API_KEY = os.getenv('openai_api_key')
if API_KEY:
    openai.api_key = API_KEY
else:
    st.error("API 키가 설정되지 않았습니다. .env 파일을 확인하세요.")
    st.stop()

####################### DB 로드 ###########################
# CSV 파일 로드
CSV_FILE = "chat_history.csv"
# CSV 파일이 존재하면 불러오기, 없으면 새로 생성
try:
    chat_history_df = pd.read_csv(CSV_FILE)
except FileNotFoundError:
    chat_history_df = pd.DataFrame(columns=["ChatID", "Role", "Content"])

########### session_state 전역변수 초기값 설정 #############

if "selected_theme" not in st.session_state:
    st.session_state.selected_theme = '파이썬 라이브러리'
if 'type_' not in st.session_state:
    st.session_state.type_ = 'python'
if 'user_id' not in st.session_state:
    st.session_state.user_id = 'Unknown'

####################### UI 구성 ###########################
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

# 교재별 매핑 데이터
mapping_data = {
    "파이썬 라이브러리": {
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
    "LLM, RAG": {
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

# 사이드바 구성하기
st.sidebar.header('목차 선택')

###############################################################
################### 주제별 매핑값 반환 ########################
###############################################################
theme_to_type = {
    '파이썬 라이브러리': 'python',
    '머신러닝': 'ml',
    '딥러닝': 'dl',
    'LLM/RAG': 'llm',
    'AI 활용': 'open_source'
}

# 콜백 함수 정의
def update_api_on_select():
    
    st.session_state.type_ = theme_to_type.get(st.session_state.selected_theme)
    response = requests.post(f"{API_BASE_URL}/set_type", json={"sidebox_type": st.session_state.type_})
    if response.status_code == 200:
        st.success(f"'{st.session_state.selected_theme}' --> 서버에 '{st.session_state.type_}'값으로 전송")
    else:
        st.error("API 호출 실패: Server code error.")

#################### 사이드바 구성 ############################
st.sidebar.header('목차 선택')
theme = st.sidebar.selectbox(
    '주제를 선택하세요.',
    options=list(theme_to_type.keys()),
    key="selected_theme",  # 상태 저장 키
    on_change=update_api_on_select  # 값 변경 시 콜백 호출
)

if theme == '파이썬 라이브러리':
    textbook = st.sidebar.selectbox('어떤 교재를 선택할까요?',
                                ['Pandas 설치 및 Jupyter Notebook 설정하기',
                                'NumPy 소개 및 설치', 'NumPy 배열(array) 생성 및 기초 연산', '배열 연산 및 브로드캐스팅',
                                '판다스 사용을 위해 데이터를 불러오기와 저장하기', '불러온 데이터 미리보기 및 기본 정보 확인', '데이터를 선택하는 기본 방법', '조건부 필터링과 데이터 타입 변환',
                                '데이터 변형해보기: 데이터 정렬과 병합', '데이터 변형해보기: 그룹화 및 집계, 피벗테이블',
                                '데이터 전처리: 결측치 탐지와 다양한 처리 방법', '데이터 전처리: 이상치 탐지 및 처리', '데이터 전처리: 데이터 정규화와 표준화 (비선형 변환 포함)', '데이터 전처리: 인코딩 (Encoding)',
                                '판다스 심화: 멀티 인덱스와 복합 인덱스'])
    st.write(f'{theme}의 "{textbook}" 교재에 대한 퀴즈를 시작하겠습니다!')

elif theme == '머신러닝':
    textbook = st.sidebar.selectbox('어떤 교재를 선택할까요?',
                                ['강의 소개', '머신러닝 개요와 구성요소', 'Anaconda 설치 및 라이브러리 소개', 'Jupyter Notebook 사용해보기',
                                '데이터셋 불러오기', '데이터 전처리', '데이터 전처리 실습',
                                '지도학습 : 회귀모델', '지도학습 : 분류모델 - 로지스틱 회귀', '지도학습 : 분류모델 - SVM', '지도학습 : 분류모델 - KNN', '지도학습 : 분류모델 - 나이브베이즈', '지도학습 : 분류모델 - 의사결정나무',
                                '비지도학습 : 군집화모델 - k-means clustering', '비지도학습 : 군집화모델 - 계층적 군집화', '비지도학습 : 군집화모델 - DBSCAN', '비지도학습 : 차원축소 - PCA', '비지도학습 : 차원축소 - t-SNE', '비지도학습 : 차원축소 - LDA',
                                '앙상블 학습 - 배깅과 부스팅', '앙상블 학습 - 랜덤 포레스트', '앙상블 학습 - 그래디언트 부스팅 머신 (GBM)', '앙상블 학습 - XGBoost'])
    st.write(f'{theme}의 "{textbook}" 교재에 대한 퀴즈를 시작하겠습니다!')

elif theme == '딥러닝':
    textbook = st.sidebar.selectbox('어떤 교재를 선택할까요?',
                                ['딥러닝 개념을 잡아봅시다!', '신경망의 기본 원리', '딥러닝 실습 환경 구축',
                                '인공 신경망(ANN)', '합성곱 신경망(CNN)', '순환 신경망(RNN)',
                                '어텐션 (Attention) 메커니즘', '자연어 처리(NLP) 모델',
                                'ResNet', '이미지 처리 모델',
                                '오토인코더', '생성형 모델', '전이학습',
                                '과적합 방지 기법', '하이퍼파라미터 튜닝', '모델 평가와 검증 및 Pytorch 문법 정리'])
    st.write(f'{theme}의 "{textbook}" 교재에 대한 퀴즈를 시작하겠습니다!')

elif theme == 'LLM & RAG':
    textbook = st.sidebar.selectbox('어떤 교재를 선택할까요?',
                                ['LLM이란? 강의소개!', 'LLM 시스템 형성을 위한 다양한 기법 및 요소 개념 익히기', 'OpenAI Playground 사용법 가이드',
                                '프롬프트 엔지니어링 개념잡기!', '프롬프트 엔지니어링 맛보기', '프롬프트 엔지니어링의 기본 원칙',
                                'Shot 계열의 프롬프팅 기법 배워보기', 'Act As 류의 프롬프팅 기법 배우기', '논리적인 추론 강화하기',
                                '대화를 활용한 프롬프팅 기법', '형식 지정 기법',
                                'LLM의 사용 준비하기', 'Vector DB 개념 및 RAG (Retrieval-Augmented Generation) 개념', '텍스트 처리의 핵심 기법과 임베딩 활용하기', 'LangChain: 개념과 활용', 'Python LangChain과 FAISS', 'Sentence-Transformer, Word2Vec, 그리고 Transformer 기반 임베딩', '문서 임베딩 실습하기'])
    st.write(f'{theme}의 "{textbook}" 교재에 대한 퀴즈를 시작하겠습니다!')

elif theme == 'OPENSOURCE':
    textbook = st.sidebar.selectbox('어떤 교재를 선택할까요?',
                                    ['RAG 기반 비구조화된 데이터를 기반으로 질문에 답변하는 오픈 소스',
                                    '다양한 유형의 소스(PDF, YouTube 동영상) 로부터 데이터를 가공해 RAG 파이프 라인을 구현하는 예제의 컬럼',
                                    'ResNet을 이용한 개 고양이 분류기',
                                    'GAN을 이용한 MNIST 숫자 생성 모델',
                                    'ETF 예측 모델 (다중선형회귀, XGBoost, ARIMA)',
                                    '서울시 공공 자전거 분석',
                                    '무더위 쉼터 데이터'])
    st.write(f'{theme}의 "{textbook}" 교재에 대한 퀴즈를 시작하겠습니다!')

st.sidebar.header('대화 내역')

###############################################################
##################### 퀴즈 생성 ###############################
###############################################################
st.write(f'{theme}에 대한 퀴즈를 내보겠습니다!')
try:
    st.write(f'현재 type_ 값/형식 : {st.session_state.type_}, {type(st.session_state.type_)}')
    response = requests.post(f"{API_BASE_URL}/generate_quiz", json={"topic": st.session_state.type_})
    response.raise_for_status()  # HTTP 에러 발생 시 예외를 발생시킴
    quiz_data = response.json()  # JSON 형식의 응답을 받음
    st.write(quiz_data)  # 퀴즈 내용을 출력
    # 또는
    # st.json(quiz_data)  # 퀴즈 내용을 JSON 형식으로 출력
except requests.exceptions.RequestException as e:
    logging.error(f"Error making API request: {e}")
    st.error(f"API 호출 실패: {e}")

####################### 대화 시작 ###########################

# 새 대화 세션 시작
def start_chat_session():
    return []

if "chat_session" not in st.session_state:
    st.session_state["chat_session"] = start_chat_session()

# 기존 채팅 기록 표시
for content in st.session_state.chat_session:
    with st.chat_message("ai" if content["role"] == "assistant" else "user"):
        st.markdown(content["content"])

################### 사용자 입력 ###############################
if user_answer := st.chat_input("답변을 입력하세요."):

    with st.chat_message("user"):
        st.markdown(user_answer)
        # 사용자의 입력을 채팅 기록에 추가
        st.session_state.chat_session.append({"role": "user", "content": user_answer})

    # 모델로부터 피드백 받기
    with st.chat_message("ai"):
        feedback = requests.post(f"{API_BASE_URL}/check_answer")
        st.markdown(feedback)
        # 응답을 채팅 기록에 추가
        st.session_state.chat_session.append({"role": "assistant", "content": feedback})

    # 대화 내역을 CSV에 저장
    chat_id = str(uuid.uuid4())[:8]  # 고유한 ChatID 생성
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
    chat_history_df = pd.concat([chat_history_df, new_data_df], ignore_index=True)

    # CSV 파일에 저장
    chat_history_df.to_csv(CSV_FILE, index=False)

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
for chat_id in chat_history_df["ChatID"].unique():
    button_label = get_button_label(chat_history_df, chat_id)
    if st.sidebar.button(button_label):
        current_chat_id = chat_id
        loaded_chat = chat_history_df[chat_history_df["ChatID"] == chat_id]
        loaded_chat_string = "\n".join(f"{row['Role']}: {row['Content']}" for _, row in loaded_chat.iterrows())
        st.text_area("Chat History", value=loaded_chat_string, height=300)