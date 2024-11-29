import openai
import streamlit as st
from dotenv import load_dotenv
import os
import pandas as pd
import uuid

load_dotenv()

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
option = st.sidebar.selectbox('주제를 선택하세요.', ['파이썬', '머신러닝', '딥러닝', 'LLM & RAG'])
if option == '파이썬 라이브러리':
    textbook = st.sidebar.selectbox('어떤 교재를 선택할까요?',
                                ['Pandas 설치 및 Jupyter Notebook 설정하기',
                                'NumPy 소개 및 설치', 'NumPy 배열(array) 생성 및 기초 연산', '배열 연산 및 브로드캐스팅',
                                '판다스 사용을 위해 데이터를 불러오기와 저장하기', '불러온 데이터 미리보기 및 기본 정보 확인', '데이터를 선택하는 기본 방법', '조건부 필터링과 데이터 타입 변환',
                                '데이터 변형해보기: 데이터 정렬과 병합', '데이터 변형해보기: 그룹화 및 집계, 피벗테이블',
                                '데이터 전처리: 결측치 탐지와 다양한 처리 방법', '데이터 전처리: 이상치 탐지 및 처리', '데이터 전처리: 데이터 정규화와 표준화 (비선형 변환 포함)', '데이터 전처리: 인코딩 (Encoding)',
                                '판다스 심화: 멀티 인덱스와 복합 인덱스'])
    st.write(f'{option}의 "{textbook}" 교재에 대한 퀴즈를 시작하겠습니다!')

elif option == '머신러닝':
    textbook = st.sidebar.selectbox('어떤 교재를 선택할까요?',
                                ['강의 소개', '머신러닝 개요와 구성요소', 'Anaconda 설치 및 라이브러리 소개', 'Jupyter Notebook 사용해보기',
                                '데이터셋 불러오기', '데이터 전처리', '데이터 전처리 실습',
                                '지도학습 : 회귀모델', '지도학습 : 분류모델 - 로지스틱 회귀', '지도학습 : 분류모델 - SVM', '지도학습 : 분류모델 - KNN', '지도학습 : 분류모델 - 나이브베이즈', '지도학습 : 분류모델 - 의사결정나무',
                                '비지도학습 : 군집화모델 - k-means clustering', '비지도학습 : 군집화모델 - 계층적 군집화', '비지도학습 : 군집화모델 - DBSCAN', '비지도학습 : 차원축소 - PCA', '비지도학습 : 차원축소 - t-SNE', '비지도학습 : 차원축소 - LDA',
                                '앙상블 학습 - 배깅과 부스팅', '앙상블 학습 - 랜덤 포레스트', '앙상블 학습 - 그래디언트 부스팅 머신 (GBM)', '앙상블 학습 - XGBoost'])
    st.write(f'{option}의 "{textbook}" 교재에 대한 퀴즈를 시작하겠습니다!')

elif option == '딥러닝':
    textbook = st.sidebar.selectbox('어떤 교재를 선택할까요?',
                                ['딥러닝 개념을 잡아봅시다!', '신경망의 기본 원리', '딥러닝 실습 환경 구축',
                                '인공 신경망(ANN)', '합성곱 신경망(CNN)', '순환 신경망(RNN)',
                                '어텐션 (Attention) 메커니즘', '자연어 처리(NLP) 모델',
                                'ResNet', '이미지 처리 모델',
                                '오토인코더', '생성형 모델', '전이학습',
                                '과적합 방지 기법', '하이퍼파라미터 튜닝', '모델 평가와 검증 및 Pytorch 문법 정리'])
    st.write(f'{option}의 "{textbook}" 교재에 대한 퀴즈를 시작하겠습니다!')

elif option == 'LLM & RAG':
    textbook = st.sidebar.selectbox('어떤 교재를 선택할까요?',
                                ['LLM이란? 강의소개!', 'LLM 시스템 형성을 위한 다양한 기법 및 요소 개념 익히기', 'OpenAI Playground 사용법 가이드',
                                '프롬프트 엔지니어링 개념잡기!', '프롬프트 엔지니어링 맛보기', '프롬프트 엔지니어링의 기본 원칙',
                                'Shot 계열의 프롬프팅 기법 배워보기', 'Act As 류의 프롬프팅 기법 배우기', '논리적인 추론 강화하기',
                                '대화를 활용한 프롬프팅 기법', '형식 지정 기법',
                                'LLM의 사용 준비하기', 'Vector DB 개념 및 RAG (Retrieval-Augmented Generation) 개념', '텍스트 처리의 핵심 기법과 임베딩 활용하기', 'LangChain: 개념과 활용', 'Python LangChain과 FAISS', 'Sentence-Transformer, Word2Vec, 그리고 Transformer 기반 임베딩', '문서 임베딩 실습하기'])
    st.write(f'{option}의 "{textbook}" 교재에 대한 퀴즈를 시작하겠습니다!')

st.sidebar.header('대화 내역')

# .env 파일에서 api 키 가져오기
API_KEY = os.getenv('openai_api_key')

# OpenAI API 키 설정
if API_KEY:
    openai.api_key = API_KEY
else:
    st.error("API 키가 설정되지 않았습니다. .env 파일을 확인하세요.")
    st.stop()

# CSV 파일명
CSV_FILE = "chat_history.csv"

# CSV 파일이 존재하면 불러오기, 없으면 새로 생성
try:
    chat_history_df = pd.read_csv(CSV_FILE)
except FileNotFoundError:
    chat_history_df = pd.DataFrame(columns=["ChatID", "Role", "Content"])



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

# 기존 채팅 기록 표시
for content in st.session_state.chat_session:
    with st.chat_message("ai" if content["role"] == "assistant" else "user"):
        st.markdown(content["content"])

# 사용자 입력 받기
if prompt := st.chat_input("메시지를 입력하세요."):

    with st.chat_message("user"):
        st.markdown(prompt)
        # 사용자의 입력을 채팅 기록에 추가
        st.session_state.chat_session.append({"role": "user", "content": prompt})

    # GPT 모델로부터 응답 받기
    with st.chat_message("ai"):
        response = openai.ChatCompletion.create(
            model="gpt-4",  # 사용할 모델 지정 (gpt-4 또는 gpt-3.5-turbo 등)
            messages=st.session_state.chat_session
        )
        # GPT의 응답 텍스트
        reply = response["choices"][0]["message"]["content"]
        st.markdown(reply)
        # 응답을 채팅 기록에 추가
        st.session_state.chat_session.append({"role": "assistant", "content": reply})

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
if len(chat_history_df) > 0:
    # 이미 버튼이 만들어져 있다면 대화 목록 표시
    for chat_id in chat_history_df["ChatID"].unique():
        button_label = get_button_label(chat_history_df, chat_id)
        if st.sidebar.button(button_label):
            current_chat_id = chat_id
            loaded_chat = chat_history_df[chat_history_df["ChatID"] == chat_id]
            loaded_chat_string = "\n".join(f"{row['Role']}: {row['Content']}" for _, row in loaded_chat.iterrows())
            st.text_area("Chat History", value=loaded_chat_string, height=300)
else:
    st.sidebar.write("저장된 대화가 없습니다.")