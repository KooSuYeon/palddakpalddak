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

# 라디오 버튼으로 큰 주제 선택
option = st.sidebar.selectbox('주제를 선택하세요.', ['파이썬 라이브러리', '머신러닝', '딥러닝', 'LLM, RAG', 'AI 활용'])
if option == '파이썬 라이브러리':
    theme = st.sidebar.selectbox('어떤 교재를 선택할까요?', ['파이썬 라이브러리', '머신러닝', '딥러닝', 'LLM & RAG', 'AI 활용'])
    st.write(f'{theme}')

elif option == '머신러닝':
    theme = st.sidebar.selectbox('어떤 교재를 선택할까요?', ['파이썬', '머신러닝', '딥러닝', 'LLM, RAG'])
    st.write(f'{theme}')

elif option == '딥러닝':
    theme = st.sidebar.selectbox(
        '어떤 교재를 선택할까요?',
        ['1. 딥러닝 개념을 잡아봅시다!', '2. 신경망의 기본 원리', '딥러닝을 배워야 하는 이유',
        '퍼셉트론과 다층 퍼셉트론(XOR 문제 포함)', '다층 퍼셉트론(MLP)', '활성화 함수', '손실 함수와 최적화 알고리즘', '역전파에 대해 알아볼까요?',
        'conda를 이용한 환경 설정', 'jupyter notebook', '가상환경 설치 및 jupyter notebook 연결', 'pytorch 설치​환경 활성화', 
        '기본 구조와 동작원리'])
    

    theme = st.sidebar.selectbox(
        '어떤 교재를 선택할까요?',
        ['딥러닝이란 무엇일까요?', '딥러닝의 역사와 활용 방안', '딥러닝을 배워야 하는 이유',
        '퍼셉트론과 다층 퍼셉트론(XOR 문제 포함)', '다층 퍼셉트론(MLP)', '활성화 함수', '손실 함수와 최적화 알고리즘', '역전파에 대해 알아볼까요?',
        'conda를 이용한 환경 설정', 'jupyter notebook', '가상환경 설치 및 jupyter notebook 연결', 'pytorch 설치​환경 활성화', 
        '기본 구조와 동작원리'])
    st.write(f'{theme}')

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