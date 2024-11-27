import openai
import streamlit as st
from dotenv import load_dotenv
import os

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
theme = st.sidebar.selectbox('주제를 선택하세요.', ['파이썬 라이브러리', '머신러닝', '딥러닝', 'LLM, RAG', 'AI 활용'])
st.write(f'{theme}에 대한 퀴즈를 내보겠습니다!')


# .env 파일에서 api 키 가져오기
API_KEY = os.getenv('openai_api_key')

# OpenAI API 키 설정
if API_KEY:
    openai.api_key = API_KEY
else:
    st.error("API 키가 설정되지 않았습니다. .env 파일을 확인하세요.")
    st.stop()

@st.cache_data  # 새로고침하면 대화 내역이 사라짐
# @st.cache_resource  # 새로고침해도 대화 내역이 남아있음
def start_chat_session():
    # 채팅 기록을 초기화합니다.
    return []

if "chat_session" not in st.session_state:
    st.session_state["chat_session"] = start_chat_session()

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
