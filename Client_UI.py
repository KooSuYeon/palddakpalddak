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
API_BASE_URL = "http://127.0.0.1:8002"  # FastAPI 서버 로컬 호스트 값
# API_BASE_URL = "http://0.0.0.0:8000"  # FastAPI 서버 외부 연결 시

logging.basicConfig(
    filename="UI.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logging.info("Streamlit UI started.")

################# FastAPI 서버 실행 #################################
subprocess.Popen(["uvicorn", "API_server:app", "--reload", "--port", "8002"])
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

# 사이드바 구성
st.sidebar.header('목차 선택')
theme = st.sidebar.selectbox(
    '주제를 선택하세요.',
    options=list(theme_to_type.keys()),
    key="selected_theme",  # 상태 저장 키
    on_change=update_api_on_select  # 값 변경 시 콜백 호출
)
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