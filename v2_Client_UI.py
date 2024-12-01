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
from streamlit.runtime.scriptrunner import RerunException # 페이지 새로고침

load_dotenv() #환경변수 값 로드 (API 포함)

# 페이지 구성
st.set_page_config(
    page_title='팔딱이',
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
API_BASE_URL = "http://127.0.0.1:8001"  # FastAPI 서버 로컬 호스트 값
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
    process = subprocess.Popen(["uvicorn", "v1_API_server:app", "--reload", "--port", "8001"])
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
    st.success("API 서버가 시작되었습니다.")

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
    st.session_state.language = "한국어"
if 'chat_log' not in st.session_state:
    st.session_state.chat_log = ""
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
        'RAG 기반 비구조화된 데이터를 기반으로 질문에 답변하는 오픈 소스': 1,
        '다양한 유형의 소스(PDF, YouTube 동영상) 로부터 데이터를 가공해 RAG 파이프 라인을 구현하는 예제의 컬럼': 2,
        'ResNet을 이용한 개 고양이 분류기': 3,
        'GAN을 이용한 MNIST 숫자 생성 모델': 4,
        'ETF 예측 모델 (다중선형회귀, XGBoost, ARIMA)': 5,
        '서울시 공공 자전거 분석': 6,
        '무더위 쉼터 데이터': 7
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
st.markdown('<p class="custom-title">📖복습 퀴즈 챗봇</p>', unsafe_allow_html=True)

# 기존 채팅 기록 표시
for content in st.session_state.chat_session:
    with st.chat_message("ai" if content["role"] == "assistant" else "user"):
        st.markdown(content["content"])

# 전체 채팅 화면
def chat_page():
    # 초기화 함수 호출하여 chat_history_df 세션 상태에 추가
    initialize_chat_history()
    
    # st.write("⬅️⬅️왼쪽에 있는 사이드바에서 원하는 주제와 교재를 선택해주세요.")
    ################ 콜백 함수 선언 (API 서버에 요청) ######################
    # 대주제 변경 시
    def update_api_type():
        st.session_state.type_ = theme_to_type.get(st.session_state.selected_theme)
        response = requests.post(f"{API_BASE_URL}/set_big_topic", json={"big_topic": st.session_state.type_})
        if response.status_code == 200:
            st.success(f"type_ 값 '{st.session_state.type_}'으로 서버전송 성공!")
        else:
            st.error("type_ 값 서버전송 실패: " + response.text)

    # 소제목 변경 시
    def update_api_order():
        st.session_state.order = mapping_data[theme].get(st.session_state.order_str)
        response = requests.post(f"{API_BASE_URL}/set_small_topic", json={"small_topic_order": st.session_state.order})
        if response.status_code == 200:
            st.success(f"order 값 '{st.session_state.order}'으로 서버전송 성공!")
        else:
            st.error("order 값 서버전송 실패: " + response.text)

    # 언어 변경 시
    def update_language():
        selected_language = st.session_state.language
        response = requests.post(f"{API_BASE_URL}/set_language", json={"lang": selected_language})
        if response.status_code == 200:
            st.success(f"'{selected_language}'로 언어 변경 성공!")
        else:
            st.error("language값 서버전송 실패: " + response.text)
    ######################################################################

    # 사이드바 구성하기
    st.sidebar.header('주제 선택')

    theme = st.sidebar.selectbox('주제를 선택하세요.', options=list(theme_to_type.keys()), key="selected_theme", on_change=update_api_type)

    if theme == '파이썬_라이브러리':
        textbook = st.sidebar.selectbox('어떤 교재를 선택할까요?', options=list(mapping_data[theme].keys()), key="order_str", on_change=update_api_order)
    elif theme == '머신러닝':
        textbook = st.sidebar.selectbox('어떤 교재를 선택할까요?', options=list(mapping_data[theme].keys()), key="order_str", on_change=update_api_order)
    elif theme == '딥러닝':
        textbook = st.sidebar.selectbox('어떤 교재를 선택할까요?', options=list(mapping_data[theme].keys()), key="order_str", on_change=update_api_order)
    elif theme == 'LLM_RAG':
        textbook = st.sidebar.selectbox('어떤 교재를 선택할까요?', options=list(mapping_data[theme].keys()), key="order_str", on_change=update_api_order)
    elif theme == 'OPENSOURCE':
        textbook = st.sidebar.selectbox('어떤 교재를 선택할까요?', options=list(mapping_data[theme].keys()), key="order_str", on_change=update_api_order)

    # 언어 선택
    language_list = ["한국어", "영어", "일본어"]
    selection = st.sidebar.segmented_control("언어", language_list, selection_mode="single", default="한국어", key="language", on_change=update_language)
    st.sidebar.markdown(f"**{selection}**가 선택되었습니다.")
    
    # 녹음 기능
    audio_value = st.sidebar.audio_input("음성으로 대화해보세요.")
    
    if audio_value:
        st.sidebar.audio(audio_value)
        
    st.sidebar.header('채팅 기록 보기')
    
    # 퀴즈 생성 함수
    def generate_quiz():
        st.session_state.quiz_status_check = 1
        try:
            st.write(f'현재 selected_theme : {st.session_state.selected_theme}')
            st.write(f'현재 user_id : {st.session_state.user_id}')
            st.write(f'현재 session_no : {st.session_state.session_no}')
            st.write(f'현재 type_ : {st.session_state.type_}')
            st.write(f'현재 order : {st.session_state.order}')
            st.write(f'현재 order_str : {st.session_state.order_str}')
            st.write(f'현재 language : {st.session_state.language}')
            st.write(f'현재 chat_session : {st.session_state.chat_session}')
            st.write(f'현재 chat_history_df : {st.session_state.chat_history_df}')
            st.write(f'현재 chat_log : {st.session_state.chat_log}')

            response = requests.post(f"{API_BASE_URL}/generate_quiz", json={"topic": st.session_state.type_})
            response.raise_for_status()  # HTTP 에러 발생 시 예외를 발생시킴
            quiz_data = response.json()  # JSON 데이터 추출
            st.session_state.quiz_data = quiz_data
            with st.chat_message("ai"):
                st.write(f'{theme}에 대한 퀴즈를 내보겠습니다!')
                st.markdown(quiz_data["QUIZ"])

            # 퀴즈 내용을 채팅 기록에 추가
            st.session_state.chat_session.append({"role": "assistant", "content": quiz_data["QUIZ"]})
            
        except requests.exceptions.RequestException as e:
            logging.error(f"Error making API request: {e}")
            st.error(f"API 호출 실패: {e}")

    

    if prompt := st.chat_input("메시지를 입력하세요."):
        
        # 유저의 답변
        with st.chat_message("user"):
            st.markdown(prompt)
            # 사용자의 입력을 채팅 기록에 추가
            st.session_state.chat_session.append({"role": "user", "content": prompt})

        # AI 에게 피드백 받기
        with st.chat_message("ai"):
            if st.session_state.quiz_status_check == 1 :
                quiz_content = st.session_state.quiz_data.get("QUIZ", "내용 없음") # 딕셔너리 형태의 quiz_data 에서 실제 QUIZ 값만 추출 (str 형식)
                response = requests.post(f"{API_BASE_URL}/check_answer", json={"quiz": quiz_content, "user_answer" : prompt})
                response.raise_for_status()  # HTTP 에러 발생 시 예외를 발생시킴
                feedback_data = response.json()
                st.markdown(feedback_data["FeedBack"])
                feedback_content = feedback_data.get("FeedBack","내용 없음")
                # 응답을 채팅 기록에 추가
                st.session_state.chat_session.append({"role": "assistant", "content": feedback_content})
                st.session_state.quiz_status_check += 1
            elif st.session_state.quiz_status_check > 1 :
                st.markdown("(팔딱이가 답변을 작성중입니다...)")
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
                    st.markdown(gpt_answer_str)  # 응답 출력
                    # 응답을 채팅 기록에 추가
                    st.session_state.chat_session.append({"role": "assistant", "content": gpt_answer_str})
                except openai.OpenAIError as e:
                    st.error(f"GPT 응답 생성 중 오류가 발생했습니다: {e}")

            elif st.session_state.quiz_status_check == 0 :
                st.markdown("QUIZ 시작 버튼을 눌러 퀴즈를 시작해주세요.") # 최초 퀴즈 생성이 되지 않은 경우, 기본값을 반환

        # 대화 내역을 CSV에 저장
        chat_id = st.session_state.user_id
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

    if st.button('QUIZ 시작'):
        generate_quiz()

    # 대화 내역을 선택할 수 있는 버튼 추가
    def get_button_label(chat_df, chat_id):
        # 가장 마지막 사용자 메시지를 가져옵니다.
        user_messages = chat_df[(chat_df["ChatID"] == chat_id) & (chat_df["Role"] == "user")]
        if not user_messages.empty:  # 'User' 메시지가 존재하는 경우
            last_user_message = user_messages.iloc[-1]["Content"]
            return f"{chat_id[0:7]} : {' '.join(last_user_message.split()[:10])}"  # 마지막 메시지의 첫 10단어를 표시
        else:
            return f"{chat_id[0:7]} : No User message found"  # 메시지가 없으면 안내 문구 표시

    # 사이드바에 저장된 대화 기록을 표시
    if len(st.session_state.chat_history_df) > 0:
        # 이미 버튼이 만들어져 있다면 대화 목록 표시
        for chat_id in st.session_state.chat_history_df["ChatID"].unique():
            button_label = get_button_label(st.session_state.chat_history_df, chat_id)
            if st.sidebar.button(button_label):
                loaded_chat = st.session_state.chat_history_df[st.session_state.chat_history_df["ChatID"] == chat_id]
                loaded_chat_string = "\n".join(f"{row['Role']}: {row['Content']}" for _, row in loaded_chat.iterrows())
                st.session_state.chat_log = loaded_chat_string
                st.text_area("채팅 내역", value=loaded_chat_string, height=300)
    else:
        st.sidebar.write("진행중인 대화가 없습니다.")
    
    # 사이드바에 '대화 저장' 버튼 추가
    if st.sidebar.button('전체 대화내역 저장'):
        if len(st.session_state.chat_history_df) > 0:
            for chat_id in st.session_state.chat_history_df["ChatID"].unique():
                loaded_chat = st.session_state.chat_history_df[st.session_state.chat_history_df["ChatID"] == chat_id]
                loaded_chat_string = "\n".join(f"{row['Role']}: {row['Content']}" for _, row in loaded_chat.iterrows())
                st.session_state.chat_log = loaded_chat_string

        # 서버 요청
        try:
            response = requests.post(
                f"{API_BASE_URL}/save_conversation",
                json={"requested_user_id": st.session_state.user_id, "chatlog": "st.session_state.chat_log"}
            )
            response.raise_for_status()
            st.success("채팅 로그 서버전송 성공!")
        except requests.exceptions.RequestException as e:
            st.error(f"서버 요청 실패: {e}")
        # else:
        #     st.sidebar.write("저장할 대화가 없습니다.")
    
    

# ID 입력 화면
def login_page():
    # 페이지 구성
    # st.set_page_config(
    #     page_title='팔딱이',
    #     page_icon='🤖',
    #     layout='centered',
    #     initial_sidebar_state='auto'
    # )
    
    st.markdown("""
        <div style="text-align: center;">
            <h1>🤖팔딱팔딱 AI QUIZ🤖</h1>
            <img src="https://viralsolutions.net/wp-content/uploads/2019/06/shutterstock_749036344.jpg" width="1280" />
        </div>
        <div style="margin-top: 30px;">  <!-- ID 입력창과 이미지 사이에 30px의 여백 추가 -->
        </div>
    """, unsafe_allow_html=True)

    user_id = st.text_input("ID를 입력하세요.", key="custom_input", placeholder="ID 입력", label_visibility="visible", help="ID를 입력하세요.")
    
    # 버튼 스타일링을 위한 CSS 적용
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
            # ID를 입력하면 채팅 페이지로 이동
            st.session_state.user_id = user_id

            try:
                # API 호출
                response = requests.get(f"{API_BASE_URL}/get_history/{user_id}")
                
                # 응답 상태 확인
                if response.status_code == 200:
                    files = response.json()  # API로부터 JSON 데이터를 받아옴

                    # 받은 파일을 로컬 디렉토리에 저장
                    for file in files:
                        file_path = os.path.join("./text_files", file["file_name"])
                        with open(file_path, "w", encoding="utf-8") as f:
                            f.write(file["content"])

                    st.success(f"안녕하세요! {st.session_state['user_id']}님 반갑습니다! " 
                            f"'{len(files)}'개의 파일을 다운로드했습니다. '로그인' 버튼을 한번 더 누르면 채팅이 시작됩니다.")
                else:
                    st.error(f"서버 요청 실패: {response.status_code} - {response.text}")

            except requests.exceptions.RequestException as e:
                st.error(f"서버와 통신하는 중 오류가 발생했습니다: {e}")

            st.success(f"안녕하세요! {st.session_state['user_id']}님 반갑습니다! '로그인' 버튼을 한번 더 누르면 채팅이 시작됩니다.")
            st.session_state.page = 'chat'  # 페이지를 'chat'으로 설정
            
        else:
            st.error('채팅에 사용할 ID를 먼저 입력해주세요.')

# 앱 실행
if st.session_state.page == 'login':
    login_page()
elif st.session_state.page == 'chat':
    chat_page()
# elif st.session_state.page == 'chatting':
#     user_chatting()