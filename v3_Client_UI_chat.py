import streamlit as st
import pandas as pd
import uuid
import requests

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



topics =  ['파이썬_라이브러리', '머신러닝', '딥러닝', 'LLM_RAG', 'OPENSOURCE']
languages = ["한국어", "영어", "일본어"]
CSV_FILE = "chat_history.csv"


# CSV 파일이 존재하면 불러오기, 없으면 새로 생

    
def initialize_chat_history():
    if "chat_session" not in st.session_state:
        st.session_state.chat_session = []
    if "chat_history_df" not in st.session_state:
      st.session_state.chat_history_df = pd.DataFrame(columns=["UserID", "ChatID", "Role", "Content"])
    

def get_quiz_from_api():
    # POST 요청에서 보내는 데이터
    data = {"topic": "asdf"}
    
    response = requests.post("http://localhost:8000/generate_quiz", json=data)
    if response.status_code == 200:
        return response.json()  # JSON 응답 반환
    else:
        st.error("Quiz을 불러오는 데 실패했습니다.")
        return None

def get_feedback_from_api():
   # POST 요청에서 보내는 데이터
   data = {"user_answer": "answer"}
   
   response = requests.post("http://localhost:8000/check_answer", json=data)
   if response.status_code == 200:
       return response.json()  # JSON 응답 반환
   else:
       st.error("Feedback을 불러오는 데 실패했습니다.")
       return None   

# 사용자 입력 받기
def chat_page():
    try:
        chat_history_df = pd.read_csv(CSV_FILE)
        st.session_state.chat_history_df = chat_history_df
    except FileNotFoundError:
        chat_history_df = pd.DataFrame(columns=["UserID", "ChatID", "Role", "Content"])
        st.session_state.chat_history_df = chat_history_df



    # 새 대화 세션 시작
    if "chat_session" not in st.session_state:
        st.session_state.chat_session = []

    if "current_chat_id" not in st.session_state:
        st.session_state.current_chat_id = str(uuid.uuid4())[:8]

    if 'user_id' not in st.session_state:
        st.session_state.user_id = None

    if 'theme' not in st.session_state:
        st.session_state.theme = False


        # Streamlit에서 상태 관리: 첫 실행인지 확인
    if "quiz_shown" not in st.session_state:
        st.session_state.quiz_shown = False
        st.session_state.chat_session = []  # 대화 기록 저장
        st.session_state.current_quiz = None  # 현재 퀴즈

    # 기존 채팅 기록 표시
    for content in st.session_state.chat_session:
        with st.chat_message("ai" if content["role"] == "assistant" else "user"):
            st.markdown(content["content"])
        # 사이드바 구성하기
    st.sidebar.header('주제 선택')
    
    # selectbox로 주제 선택
    
    # 테마 선택
    theme = st.sidebar.selectbox('주제를 선택하세요.', topics)
    
    # 교재 선택
    textbook = st.sidebar.selectbox('어떤 교재를 선택할까요?', list(mapping_data[theme].keys()))
    
    # 언어 선택
    language= st.sidebar.segmented_control(
        "언어", languages, selection_mode="single", default="한국어"
    )
    
    # 녹음 기능
    audio_value = st.sidebar.audio_input("녹음해주세요.")
    
    if audio_value:
        st.sidebar.audio(audio_value)
        
    st.sidebar.header('대화 내역')

    
    # 사이드바에 '대화 저장' 버튼 추가
    if st.sidebar.button('새 세션 만들기'):
        initialize_chat_history()
        chat_history_df.to_csv("chat_history.txt", sep="\t", index=False)
        st.sidebar.write("대화가 TXT 파일로 저장되었습니다.")
        st.session_state.theme = theme
        st.session_state.language = language
        st.session_state.textbook = textbook

    initialize_chat_history()  # 초기화 함수 호출하여 chat_history_df 세션 상태에 추가
  
    
    # 처음 실행 시 퀴즈를 출력
    if not st.session_state.quiz_shown:
        st.session_state.current_quiz = get_quiz_from_api()  # 퀴즈 가져오기
        with st.chat_message("ai"):
            st.markdown(st.session_state.current_quiz)
            st.session_state.chat_session.append(
                {"role": "assistant", "content": st.session_state.current_quiz}
            )
        st.session_state.quiz_shown = True
    
    # 사용자 입력 처리
    if prompt := st.chat_input("메시지를 입력하세요."):
        # 1. 사용자 입력 출력
        with st.chat_message("user"):
            st.markdown(prompt)
            st.session_state.chat_session.append({"role": "user", "content": prompt})
    
        # 2. 피드백 출력
        with st.chat_message("ai"):
            feedback = get_feedback_from_api()  # API에서 피드백 가져오기
            st.markdown(feedback)
            st.session_state.chat_session.append({"role": "assistant", "content": feedback})
    
        # 3. 새로운 퀴즈 출력
        st.session_state.current_quiz = get_quiz_from_api()  # 새 퀴즈 가져오기
        with st.chat_message("ai"):
            st.markdown(st.session_state.current_quiz)
            st.session_state.chat_session.append(
                {"role": "assistant", "content": st.session_state.current_quiz}
            )
    
        # 대화 내역을 CSV에 저장
        new_rows = []

        for content in st.session_state.chat_session:
            new_rows.append({
                "UserID": st.session_state.user_id,
                "ChatID": st.session_state.current_chat_id,
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
    def get_button_label(user_id, chat_df, chat_id):
        # 가장 마지막 사용자 메시지를 가져옵니다.
        user_messages = chat_df[(chat_df["ChatID"] == chat_id) & (chat_df["Role"] == "user") & (chat_df["UserID"] == user_id)]
        if not user_messages.empty:  # 'User' 메시지가 존재하는 경우
            last_user_message = user_messages.iloc[-1]["Content"]
            return f"Chat {chat_id[0:7]}: {' '.join(last_user_message.split()[:5])}..."  # 마지막 메시지의 첫 5단어를 표시
        else:
            return f"Chat {chat_id[0:7]}: No User message found"  # 메시지가 없으면 안내 문구 표시

    # 사이드바에 저장된 대화 기록을 표시
    if len(st.session_state.chat_history_df) > 0:
        # UserID가 세션의 user_id와 같은 데이터만 필터링
        user_chat_df = st.session_state.chat_history_df[st.session_state.chat_history_df["UserID"] == st.session_state.user_id]
        
        if len(user_chat_df) > 0:
            # 필터링된 데이터에서 ChatID별로 버튼 생성
            for chat_id in user_chat_df["ChatID"].unique():
                button_label = get_button_label(st.session_state.user_id, user_chat_df, chat_id)
                
                # 사이드바 버튼 생성
                if st.sidebar.button(button_label, key=f"btn_{chat_id}"):
                    # 선택된 ChatID의 대화 데이터 필터링
                    loaded_chat = user_chat_df[user_chat_df["ChatID"] == chat_id]
                    loaded_chat_string = "\n".join(f"{row['Role']}: {row['Content']}" for _, row in loaded_chat.iterrows())
                    
                    # 텍스트 영역에 대화 내용 표시
                    st.text_area("Chat History", value=loaded_chat_string, height=300)
        else:
            st.sidebar.write("현재 사용자에 대한 저장된 대화가 없습니다.")
    else:
        st.sidebar.write("저장된 대화가 없습니다.")
    
