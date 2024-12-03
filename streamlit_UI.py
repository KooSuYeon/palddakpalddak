import streamlit as st
import pandas as pd
import numpy as np
import time

st.set_page_config(
    page_title="팔딱팔딱 AI QUIZ", 
    page_icon="❓"
)
# 디폴트 페이지 설정
def show_default_page():
    st.title('AI 학습 퀴즈')
    st.write("왼쪽 사이드바에서 원하는 과목들을 선택해주세요.")
    st.image("https://viralsolutions.net/wp-content/uploads/2019/06/shutterstock_749036344.jpg")

# 사이드바 구성
st.sidebar.title("과목 선택")
select_subject = st.sidebar.selectbox(
    "학습하고 싶은 과목을 선택하세요", 
    ["==선택==", "파이썬 라이브러리", "LLM, RAG", "AI활용", "머신러닝", "딥러닝"], 
    key="selected_option")
# 선택값을 저장하는 변수 초기화
if "selected_option" not in st.session_state:
    st.session_state["selected_option"] = "==선택=="

start_button = st.sidebar.button("퀴즈 시작!") #시작버튼
#if start_button:
    #start_button을 누르면 True값을 가집니다. 이 때 발생할 값들 정리

reset_button = st.sidebar.button("퀴즈 초기화!") #리셋버튼
if reset_button:
    st.cache_data.clear()
    st.session_state["selected_option"] = "==선택=="

if select_subject == "==선택==":
    show_default_page()

st.write(f"현재 선택된 과목은 {select_subject}입니다.")

st.title("AI 학습 QUIZ")

st.text("텍스트칸입니다. 아직 쓸 말이 없네요")

answers = []
quiz_data = [ #예시문제
    {
        "question": "1+1의 값은?(AI활용)",
        "options": ["1. 0", "2. 1", "3. 2", "4. 3"],
        "answer": "3. 2"
    },
    {
        "question": "3x3의 값은?(머신러닝)",
        "options": ["1. 3", "2. 6", "3. 9", "4. 12"],
        "answer": "3. 9"
    },
    {
        "question": "대한민국의 수도는?(딥러닝)",
        "options": ["1. 서울", "2. 부산", "3. 제주", "4. 스파르타사무실"],
        "answer": "1. 서울"
    }
]

# 진행 바와 Iteration 설정
show_progress = st.checkbox("퀴즈 진행 상태 표시", value=True)
if show_progress:
    latest_iteration = st.empty()
    bar_container = st.empty()
    bar = bar_container.progress(0)

# 점수 초기화
st.session_state["score"] = 0
total_questions = len(quiz_data)

for idx, quiz in enumerate(quiz_data):
    if show_progress:
        latest_iteration.text(f"문제 {idx + 1} / {total_questions}")
        bar.progress(int((idx / total_questions) * 100))
        
    # 퀴즈 출력
    st.header(f"문제 {idx + 1}")
    st.write(quiz["question"])
    answers.append(st.radio("답을 선택하세요:", quiz["options"], key=f"question_{idx}"))

#구분선
st.divider()

# 최종 점수 확인 및 진행 바 제거
if st.button("최종 점수 확인"):
    for i, quiz in enumerate(quiz_data):
        if answers[i] == quiz["answer"]:
            st.session_state["score"] += 1
    st.write(f"총 {total_questions}문제 중 {st.session_state['score']}문제를 맞췄습니다!")
    if st.session_state["score"] == total_questions:
        st.balloons()

    # 진행 상태 제거
    if show_progress:
        latest_iteration.empty()
        bar_container.empty()