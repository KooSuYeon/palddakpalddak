import streamlit as st
from v3_Client_UI_chat import chat_page
from v3_Client_UI_login import login_page


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
    <p class="custom-title">복습 퀴즈 챗봇📖</p>
    """,
    unsafe_allow_html=True,
)



# 앱 실행
if 'page' not in st.session_state:
    st.session_state.page = 'login'  # 초기 페이지 설정

if st.session_state.page == 'login':
    login_page()
elif st.session_state.page == 'chat':
    chat_page()