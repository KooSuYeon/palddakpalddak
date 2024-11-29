#### 🤖 RAG를 활용한 Sparta 복습용 챗봇 만들기

---
📣 How To Use
```
- 사전에 공유된 구글 드라이브 링크에서 dataset 디렉터리를 다운로드 받아준 후 루트 디렉터리에 해당 디렉터리를 위치시켜 줍니다.
- pip install -r requirements.txt 를 통해 개발 할 때 사용한 라이브러리와 버전을 맞춰줍니다.
- 두 개의 터미널을 열어줍니다
- 첫 번째 터미널에 백엔드 서버를 켜줍니다!  uvicorn v1_API_server:app --reload
- 두 번째 터미널에 프론트엔드 파일을 실행시켜줍니다! streamlit run v1_Client_UI.py
```

---

🔍 Directory Structure

- 버전 관리가 필요한 파일들은 디렉터리 생성 후 넣어두었습니다.
- be, fe, rag_model : 백엔드, 프론트엔드, rag모델 버전 관리
- log: 백엔드 로그 관리
- rag_model_output : 사용자 채팅 관리

```
palddakpalddak/
├── __pycache__/
├── be/
│   ├── get_history_app.py
│   ├── get_specific_input.py
│   └── v0_API_server.py
├── fe/
│   ├── v1_fe.py
│   └── v2_fe.py
├── log/
│   └── UI.log
├── preProcessing/
│   ├── preProcessing_ml.py
│   └── preProcessing_open_source
├── rag_model/
│   ├── naive_rag_model.py
│   ├── v0_rag_chatbot.ipynb
│   ├── v1_rag_chatbot.ipynb
│   ├── v2_rag_chatbot.ipynb
│   ├── v3_rag_chatbot.ipynb
│   ├── v4_rag_chatbot.ipynb
│   └── v5_rag_chatbot.ipynb
├── rag_model_output/
├── .gitignore
├── rag_model.py
├── Readme.md
├── requirements.txt
├── st_ui_HSgoon.py
├── v1_API_server.py
└── v1_Client_UI.py

```

---
🎢 Timeline
- [x] 11/21 챗봇 종류 정하기 및 데이터셋 범위 설정
- [x] 11/22 데이터셋 준비 & 전처리 완료
- [x] 11/27 대화 내용 저장 & RAG 1차 고도화 완료
- [x] 11/27 프론트엔드 채팅 화면 구현 완료
- [x] 11/27 백엔드 질문 전송, 답변 전송 구현 완료
- [x] 11/28 대화 내용을 구체적인 이름을 가진 txt 파일 저장으로 수정 완료 & RAG 2차 고도화 완료
- [x] 11/28 프론트엔드 옵션 선택 사이드 바 구현 완료 & 변경된 input값 적용 완료
- [x] 11/28 대화 저장 API 생성 완료 & 백엔드 수정된 input 값 적용 완료
- [ ] 11/29 프론트엔드 user_id 로그인 화면 구현
- [ ] 11/29 프론트엔드 세션 버튼 생성
- [ ] 11/29 백엔드 세션 저장 API 연결
- [ ] 11/29 RAG 모델에 이전 대화 내용 기억하는 기능 추가하기

---
🦾 팀원
| 이름   | 역할                            |
|--------|---------------------------------|
| 구수연 | 팀장, AI 모델 개발, 데이터 수집, 대화세션 관리, 모델 성능개선 |
| 박성진 | 데이터 전처리, API, 데이터 수집, SA 문서, 서버 개발 |
| 윤수진 | Streamlit UI, 데이터 수집, 대화세션 관리 |
| 이현승 | Streamlit UI, 데이터 수집       |
| 김윤소 | AI 모델 개발, API, 데이터 수집  |



