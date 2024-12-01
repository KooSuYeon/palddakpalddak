# 🤖 RAG를 활용한 Sparta 복습용 챗봇 만들기

## 📖 목차
1. [How To Use](#How-To-Use)
2. [Directory Structure](#Directory-Structure)
3. [Timeline](#Timeline)
4. [팀원](#팀원)


5. [프로젝트 소개](#프로젝트-소개)
6. [프로젝트 계기](#프로젝트-계기)
7. [주요기능](#주요기능)
8. [서비스 구조](#서비스-구조)
---
## 📣 How To Use
```
- 사전에 공유된 구글 드라이브 링크에서 dataset 디렉터리를 다운로드 받아준 후 루트 디렉터리에 해당 디렉터리를 위치시켜 줍니다.
- pip install -r requirements.txt 를 통해 개발 할 때 사용한 라이브러리와 버전을 맞춰줍니다.
- 터미널을 열어줍니다.
- 터미널에 프론트엔드 파일을 실행시켜줍니다.   
  streamlit run v1_Client_UI.py
```

---

## 🔍 Directory Structure

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
## 🎢 Timeline
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
## 🦾 팀원
| 이름   | 역할                            |
|--------|---------------------------------|
| 구수연 | 팀장, AI 모델 개발, 데이터 수집, 대화세션 관리, 모델 성능개선 |
| 박성진 | 데이터 전처리, API, 데이터 수집, SA 문서, 서버 개발 |
| 윤수진 | Streamlit UI, 데이터 수집, 대화세션 관리 |
| 이현승 | Streamlit UI, 데이터 수집       |
| 김윤소 | AI 모델 개발, API, 데이터 수집  |

---
## 📋 프로젝트 소개
이 프로젝트는 AI 부트캠프에서 학습한 내용을 효과적으로 복습할 수 있는 퀴즈 기반 챗봇 서비스입니다. 사용자와의 대화를 통해 퀴즈를 제공하고, 답변에 대한 피드백을 제공함으로써 배운 내용을 더욱 탄탄히 다질 수 있습니다.

---
## 💡 프로젝트 계기
AI 부트캠프에서 다양한 강의를 듣고, 많은 주제를 학습했습니다. 하지만 방대한 내용을 체계적으로 복습하고 이해를 확인할 수 있는 도구가 부족하다는 것을 느꼈습니다. 이에, 강의 자료를 활용해 퀴즈를 제공하고, 답변에 대해 즉각적인 피드백을 제공하는 도구를 개발하고자 이 프로젝트를 기획했습니다. 이 챗봇은 배운 내용을 복습하며 학습 효과를 극대화할 수 있는 최적의 학습 도구가 될 것입니다.

---
## 💬주요기능
1. 퀴즈 생성 및 피드백 제공
   - 사이드바의 주제 및 교재 선택 기능을 통해 원하는 내용을 지정하고 QUIZ 시작 버튼을 누르면, AI가 관련된 퀴즈를 자동으로 생성합니다.
   - 사용자가 채팅으로 답변을 입력하면, AI는 답변에 대한 피드백을 제공합니다.
   - 추가적인 질문에 대해서도 AI가 대화를 이어가며 지속적으로 피드백을 제공합니다.
   - QUIZ 시작 버튼을 누를 때마다 중복되지 않는 새로운 퀴즈가 생성됩니다.
2. 언어 선택 기능
   - 언어 선택 버튼을 통해 원하는 언어를 선택하면, AI가 해당 언어로 답변을 생성합니다.
3. 음성 입력 및 피드백
   - AI가 퀴즈를 생성한 후, 사용자가 음성으로 답변을 입력하면, AI는 음성으로 피드백을 제공합니다.

이러한 기능을 통해 학습자는 다양한 방식으로 학습 내용을 복습하며 더욱 효과적으로 학습할 수 있습니다.

---
## 🗂️서비스 구조
1. 전체 흐름
   - 사용자가 사이드바에서 주제를 선택하고 QUIZ 시작 버튼을 클릭합니다.
   - AI 백엔드에서 선택한 주제와 교재에 맞는 퀴즈를 생성해 클라이언트로 전달합니다.
   - 사용자가 답변을 입력하면, AI가 백엔드에서 입력된 답변을 분석하고 피드백을 생성해 반환합니다.
   - 음성 모드에서는 음성 입력을 받아 텍스트로 변환하고, 다시 음성 피드백을 제공합니다.
2. 구성 요소
   - 프론트엔드
      - UI : 사이드바(주제/교재 선택, 언어 선택, 음성 녹음, 대화 내역), 채팅(대화 내용, 사용자 입력) 인터페이스로 구성되었습니다.
      - 사용자와 인터페이스 간의 상호작용 : 사용자의 선택 및 입력을 백엔드로 전달합니다.
   - 백엔드
      - 퀴즈 생성 모듈 : 강의 자료를 기반으로 적절한 퀴즈를 자동으로 생성합니다.
      - 답변 분석 및 피드백 생성 모듈 : 사용자 입력 데이터를 기반으로 피드백을 생성합니다.
   - 데이터베이스
      - 교재 및 강의 자료 파일을 저장합니다.
      - 생성된 퀴즈 기록, 사용자의 입력, 그에 기반한 피드백 데이터를 관리합니다.
3. 동작 예시
   -  사용자가 '머신러닝' 주제와 그와 관련된 교재를 선택한 후 `QUIZ 시작 버튼`을 통해 퀴즈를 요청 → AI가 QUIZ 생성 → 사용자가 답변 → AI가 피드백을 제공.
