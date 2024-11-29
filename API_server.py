from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, validator
from rag_model import get_question, get_feedback, get_session_no  # rag_model.py 파일을 임포트
import uvicorn
import logging

############ 로그 파일 생성 ######################
logging.basicConfig(
    filename="API_server.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logging.info("Backend API started.")
logger = logging.getLogger(__name__)
##################################################

# 모델 메서드 호출에 필요한 전역변수 선언
global user_id
global session_no
global type_
global order
global quiz

user_id: str = "sj5black"
session_no: int = get_session_no(user_id)
type_: str = "python"
order: int = 3

# FastAPI 애플리케이션 생성
app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 보안상 필요한 도메인만 추가
    allow_credentials=True,
    allow_methods=["GET", "POST"], # GET, POST 요청만 허용
    allow_headers=["*"],
)

###############################################################
###################### 클래스 선언 ############################
###############################################################

# API 요청 모델
class QuizRequest(BaseModel):
    topic: str
    
    #@validator("topic")
    #def validate_topic(cls, value):
    #    if not value.strip():
    #        raise ValueError("Topic must not be empty.")
    #    return value
    pass

class AnswerRequest(BaseModel):
    # context: str
    user_answer: str

    # @validator("context", "answer")
    # def validate_not_empty(cls, value):
    #     if not value.strip():
    #         raise ValueError("Fields must not be empty.")
    #     return value

class TypeRequest(BaseModel):
    sidebox_type: str
###############################################################
###################### 요청 메서드 처리 #######################
###############################################################

# 서버 로드 체크
@app.get("/server_check")
async def server_check():
    return {"status": "ok"}

# type 변수 
@app.post("/set_type")
async def set_type(request: TypeRequest):
    type_ = request.sidebox_type  # 받은 type을 전역변수에 저장
    logger.info(f"set_type test -> type_ : {type_}")
    return {"message": f"Selected type has been set to: {type_}"}

# 퀴즈 생성 엔드포인트
@app.post("/generate_quiz")
async def generate_quiz(request: QuizRequest):
    logger.info(f"generate_quiz initial type_ : {type_}")
    logger.info(f"generate_quiz initial session_no : {session_no}")
    logger.info(f"generate_quiz initial user_id : {user_id}")
    logger.info(f"generate_quiz initial request.topic : {request.topic}")
    logger.info(f"generate_quiz initial order : {order}")

    # if type_ != request.topic:
    #     raise HTTPException(status_code=400, detail="선택된 주제와 AI가 생성한 topic이 일치하지 않았습니다.")
    try:
        logger.info(f"Generating quiz for topic: {request.topic}")
        quiz = get_question(session_no, user_id, request.topic, order)
        return {"퀴즈": quiz}
    except ValueError as ve:
        logger.error(f"ValueError: {ve}")
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error.")

# 답변 검토 엔드포인트
@app.post("/check_answer")
async def check_answer(request: AnswerRequest):
    try:
        logger.info(f"Checking answer for context {request.user_answer}")
        feedback = get_feedback(session_no, user_id, type_, order, quiz, request.user_answer)
        return {"피드백": feedback}
    except ValueError as ve:
        logger.error(f"ValueError: {ve}")
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error.")

# 메인 함수
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
