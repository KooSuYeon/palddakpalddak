from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from model import create_quiz, check_answer # rag_model.py 파일을 임포트
import uvicorn

app = FastAPI()

# API 요청 모델
class QuizRequest(BaseModel):
    topic: str

class AnswerRequest(BaseModel):
    context: str
    answer: str

# 퀴즈 생성 엔드포인트
@app.post("/generate_quiz")
async def generate_quiz(request: QuizRequest):
    try:
        quiz = create_quiz(request.topic)
        return {"quiz": quiz}
    except Exception as e:
        raise HTTPException
    

# 답변 검토 엔드포인트
@app.post("/check_answer")
async def check_answer(request):
    try:
        result = check_answer(request.context, request.answer)
        return {"result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
