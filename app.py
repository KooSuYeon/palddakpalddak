from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from model import create_quiz, create_answer # rag_model.py 파일을 임포트
import uvicorn
import os
import glob
from fastapi.responses import JSONResponse
from datetime import datetime

app = FastAPI()

FILE_DIR = "./text_files"

# API 요청 모델
class QuizRequest(BaseModel):
    topic: str

class AnswerRequest(BaseModel):
    context: str
    answer: str

class Conversation(BaseModel):
    user_id: int
    conversation: str


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
async def check_answer(request: AnswerRequest):
    try:
        result = create_answer(request.context, request.answer)
        return {"result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



@app.get("/get_history{user_id}")
async def get_history(user_id):
    file_path = os.path.join(FILE_DIR, f"{user_id}.txt")
    
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as file:
            content = file.read()
        return content
    else:
        raise HTTPException(status_code=404, detail="File not found")
    

# 대화 불러오기 api
@app.get("/get_history/{user_id}")
async def get_history(user_id: int):
    # 파일 경로 패턴
    pattern = os.path.join(FILE_DIR, f"{user_id}_*.txt")
    
    # 패턴에 맞는 파일 찾기
    file_paths = glob.glob(pattern)
    
    print(file_paths)
    if not file_paths:
        raise HTTPException(status_code=404, detail="Files not found")
    
    files_content = []
    for file_path in file_paths:
        with open(file_path, "r", encoding="utf-8") as file:
            content = file.read()
            files_content.append({
                "file_name": os.path.basename(file_path),
                "content": content
            })
    
    return JSONResponse(content=files_content)


# 대화 저장 API
@app.post("/save_conversation")
async def save_conversation(conversation: Conversation):
    # 현재 시간을 밀리초 단위로 포함하여 파일 이름 설정
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")  # 예: 20241126_153045_123456
    file_name = f"{conversation.user_id}_{timestamp}.txt"
    file_path = os.path.join(FILE_DIR, file_name)
    
    # 파일 디렉토리가 없으면 생성
    if not os.path.exists(FILE_DIR):
        os.makedirs(FILE_DIR)
    
    # 대화 내용 파일에 추가
    with open(file_path, "a", encoding="utf-8") as file:
        file.write(conversation.conversation)
    
    return {"message": "Conversation saved successfully."}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
