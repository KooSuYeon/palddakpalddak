from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain_core.runnables import RunnablePassthrough
from langchain.schema import Document
from langchain.schema import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from dotenv import load_dotenv
import os

import json
from pprint import pprint


def get_llm(api_key:str):

    llm = ChatOpenAI(model="gpt-4o", api_key=api_key)
    return llm

def load_files_to_list(file_path, text_list):
    if os.path.exists(file_path):  
        with open(file_path, 'r', encoding='utf-8') as file:  
            content = file.read() 
            text_list.append(content) 
    else:
        print(f"File {file_path} not found.")

def save_docs_list(type_:str, length_:int, list_:list):
    file_paths = [f"dataset/{type_}_dataset_{i}.txt" for i in range(1, length_+1)]  # 파일 경로 목록 (DL_dataset_1.txt, DL_dataset_2.txt, ...)

    for j in range(len(file_paths)):
        load_files_to_list(file_paths[j], list_)

    print(f"{type_} 관련 {len(list_)} 개의 파일 저장 완료")

    return list_

def get_retriever(texts: str, current_index:int, api_key=str):

    # text_list를 Document 객체로 변환
    documents = [Document(page_content=texts)]

    recursive_text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=20,
        length_function=len,
        is_separator_regex=False,
    )

    splits_recur = recursive_text_splitter.split_documents(documents)
    total_chunks = len(splits_recur)
    # 다음 인덱스 계산
    next_index = current_index + 10
    if next_index > total_chunks:  # 초과 시 순환 처리
        selected_splits = splits_recur[current_index:] + splits_recur[:next_index % total_chunks]
    else:
        selected_splits = splits_recur[current_index:next_index]

    splits = selected_splits


    # print("Top 10 chunks:")
    # for i, chunk in enumerate(splits[:10], 1):
    #     pprint(f"\nChunk {i}:\n{chunk.page_content}")

    # OpenAI 임베딩 모델 초기화
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002", api_key=api_key)
    vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)
    bm25_retriever = BM25Retriever.from_documents(splits)
    faiss_retriever = vectorstore.as_retriever()

    retriever = EnsembleRetriever(
                retrievers=[bm25_retriever, faiss_retriever],
                weights=[0.5, 0.5],  # 가중치 설정 (가중치의 합은 1.0)
            )

    return retriever   



def save_file(txt:str, file_name:str):

    with open(file_name, 'w', encoding='utf-8') as content_file:
        content_file.write(txt)

    print(f"TEXT 파일 저장 완료: {file_name}")


class DebugPassThrough(RunnablePassthrough):
    def invoke(self, *args, **kwargs):
        output = super().invoke(*args, **kwargs)
        # print("Debug Output:", output)
        return output
    
# Prompt 및 Chain 구성
class ContextToText(RunnablePassthrough):
    def invoke(self, inputs, config=None, **kwargs):  # config 인수 추가
        # context의 각 문서를 문자열로 결합
        context_text = " ".join([doc.page_content for doc in inputs["context"]])
        # print(f"Context output: {context_text}")
        return {"context": context_text, "quiz_list": inputs["quiz_list"]}

def choose_txt_list(type_:str):

    txt_list = []
    if type_ == "dl":
        return save_docs_list("DL", 23, txt_list)
    if type_ == "ml":
        return save_docs_list("ML", 16, txt_list)
    if type_ == "llm":
        return save_docs_list("LLM", 18, txt_list)
    if type_ == "python":
        return save_docs_list("PYTHON", 16, txt_list)
    if type_ == "open_source":
        return save_docs_list("OPENSOURCE", 7, txt_list)
    


concept_prompt = ChatPromptTemplate.from_messages([
    ("system", """
    당신은 AI 강사입니다.
    아래의 {context} 안에서만 반드시 **한국말**로 된 하나의 질문을 생성해주세요.
    (최대한 코드에 관한 시나리오적 질문이면 더 좋습니다.)
    {quiz_list}에 존재하는 질문들과는 최대한 덜 유사한 질문을 생성해주세요.
    아래의 제약 조건과 출제 방식에 맞춘 질문을 생성해주세요.
     
    제약 조건:
    1. "Context"에서 제공된 내용만 기반으로 질문을 생성하세요.
    2. AI 관련 내용이 아닌 질문은 생성하지 마세요
    3. "QuizList"에 이미 있는 질문과 유사하지 않은 새로운 질문을 생성하세요.

    출제 방식:
    - 질문은 반드시 보기가 있는 객관식(MCQ) 또는 O,X 형태로 출제하세요.
    - "Context"에 명시적으로 언급된 개념, 정의, 또는 내용을 활용하세요.
    - 질문은 반드시 질문 내용만 담겨야 합니다. 정답을 포함하지마세요.
    - 질문 내용에는 "quiz:" 나 "질문:" 같은 불필요한 수식어구는 담겨서는 안됩니다.
    
    Context:
    {context}
    
    QuizList:
    {quiz_list}

    """)
])

open_source_prompt = ChatPromptTemplate.from_messages([
    ("system", """
    당신은 AI 강사입니다.
    아래의 {context} 안에서만 반드시 **한국말**로 된, 이전에 물어봤던 문제와 유사하지 않은
    단 하나의 질문을 생성해주세요.
    
    반드시 질문은 질문 내용으로 명확히 답할 수 있는 질문이어야 합니다. 
    질문을 만들 때에는 코드와 관련된 특정 동작이나 목적에 대해 물어야 하며, 질문 안에 반드시 **코드를 포함**해야 합니다.
    에시 코드는 질문에 포함하지마세요.
    
    예시:
        코드:
            dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
     
        이 코드에서 이미지 데이터를 불러올 때 한 번에 32개의 이미지를 불러오나요? (O/X)
     
    [중요]
    아래의 금지리스트들과 유사한 질문을 절대 생성하지 마시오.
    금지리스트: 이전에 만들었던 질문, "QuizList"
    
    주관적인 변수에 대한 의견을 묻는 질문은 절대 생성하지 마세요.
    예를 들어, "타이타닉 데이터셋의 'embarked' 열을 숫자로 매핑할 때, 'S'는 어떤 숫자로 매핑되나요?", "2016년 이후의 데이터를 사용했을 때 r2 score와 mse가 더 좋은 점수를 보였다 (O, X 질문)"와 같은 질문은 피해야 합니다.
     
    개발 외의 역량을 붇는 질문은 절대 생성하지 마세요.
    예를 들어, "위험중립형 투자자는 kodex골드, tiger나스닥100, kodex 삼성그룹을 각각 몇 개 매수하는 것이 추천되나요?"와 같은 질문은 피해야 합니다.
     
    질문에서 제공된 정보로 명확히 답할 수 없는 질문만 절대 생성하지 마세요.
    예를 들어, "데이터가 증가할수록 모델의 평가지표가 안 좋아지는 이유는 무엇인가요? (O, X)", "이미지 데이터셋을 불러올 때 한 번에 몇 개의 이미지를 불러오나요? (a) 16개 (b) 32개 (c) 64개 (d) 128개"와 같은 질문은 피해야 합니다.
     
    질문의 형태는 반드시 객관식 또는 OX 질문 형태여야만 합니다. (어떤, 무엇을 묻는 질문은 생성하지 마세요.)
    예를 들어, "인스턴트 초기화에 쓰이는 생성자에 __call__ 메소드를 호출하면 어떤 값이 반환되나요? (O/X)"과 같은 질문은 피해야 합니다.
    예를 들어, 주관식과 OX 질문이 결합되거나 객관식과 OX 질문이 결합된 형태는 절대 생성하지 마세요.
     
    또한, 아래의 제약 조건과 출제 방식에 맞춘 질문을 생성해주세요.
     
    제약 조건:
    1. "Context"에서 제공된 내용만 기반으로 질문을 생성하세요.
    2. AI와 관련된 질문만 생성하세요.
    3. 질문의 형태는 객관식(MCQ) 또는 O,X 형태여야 합니다.
    4. 질문은 반드시 질문 내용만 담겨야 합니다. "quiz:" 나 "질문:" 같은 불필요한 수식어구는 붙이지 마세요.
    5. 질문에서 제공된 정보로 명확히 답할 수 있는 질문만 생성하세요.

    출제 방식:
    - 질문은 반드시 객관식(MCQ) 또는 O,X 형태로 출제합니다.
    - "Context"에 명시적으로 언급된 개념, 정의, 또는 내용을 활용하세요.
    
    Context:
    {context}
    
    QuizList:
    {quiz_list}

    """)
])

discription_prompt = ChatPromptTemplate.from_messages([
    ("system", f"""
    quiz에 대해 정답을 찾을 수 있는 부분을 context에서만 찾아서 한국말로 보여주세요.
    되도록이면 context 중에서도 코드 부분을 보여주세요.
    단, quiz와는 내용이 정확하게 일치하는 부분은 제외해주세요.
    찾을 수 없다면 아무것도 출력하지 마세요.
         
    [주의]
    코드외의 정답에 대한 직접적인 설명적인 힌트는 포함시키지 마세요.
        
    quiz : {{quiz}}
    context : {{context}}
    """)
])


def create_open_source_rag_chain(retriever, llm):
    return (
        {
            "context": retriever,
            "quiz_list": DebugPassThrough()
        }
        | DebugPassThrough()  # DebugPassThrough()가 실제로 어떤 역할을 하는지 확인
        | ContextToText()     # Text 변환을 위한 ContextToText
        | open_source_prompt              # prompt 사용
        | llm                 # LLM 호출
        | StrOutputParser()   # 출력 파서
    )

# RAG Chain 생성 함수
def create_concept_rag_chain(retriever, llm):
    return (
        {
            "context": retriever,
            "quiz_list": DebugPassThrough()
        }
        | DebugPassThrough()  # DebugPassThrough()가 실제로 어떤 역할을 하는지 확인
        | ContextToText()     # Text 변환을 위한 ContextToText
        | concept_prompt              # prompt 사용
        | llm                 # LLM 호출
        | StrOutputParser()   # 출력 파서
    )

def is_similar(new_quiz, quiz_list, threshold=0.8):
    vectorizer = TfidfVectorizer().fit_transform([new_quiz] + quiz_list)
    vectors = vectorizer.toarray()
    similarities = cosine_similarity(vectors[0:1], vectors[1:])
    return any(sim >= threshold for sim in similarities[0])


def get_question(type_:str, order:str):

    load_dotenv()
    api_key = os.getenv("OPEN_AI_KEY")

    global current_index
    global quiz_list
    print(current_index)
    print(*quiz_list)

    quiz_list = quiz_list[-5:]  # 최신 5개 퀴즈만 보관
    txt_list = choose_txt_list(type_)

    retriever = get_retriever(txt_list[order-1], current_index, api_key)
    if type_ == "open_source":
        rag_chain = create_open_source_rag_chain(retriever, get_llm(api_key))
    else:
        rag_chain = create_concept_rag_chain(retriever, get_llm(api_key))

    query = concept_prompt.format(
            context=retriever,
            quiz_list=quiz_list,       
        )
    
    response = rag_chain.invoke("퀴즈 하나를 생성해줘")

    while True:  # 유사하지 않은 퀴즈가 생성될 때까지 반복
        query = concept_prompt.format(
            context=retriever,
            quiz_list=quiz_list,
        )
        response = rag_chain.invoke("퀴즈 하나를 생성해줘")

        # 생성된 퀴즈가 유사하지 않다면 반복 종료
        if len(quiz_list) == 0 or not is_similar(response, quiz_list, 0.7):
            break

    if (type_=="open_source"):
        discription = get_discription(response, type_, order, get_llm(api_key), api_key)
        question = ''.join([discription.content, str(response)])
        return question
    
    return ''.join(response)

def get_feedback(quiz:str, user_answer:str):

    load_dotenv()
    api_key = os.getenv("OPEN_AI_KEY")

    global current_index
    print(current_index)
    print(*quiz_list)

    feedback_prompt = ChatPromptTemplate.from_messages([
            ("system", f"""
            AI 강사로서 다음 퀴즈의 정답 여부를 확인하고 피드백을 제공하세요.
            피드백은 아래와 같은 형식이어야 합니다:
            
            - 정답 여부: "N번" 또는 "예/아니오"
            - 추가 설명: (정답과 관련된 추가 정보를 제공하세요)
            
            퀴즈 : {{quiz}}
            답변 : {{user_answer}}
            
            """)
        ])

    feedback_chain = feedback_prompt | get_llm(api_key)
    feedback = feedback_chain.invoke({"quiz": quiz, "user_answer": user_answer})
    current_index += 5

    return feedback.content



def get_discription(quiz, type_, order):

    load_dotenv()
    api_key = os.getenv("OPEN_AI_KEY")

    discription_chain = discription_prompt | get_llm(api_key)
    txt_list = choose_txt_list(type_)
    retriever = get_retriever(txt_list[order-1], current_index, api_key)
    discription = discription_chain.invoke({"quiz": quiz, "context": retriever})

    return discription

# def run_rag_chatbot(type_, order, user_):

#     global j
#     global quiz_list

#     load_dotenv()
#     api_key = os.getenv("OPEN_AI_KEY")

#     question = ""

#     quiz = get_question(type_, order, api_key)
#     quiz_list.append(quiz)
#     j+=1
#     save_file(''.join(question), f"quiz_list_{j}.txt")
        
#     # 2. 사용자 답변 수집
#     user_answer = input("답변을 입력하세요 종료를 원하시면 exit을 입력해주세요.: ").strip()
    

#     if user_answer.strip().lower() == "exit":
#         print("=========== 대화를 종료합니다.")
#         return 
            
        
#     if not user_answer:
#         print("=============== 답변이 비어 있습니다. 다시 입력해주세요.")
#         return 
    
#     save_file(''.join(user_answer), f"answer_list_{j}.txt")

#     feedback = get_feedback(quiz, user_answer, api_key)
#     save_file(''.join(str(feedback.content)), f"feedback_list_{j}.txt")
    

        

if __name__ == '__main__':

    global quiz_list
    global current_index
    global j

    quiz_list = []
    current_index = 0
    j = 0
    valid_type = ["dl", "ml", "llm", "python", "open_source"]
