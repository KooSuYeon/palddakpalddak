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
from typing import List

from dotenv import load_dotenv
import os
import re
import logging
import deepl

############ 로그 파일 생성 ######################
logging.basicConfig(
    filename="RAG_MODEL.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logging.info("API Server started.")
logger = logging.getLogger(__name__)
##################################################

def get_llm(api_key:str):

    llm = ChatOpenAI(model="gpt-4o", api_key=api_key)
    return llm

def load_files_to_list(file_path):
    if os.path.exists(file_path):  
        with open(file_path, 'r', encoding='utf-8') as file:  
            content = file.read() 
            return content
    else:
        print(f"File {file_path} not found.")

def save_docs_list(type_:str, length_:int, list_:list):
    file_paths = [f"dataset/{type_}_dataset_{i}.txt" for i in range(1, length_+1)]  # 파일 경로 목록 (DL_dataset_1.txt, DL_dataset_2.txt, ...)

    for j in range(len(file_paths)):
        content = load_files_to_list(file_paths[j])
        if content:  # 파일이 성공적으로 로드된 경우만 추가
            list_.append(content)

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

    current_index %= total_chunks
    # 다음 인덱스 계산
    next_index = current_index + 10
    if next_index >= total_chunks:  # 초과 시 순환 처리
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

def get_retriever_discription(texts: str, api_key=str):

    # text_list를 Document 객체로 변환
    documents = [Document(page_content=texts)]

    recursive_text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=20,
        length_function=len,
        is_separator_regex=False,
    )

    splits_recur = recursive_text_splitter.split_documents(documents)
    splits = splits_recur


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

def save_file(txt: str, file_name: str, directory: str):
    """
    주어진 디렉토리에 텍스트 파일을 저장합니다. 디렉토리가 없으면 생성합니다.
    
    :param txt: 저장할 텍스트 내용
    :param file_name: 저장할 파일 이름
    :param directory: 파일을 저장할 디렉토리 경로
    """
    # 디렉토리가 존재하지 않으면 생성
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"디렉토리가 생성되었습니다: {directory}")
    
    # 디렉토리와 파일 이름 결합
    file_path = os.path.join(directory, file_name)
    
    # 파일 저장
    with open(file_path, 'w', encoding='utf-8') as content_file:
        content_file.write(txt)
    
    print(f"=========TEXT 파일 저장 완료: {file_path}===========")



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
        return save_docs_list("DL", 16, txt_list)
    elif type_ == "ml":
        return save_docs_list("ML", 23, txt_list)
    elif type_ == "llm":
        return save_docs_list("LLM", 18, txt_list)
    elif type_ == "python":
        return save_docs_list("PYTHON", 15, txt_list)
    elif type_ == "open_source":
        return save_docs_list("OPENSOURCE", 6, txt_list)
    print(f"=========={type_}교재 불러오기 완료========")
    


concept_prompt = ChatPromptTemplate.from_messages([
    ("system", """
    당신은 AI 강사입니다.
    아래의 {context} 안에서만 반드시 **한국말**로 된 단, 하나의 질문을 생성해주세요.
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
    되도록이면 코드의 시나리오적 질문을 생성해주세요.
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
    quiz에 대해 정답을 찾을 수 있는 부분을 context에서 찾아서 한국말로 보여주세요.
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


# 세션 넘버가 바뀌어야 할 때 불러야 하는 함수
def get_session_no(id: str, file_path: str) -> int:
    """
    특정 디렉토리에서 id가 포함된 모든 txt 파일 검색 후 가장 큰 session_no 반환.

    :param id: 검색할 파일 이름의 ID 부분
    :param file_path: 검색할 디렉토리 경로
    :return: 가장 큰 session_no, 없으면 0
    """
    # file_path에 포함된 모든 txt 파일 검색
    try:
        txt_files = [f for f in os.listdir(file_path) if f.startswith(id) and f.endswith('.txt')]
    except FileNotFoundError:
        print(f"Error: 디렉토리 {file_path}를 찾을 수 없습니다.")
        return 0

    # session_no를 저장할 리스트
    session_numbers = []

    # 파일 이름에서 session_no 추출
    for file in txt_files:
        try:
            parts = file.split('_')
            if len(parts) > 1:
                session_no = int(parts[1])  # 두 번째 요소를 session_no로 해석
                session_numbers.append(session_no)
        except ValueError:
            # session_no가 숫자가 아닌 경우 무시
            continue
    # session_no 리스트가 비어 있다면 0 반환, 그렇지 않으면 최대값 반환
    return max(session_numbers) if session_numbers else 0


# 자동으로 다음 index를 반환해주는 코드 (직접 사용 X, 다른 함수내에서 자동으로 불러와짐)
def get_next_index(directory: str, category:str, id: str, session_no: int, type_: str, order: str) -> int:
    """
    특정 id, session_no, type_, order에 해당하는 quiz 파일의 다음 인덱스를 반환합니다.
    
    :param directory: 파일을 검색할 디렉토리 경로
    :param id: 검색할 id 값
    :param session_no: 세션 번호
    :param type_: 검색할 type 값
    :param order: 검색할 order 값
    :return: 다음에 사용될 quiz 인덱스 (j 값)
    """
    pattern = re.compile(rf"^{id}_{session_no}_{type_}_{order}_{category}_(\d+)\.txt$")
    max_index = 0

    # 디렉토리 내 모든 파일 탐색
    for root, _, files in os.walk(directory):
        for file in files:
            match = pattern.match(file)
            if match:
                # 파일 이름에서 인덱스를 추출
                index = int(match.group(1))
                max_index = max(max_index, index)

    # 기존 인덱스가 없으면 1, 있으면 max_index + 1 반환
    return max_index + 1




# txt파일로 저장된 이전에 저장된 내용을 불러오는 코드 (직접 사용 X, 다른 함수내에서 자동으로 불러와짐)
from typing import List
import os

def get_quiz_files_by_id_type_order_and_session(directory: str, session_no: int, id: str, type_: str, order: int) -> List[str]:
    """
    주어진 디렉토리에서 특정 id, type_, order, session_no에 관련된 모든 _quiz 텍스트 파일의 내용을 리스트로 반환합니다.
    
    :param directory: 파일을 검색할 디렉토리 경로
    :param id: 검색할 id 값
    :param type_: 검색할 type 값
    :param order: 검색할 order 값
    :param session_no: 검색할 session_no 값
    :return: id, type_, order, session_no에 관련된 _quiz 텍스트 파일 내용들의 리스트
    """
    quiz_contents = []
    
    # 디렉토리 내 모든 파일 탐색
    for root, _, files in os.walk(directory):
        for file in files:
            # 파일 이름이 id, type_, order, session_no에 관련된 형식인지 확인
            if (file.startswith(f"{id}_{session_no}_") and 
                f"_{type_}_" in file and 
                f"_{order}_" in file and 
                "_quiz" in file and 
                file.endswith(".txt")):
                file_path = os.path.join(root, file)
                # 파일 열어서 내용 읽기
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    quiz_contents.append(content)
    
    return quiz_contents


###############################################################
################ AI 퀴즈 (서버 의존성 기억) ########################
###############################################################

# RAG에서 docs를 순차적으로 읽게 하기 위한 함수 (직접 사용 X, 함수 내부에서 사용)
def get_current_index(session_no: int, id: str, type_: str, order: int, base_path: str) -> int:
    """
    주어진 조건에 맞는 파일에서 current_index를 가져오고, += 5 한 값을 반환합니다.
    만약 파일이 없다면 0을 반환합니다.
    
    Args:
        session_no (int): 세션 번호
        id (str): 사용자 ID
        type_ (str): 퀴즈 타입
        order (int): 순서
        base_path (str): 파일이 저장된 기본 디렉토리 경로

    Returns:
        int: 새로운 current_index 값
    """
    pattern = rf"{id}_{session_no}_{type_}_{order}_quiz_(\d+)\.txt"  # 파일명 패턴
    max_quiz_number = 0

    # base_path 디렉토리 내 파일들을 검색
    for filename in os.listdir(base_path):
        match = re.match(pattern, filename)
        if match:
            quiz_number = int(match.group(1))  # 파일명에서 quiz 숫자 추출
            max_quiz_number = max(max_quiz_number, quiz_number)

    # 새로운 current_index 값 계산 (quiz_number * 5)
    return max_quiz_number * 5


def get_question_language(session_no:int, id:str, type_:str,  order:int, language:str, rag_output_path:str, current_index:int):
    
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")

    current_index = get_current_index(session_no, id, type_, order, rag_output_path)
    quiz_list = get_quiz_files_by_id_type_order_and_session(rag_output_path, session_no, id, type_, order)
    # print(*quiz_list)
    txt_list = choose_txt_list(type_)
    file_number = get_next_index(rag_output_path,"quiz", id, session_no, type_, order)


    retriever = get_retriever(txt_list[order-1], current_index, api_key)
    if type_ == "open_source":
        rag_chain = create_open_source_rag_chain(retriever, get_llm(api_key))
    else:
        rag_chain = create_concept_rag_chain(retriever, get_llm(api_key))

    
    response = rag_chain.invoke("퀴즈 하나를 생성해줘")

    for _ in range(10):  # 유사하지 않은 퀴즈가 생성될 때까지 반복
        response = rag_chain.invoke("퀴즈 하나를 생성해줘")

        # 생성된 퀴즈가 유사하지 않다면 반복 종료
        if len(quiz_list) == 0 or not is_similar(response, quiz_list, 0.7):
            break
    
    logger.info(f"RAW :  {response}")
    if (type_=="open_source"):
        discription = get_discription(response, type_, order)
        question = ''.join([discription.content, "\n", str(response)])
    else:
        question = ''.join(response)

    logger.info(f"번역 전 :  {question}")
    
    if language != "KO":
        question = get_deepl_discription(question, language)

    # translation = get_translation(question, language).content
    logger.info(f"번역 후 :  {question}")
    if (id != "none"):
        save_file(''.join(question), f"{id}_{session_no}_{type_}_{order}_quiz_{file_number}.txt", rag_output_path)

    return ''.join(question)

###############################################################
######################## AI 피드백 ############################
###############################################################

def get_feedback(session_no:str, id:str, type_:str, order:int, quiz:str, user_answer:str, language:str, rag_output_path:str):

    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")

    user_file_number = get_next_index(rag_output_path,"user", id, session_no, type_, order)
    feedback_file_number = get_next_index(rag_output_path,"feedback", id, session_no, type_, order)


    feedback_prompt = ChatPromptTemplate.from_messages([
            ("system", f"""
            AI 강사로서 다음 퀴즈의 정답 여부를 확인하고 한국말로 피드백을 제공하세요.
            피드백은 아래와 같은 형식이어야 합니다:
            
            - 정답 여부: "N번" 또는 "예/아니오"
            - 추가 설명: (정답과 관련된 추가 정보를 제공하세요)
            
            퀴즈 : {{quiz}}
            답변 : {{user_answer}}
            
            """)
        ])

    feedback_chain = feedback_prompt | get_llm(api_key)
    feedback = feedback_chain.invoke({"quiz": quiz, "user_answer": user_answer}).content


    if language != "KO":
        feedback = get_deepl_discription(feedback, language)

    if (id != ""):
        save_file(''.join(user_answer), f"{id}_{session_no}_{type_}_{order}_user_{user_file_number}.txt", rag_output_path)
        save_file(feedback, f"{id}_{session_no}_{type_}_{order}_feedback_{feedback_file_number}.txt", rag_output_path)
    
    return feedback

def read_quiz_from_file(directory: str, category: str, id: str, session_no: int, type_: str, order: str) -> str:
    """
    특정 id, session_no, type_, order에 해당하는 quiz 파일을 읽어서 내용을 반환합니다.
    
    :param directory: 파일을 검색할 디렉토리 경로
    :param id: 파일을 찾을 id 값
    :param session_no: 세션 번호
    :param type_: 타입 값
    :param order: 순서 값
    :return: 파일에서 읽은 퀴즈 내용 (텍스트)
    """
    # 파일 인덱스를 자동으로 찾기
    file_number = get_next_index(directory, category, id, session_no, type_, order)

    # 파일 경로 구성
    file_name = f"{id}_{session_no}_{type_}_{order}_quiz_{file_number}.txt"
    file_path = os.path.join(directory, file_name)

    # 파일이 존재하는지 확인 후 읽기
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            quiz_content = file.read()  # 파일 내용 읽기
        return quiz_content
    else:
        raise FileNotFoundError(f"{file_path} 파일을 찾을 수 없습니다.")


def get_discription(quiz, type_, order):

    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")

    discription_chain = discription_prompt | get_llm(api_key)
    txt_list = choose_txt_list(type_)
    retriever = get_retriever_discription(txt_list[order-1], api_key)
    discription = discription_chain.invoke({"quiz": quiz, "context": retriever})

    return discription
        

def get_deepl_discription(content:str, language:str):

    load_dotenv()
    auth_key = os.getenv("DEEPL_API_KEY")
    translator = deepl.Translator(auth_key)
    result = translator.translate_text(content, target_lang=language)
    return result.text


########## 호스팅 연결 테스트용 함수 ##################

def get_question_language_test(session_no:int, id:str, type_:str,  order:int, language:str, rag_output_path:str, current_index:int):
    return "연결만 성공.. (AI 모듈 확인 필요) \n rag_model 의 get_question_language 함수 디버깅 필요"

def get_feedback_test(session_no:str, id:str, type_:str, order:int, quiz:str, user_answer:str, language:str, rag_output_path:str):
    return "연결만 성공.. (AI 모듈 확인 필요) \n rag_model 의 get_feedback 함수 디버깅 필요"