import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
# import faiss
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain.chains import LLMChain
from langchain.schema import Document


class ContextToPrompt:
    def __init__(self, prompt_template):
        self.prompt_template = prompt_template

    def invoke(self, inputs):
        # 문서 내용을 텍스트로 변환
        if isinstance(inputs["context"], list):
            context_text = "\n".join([doc.page_content for doc in inputs["context"]])
        else:
            context_text = str(inputs["context"])  # context를 문자열로 변환

        # 'answer'가 없으면 빈 문자열로 설정
        answer = inputs.get("answer", "")

        # 'topic'이 입력된 데이터에서 올바르게 전달되었는지 확인
        topic = inputs.get("topic", "AI")  # 'topic'이 없으면 기본값 설정
        
        # 프롬프트 템플릿에 적용
        formatted_prompt = self.prompt_template.format_messages(
            context=context_text,
            topic=topic,
            answer = answer
        )
        return formatted_prompt


class RetrieverWrapper:
    def __init__(self, retriever):
        self.retriever = retriever
        
    def invoke(self, inputs):
        if isinstance(inputs, dict):
            query = inputs.get("question", "")
        else:
            query = inputs
        # 검색 수행
        response_docs = self.retriever.get_relevant_documents(query)
        return response_docs

def read_text_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()


def run_rag_chat():
    load_dotenv()  # .env 파일에서 환경 변수 로드

    api_key = os.getenv("OPENAI_API_KEY")

    if api_key is None:
        raise ValueError("API key not found. Please set the OPENAI_API_KEY environment variable.")

    model = ChatOpenAI(model="gpt-4o-mini")
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")

    file_path = 'output_texts.txt'
    docs_text = read_text_file(file_path)

    docs = [Document(page_content=docs_text)]

    recursive_text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=40,
        length_function=len,
        is_separator_regex=False,
    )

    splits = recursive_text_splitter.split_documents(docs)


    vector_store = FAISS.from_documents(documents=splits, embedding=embeddings)

    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})

    # 프롬프트 템플릿 정의
    quiz_prompt = ChatPromptTemplate.from_messages([
         ("system", """
         너는 대규모 언어 모델(LLM) 퀴즈 챗봇이야. 제공된 컨텍스트만 사용하여 질문에 답해줘.
         너는 주어진 컨텍스트 안에서 주어진 문제에 대한 문제를 만들어서 퀴즈를 낼거야. 객관식, 주관식 문제 랜덤으로 섞어서 내줘.
         한번에 문제 하나씩만 낼거야.

         예시:

         컨텍스트: 
         대규모 언어 모델(LLM)은 자연어 처리(NLP) 작업을 수행하는 데 사용되는 모델입니다. 
         이 모델은 대규모 데이터셋으로 훈련되어 다양한 언어적 패턴과 구조를 학습합니다. 
         대표적인 LLM에는 OpenAI의 GPT-3, GPT-4, BERT, T5 등이 있습니다. 
         LLM은 문서 요약, 번역, 질문 답변, 텍스트 생성 등 다양한 응용 분야에서 사용됩니다.

         주제: LLM

         퀴즈: 다음 중 대규모 언어 모델(LLM)의 대표적인 예시는 무엇인가요?
         a) GPT-3
         b) GPT-4
         c) BERT
         d) 모두 맞다
         """),
         ("user", "컨텍스트: {context}\n\n주제: {topic}")
        ])

    answer_prompt = ChatPromptTemplate.from_messages([
         ("system", """
         너는 대규모 언어 모델(LLM) 퀴즈 챗봇이야. 
         너는 주어진 컨텍스트에 있는 문제와 답을 보고, 답이 맞는지 확인하고, 답에 대한 해설을 제공해줘.

         예시:

         컨텍스트: 

         퀴즈: 프롬프트 엔지니어링에서 user 프롬프트가 가져야 할 중요한 요소는 무엇인가요?  
          a) 명확한 요청  
          b) 애매한 표현  
          c) 긴 문장  
          d) 불필요한 정보  

        답변: c

        해설: 오답입니다. 답은 a) 명확한 요청입니다.

        프롬프트 엔지니어링에서 중요한 요소는 사용자 프롬프트가 명확하고 구체적인 요청을 포함하는 것입니다. 이는 AI 모델이 올바르게 해석하고 적절한 응답을 생성할 수 있도록 도와줍니다.

        다음은 각 선택지에 대한 설명입니다:

        a) 명확한 요청: AI에게 제공되는 요청은 명확하고 구체적이어야 합니다. 모호하거나 불명확한 요청은 AI가 정확한 답변을 생성하기 어렵게 만듭니다.

        b) 애매한 표현: 애매한 표현은 AI 모델이 요청을 제대로 이해하지 못하게 할 수 있습니다. 이는 응답의 질을 떨어뜨리고, 의도와 다른 결과를 초래할 수 있습니다.

        c) 긴 문장: 긴 문장은 꼭 필요한 정보만을 담고 있지 않을 수 있으며, 오히려 혼란을 줄 수 있습니다. 명확한 요청이 중요한 이유는 간결하고 핵심을 잘 전달하는 것이 AI가 이해하기에 더 좋기 때문입니다.

        d) 불필요한 정보: 프롬프트에 불필요한 정보가 포함되면 AI가 중요한 정보를 파악하기 어려워질 수 있습니다. 이는 응답의 정확성과 관련성을 저하시키는 요인이 됩니다.

        따라서, 명확한 요청을 통해 AI가 요구 사항을 정확히 이해하고, 적절한 응답을 생성할 수 있도록 하는 것이 프롬프트 엔지니어링에서 가장 중요한 요소입니다.

         """),
         ("user", "컨텍스트: {context}\n\n답변: {answer}")
        ])

    # RAG 체인 설정
    rag_chain_debug = {
        "context": RetrieverWrapper(retriever),
        "quiz_prompt": ContextToPrompt(quiz_prompt),
        "answer_prompt": ContextToPrompt(answer_prompt),
        "llm": model
    }

    while True:
        
        # 0. 사용자로부터 주제 입력 받기
        topic = input("주제를 입력하세요: ")

        # 1. 주제에 맞는 퀴즈 문제 검색
        query = f"{topic}에 내용을 찾아주세요."
        response_docs = rag_chain_debug["context"].invoke({"question": query})

        # 2. 퀴즈 문제를 프롬프트로 변환
        quiz_prompt_messages = rag_chain_debug["quiz_prompt"].invoke({
            "context": response_docs,
            "topic": topic,
            "answer": ""
        })

        # 3. LLM으로 문제 제공
        quiz_response = rag_chain_debug["llm"].invoke(quiz_prompt_messages)
        print("\n " + quiz_response.content)

        # 4. 사용자로부터 답 입력 받기
        user_answer = input("답을 입력하세요: ")

        # 5. 답에 대한 검토 및 해설 제공
        answer_prompt_messages = rag_chain_debug["answer_prompt"].invoke({
            "context": quiz_response,
            "topic": topic,
            "answer": user_answer
        })
        answer_response = rag_chain_debug["llm"].invoke(answer_prompt_messages)

        print("\n답변 검토 및 해설:")
        print(answer_response.content)

        # 6. 다시 주제 입력을 받기 위해 반복
        print("\n========================")
        print()



if __name__ == '__main__':
    run_rag_chat()