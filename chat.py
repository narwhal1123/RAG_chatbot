import streamlit as st

st.set_page_config(page_title="소득세 챗봇", page_icon="🤖")
# 핫로딩을 지원해서 껐다켰다 안하고 바로바로 추가해서 확인 가능
st.title("🤖 소득세 챗봇")
st.caption("소득세에 관련된 모든것을 답해드립니다.")

# 아래 유저 채팅에서 엔터 칠때마다 페이지가 새로 로딩되어서 말풍선이 하나씩밖에 표시되지 않는다.
# 그래서 채팅 데이터 어딘가에 저장해서 채팅을 유지해야되는데 이때 session state를 이용하여 유지함
if 'message_list' not in st.session_state:
    st.session_state.message_list = []

# 실제 콘솔에 어떻게 뜨는지 확인용
# print(f"before == {st.session_state.message_list}")    
# if user_question := st.chat_input(placeholder="소득세에 관련된 궁금한 내용들을 말씀해주세요!"):
#     with st.chat_message("user"):
#         st.write(user_question)
#     st.session_state.message_list.append({"role": "user", "content": user_question})
# print(f"after == {st.session_state.message_list}")


# # 이전 채팅을 다 불러와서 표시해줌
# for message in st.session_state.message_list:
#     with st.chat_message(message["role"]):
#             st.write(message["content"])
    
# # 최근 채팅 추가
# if user_question := st.chat_input(placeholder="소득세에 관련된 궁금한 내용들을 말씀해주세요!"):
#     with st.chat_message("user"):
#         st.write(user_question)
#     st.session_state.message_list.append({"role": "user", "content": user_question})
    
#     with st.chat_message("ai"):
#         st.write("여기는 AI 메세지")
#     st.session_state.message_list.append({"role": "ai", "content": "여기는 AI 메세지"})
 
 

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_openai import ChatOpenAI
from langsmith import Client
from langchain_classic.chains import RetrievalQA
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate


load_dotenv()
embedding = OpenAIEmbeddings(model='text-embedding-3-large')   
index_name = 'tax-markdown-index'
database = PineconeVectorStore.from_existing_index(index_name=index_name,embedding=embedding ) ## 생성해둔 벡터 DB에서 읽기
llm = ChatOpenAI(model='gpt-4o')
client = Client()
prompt = client.pull_prompt("rlm/rag-prompt", dangerously_pull_public_prompt=True)
retriever = database.as_retriever(search_kwargs={'k':4})
qa_chain = RetrievalQA.from_chain_type(
    llm,
    retriever=retriever,
    chain_type_kwargs={"prompt":prompt}
)  
 
## ai메세지 만들기
def get_ai_message(user_message):
    dictionary = ["사람을 나타내는 표현 -> 거주자"]

    prompt = ChatPromptTemplate.from_template(f"""
        사용자의 질문을 보고, 우리의 사전을 참고해서 사용자의 질문을 변경해주세요.
        만약 변경할 필요가 없다고 판단된다면, 사용자의 질문을 변경하지 않아도 됩니다.
        그런 경우에는 질문만 리턴해주세요
        사전: {dictionary}
        
        질문: {{question}}
    """)

    dictionary_chain = prompt | llm | StrOutputParser()  ## 프롬프트를 llm에 한번 돌려서 질문을 다시 뽑아냄
    tax_chain = {"query": dictionary_chain} | qa_chain
    
    ai_message = tax_chain.invoke({"question": user_message})
    
    return ai_message["result"]
    
    

for message in st.session_state.message_list:
    with st.chat_message(message["role"]):
            st.write(message["content"])
    
if user_question := st.chat_input(placeholder="소득세에 관련된 궁금한 내용들을 말씀해주세요!"):
    with st.chat_message("user"):
        st.write(user_question)
    st.session_state.message_list.append({"role": "user", "content": user_question})
    
    with st.spinner("답변을 생성하는 중입니다."):
        ai_message = get_ai_message(user_question)
        
        with st.chat_message("ai"):
            st.write(ai_message)
        st.session_state.message_list.append({"role": "ai", "content": ai_message})
    