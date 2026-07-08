from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_openai import ChatOpenAI
# from langsmith import Client
# from langchain_classic.chains import RetrievalQA
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_classic.chains import (
    create_history_aware_retriever,
    create_retrieval_chain,
)
from langchain_classic.chains.combine_documents import (
    create_stuff_documents_chain,
)
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory

store = {}


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


def get_retriever():
    embedding = OpenAIEmbeddings(model='text-embedding-3-large')       
    index_name = 'tax-markdown-index'    
    database = PineconeVectorStore.from_existing_index(index_name=index_name,embedding=embedding)        
    retriever = database.as_retriever(search_kwargs={'k':4})    
    return retriever


def get_llm(model='gpt-4o'):
    llm = ChatOpenAI(model=model)
    return llm
    

def get_dictionary_chain():
    dictionary = ["사람을 나타내는 표현 -> 거주자"]

    llm = get_llm()
    prompt = ChatPromptTemplate.from_template(f"""
        사용자의 질문을 보고, 우리의 사전을 참고해서 사용자의 질문을 변경해주세요.
        만약 변경할 필요가 없다고 판단된다면, 사용자의 질문을 변경하지 않아도 됩니다.
        그런 경우에는 질문만 리턴해주세요
        사전: {dictionary}
        
        질문: {{question}}
    """)
    dictionary_chain = prompt | llm | StrOutputParser()     
    return dictionary_chain  


def get_rag_chain():
    llm = get_llm()    
    retriever = get_retriever()
    # client = Client()  
    # prompt = client.pull_prompt("rlm/rag-prompt", dangerously_pull_public_prompt=True)
    ## Chat History 유지를 위하여 prompt 변경
    ## https://reference.langchain.com/python/langchain-classic/chains/conversational_retrieval/base/ConversationalRetrievalChain
    

    # Contextualize question
    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, just "
        "reformulate it if needed and otherwise return it as is."
    )
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
)
    # Answer question
    qa_system_prompt = (
        "You are an assistant for question-answering tasks. Use "
        "the following pieces of retrieved context to answer the "
        "question. If you don't know the answer, just say that you "
        "don't know. Use three sentences maximum and keep the answer "
        "concise."
        "\n\n"
        "{context}"
    )
    
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    
    # Below we use create_stuff_documents_chain to feed all retrieved context
    # into the LLM. Note that we can also use StuffDocumentsChain and other
    # instances of BaseCombineDocumentsChain.
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    
    # Usage:
    # chat_history = []  # Collect chat history here (a sequence of messages)
    # rag_chain.invoke({"input": query, "chat_history": chat_history})
    
    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    ).pick('answer')
    #.pick('answer') ## pick안하면 참고한 문서까지 다 출력됨
    ## 공식문서에는 ['answer'] 쓰라고 되어있는데 이러면 스트리밍시 에러발생한다고 함
    
    return conversational_rag_chain
    
    # return qa_chain

## 한번에 출력되서 느리게 느껴질수 있다. 그래서 스트리밍식으로 답변이 나오게 get_ai_response로 변경
# def get_ai_message(user_message):
#     dictionary_chain = get_dictionary_chain()
#     rag_chain = get_rag_chain()
#     tax_chain = {"input": dictionary_chain} | rag_chain    
#     ai_message = tax_chain.invoke(
#         {
#             "question": user_message
#         },
#         config={
#             "configurable": {"session_id": "abc123"}
#         },
#     )    
    
#     return ai_message


def get_ai_response(user_message):
    dictionary_chain = get_dictionary_chain()
    rag_chain = get_rag_chain()
    tax_chain = {"input": dictionary_chain} | rag_chain    
    ai_response = tax_chain.stream(
        {
            "question": user_message
        },
        config={
            "configurable": {"session_id": "abc123"}
        },
    )    
    
    return ai_response