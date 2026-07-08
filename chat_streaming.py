import streamlit as st
from dotenv import load_dotenv
# from llm import get_ai_message  ## 파일 분리한곳에서 import
from llm import get_ai_response



st.set_page_config(page_title="소득세 챗봇", page_icon="🤖")

st.title("🤖 소득세 챗봇")
st.caption("소득세에 관련된 모든것을 답해드립니다.")


load_dotenv()
    
if 'message_list' not in st.session_state:
    st.session_state.message_list = []

for message in st.session_state.message_list:
    with st.chat_message(message["role"]):
        st.write(message["content"])
    
if user_question := st.chat_input(placeholder="소득세에 관련된 궁금한 내용들을 말씀해주세요!"):
    with st.chat_message("user"):
        st.write(user_question)
    st.session_state.message_list.append({"role": "user", "content": user_question})
    
    with st.spinner("답변을 생성하는 중입니다."):
        # ai_message = get_ai_message(user_question)
        ai_response = get_ai_response(user_question)
        
        with st.chat_message("ai"):
            # ai_message = st.write(ai_message)
            ai_message = st.write_stream(ai_response) ## streaming형식에서는 write_stream으로 해야 위에 대화가 유지된다.
            st.session_state.message_list.append({"role": "ai", "content": ai_message})
        # st.session_state.message_list.append({"role": "ai", "content": ai_message}) ## 스트림 아닐때는 이렇게 밖에서 출력하고 스트림일때는 append
    