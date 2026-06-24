import streamlit as st
import sys
import os
sys.path.insert(0, "src")

from chain import ask

# Page Config
st.set_page_config(
    page_title = "DS Teaching Assistant",
    page_icon = "📚",
    layout = "centered"
)

# Header
st.title("📚 Data Science Teaching Assistant")
st.markdown("Ask any question from related to Data Science")
st.divider()

# Session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if query := st.chat_input("Ask a question about Data Science..."):

    # display user question
    with st.chat_message("user"):
        st.markdown(query)
    
    # save to history
    st.session_state.messages.append({
        "role" : "user",
        "content" : query
    })

    # get answer from RAG pipeline
    with st.chat_message("assistant"):
        with st.spinner("Searching textbooks..."):
            result = ask(query)

        # display answer
        st.markdown(result["answer"])

        # display sources
        if result["sources"]:
            st.divider()
            st.markdown("**📖 Sources:**")
            for source in result["sources"]:
                st.markdown(f"- {source}")


    # save assistant response to history
    st.session_state.messages.append({
        "role" : "content",
        "content" : result["answer"]
    })