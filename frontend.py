import streamlit as st
import requests

# User ID for demo purposes
USER_ID = "demo_user"

st.set_page_config(page_title="Virtual Tutor Chatbot", layout="wide")

st.title("🌟 Virtual Tutor Chatbot with Multilingual Support 🌍")

# Upload Context Section
st.sidebar.header("📄 Upload Context")
uploaded_file = st.sidebar.file_uploader("Choose a PDF file", type=["pdf"])

if uploaded_file:
    with st.spinner("Uploading and processing the file..."):
        response = requests.post(
            "http://127.0.0.1:8000/upload_context/",
            params={"user_id": USER_ID},
            files={"file": uploaded_file.getvalue()},
        )
        if response.status_code == 200:
            st.sidebar.success("Context uploaded successfully!")
        else:
            st.sidebar.error("Failed to upload the file.")

# Ask a Question Section
st.header("💬 Ask a Question")
question = st.text_input("Enter your question (Supports multiple languages):")
if st.button("Get Answer"):
    if question:
        with st.spinner("Fetching the answer..."):
            response = requests.post(
                "http://127.0.0.1:8000/ask/",
                json={"question": question, "context": None},  # Context is optional
                params={"user_id": USER_ID},
            )
        if response.status_code == 200:
            data = response.json()
            st.success(f"Answer: {data['answer']}")
            st.info(f"Context Used: {data['context_used']}")
        else:
            st.error(response.json().get("detail", "Failed to fetch answer."))
    else:
        st.warning("Please enter a question!")

# Footer
st.sidebar.markdown("**Created by Virtual Tutor Team**")
