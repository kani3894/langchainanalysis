import streamlit as st
import requests

st.title("DocSift: Minimal Document Analyzer")

st.markdown("""
**Note:**
- Maximum file size for .txt is less than 50 KB (~12,000 words)
- Maximum file size for .pdf is 500 KB
""")

uploaded_file = st.file_uploader("Upload a .txt or .pdf file", type=["txt", "pdf"])

if uploaded_file is not None:
    if st.button("Analyze"):
        files = {"file": uploaded_file.getvalue()}
        response = requests.post("http://localhost:8000/analyze_file", files=files)
        
        if response.status_code == 200:
            response_text = response.text
            response_text = response_text.replace("\\n", "\n").replace("\n ", "\n\n")
            st.markdown(response_text, unsafe_allow_html=True)
        else:
            st.error("Error: " + response.json().get("error", "Unknown error"))
