import streamlit as st
import requests
import pandas as pd

st.title("DocSift: Minimal Document Analyzer")

st.markdown("""
**Note:**
- Maximum file size for .txt is less than 50 KB (~12,000 words)
- Maximum file size for .pdf is 500 KB
""")

uploaded_files = st.file_uploader("Upload .txt or .pdf files", type=["txt", "pdf"], accept_multiple_files=True)

if uploaded_files:
    if st.button("Analyze"):
        for uploaded_file in uploaded_files:
            # Corrected the files key and format to match FastAPI's expected input
            files = {"files": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
            response = requests.post("http://localhost:8000/analyze_file", files=files)

            if response.status_code == 200:
                response_text = response.text
                response_text = response_text.replace("\\n", "\n").replace("\n ", "\n\n")
                st.markdown(f"### Results for {uploaded_file.name}")
                st.markdown(response_text, unsafe_allow_html=True)

                try:
                    df = pd.read_csv("output.csv")
                    st.write("### Analysis Results")
                    st.dataframe(df)
                except Exception as e:
                    st.error(f"Failed to load CSV for {uploaded_file.name}: {e}")
            else:
                st.error(f"Error analyzing {uploaded_file.name}: " + response.json().get("error", "Unknown error"))