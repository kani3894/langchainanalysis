import streamlit as st
import requests
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

st.title("DocSift: Minimal Document Analyzer")

st.markdown("""
**Note:**
- Maximum file size for .txt is less than 50 KB (~12,000 words)
- Maximum file size for .pdf is 500 KB
""")

uploaded_files = st.file_uploader("Upload .txt or .pdf files", type=["txt", "pdf"], accept_multiple_files=True)

if uploaded_files:
    if st.button("Analyze"):
        def analyze_file(uploaded_file):
            files = {"files": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
            try:
                response = requests.post("http://localhost:8000/analyze_file", files=files, timeout=(3.05, 60))
                response.raise_for_status()
                response_text = response.text.replace("\\n", "\n").replace("\n ", "\n\n")
                return uploaded_file.name, response_text, None
            except requests.exceptions.RequestException as e:
                return uploaded_file.name, None, str(e)

        with ThreadPoolExecutor() as executor:
            futures = {executor.submit(analyze_file, file): file for file in uploaded_files}
            for future in as_completed(futures):
                file_name, response_text, error = future.result()
                if error:
                    st.error(f"Error analyzing {file_name}: {error}")
                else:
                    st.markdown(f"### Results for {file_name}")
                    st.markdown(response_text, unsafe_allow_html=True)
                    try:
                        df = pd.read_csv("output.csv")
                        st.write("### Analysis Results")
                        st.dataframe(df)
                    except Exception as e:
                        st.error(f"Failed to load CSV for {file_name}: {e}")