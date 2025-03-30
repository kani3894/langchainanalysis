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
        multiple_files = [
            ("files", (file.name, file.getvalue(), file.type)) for file in uploaded_files
        ]
        response = requests.post("http://localhost:8000/analyze_file", files=multiple_files)

        if response.status_code == 200:
            st.success("Processing complete. Results saved to output.csv.")
            try:
                df = pd.read_csv("output.csv")
                st.write("### Analysis Results")
                st.dataframe(df)
            except Exception as e:
                st.error(f"Failed to load CSV: {e}")
        else:
            st.error("Error: " + response.json().get("error", "Unknown error"))
