import streamlit as st
import requests
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
import os

st.title("DocSift: Minimal Document Analyzer")

st.markdown("""
**Note:**
- Maximum file size for .txt is less than 50 KB (~12,000 words)
- Maximum file size for .pdf is 500 KB
""")

# Clear the raw_llm_output.txt and output.csv files on server start
open('raw_llm_output.txt', 'w').close()
open('output.csv', 'w').close()

uploaded_files = st.file_uploader("Upload .txt or .pdf files", type=["txt", "pdf"], accept_multiple_files=True)

if uploaded_files:
    if st.button("Analyze"):
        def analyze_file(uploaded_file):
            files = {"files": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
            try:
                # Estimate number of chunks
                chunk_size = 1000  # Example chunk size
                num_chunks = len(uploaded_file.getvalue()) // chunk_size + 1
                st.write(f"Estimated number of chunks for {uploaded_file.name}: {num_chunks}")

                response = requests.post("http://localhost:8000/analyze_file", files=files, timeout=(3.05, 420))
                response.raise_for_status()
                response_text = response.text.replace("\\n", "\n").replace("\n ", "\n\n")
                
                # Format the text to be more readable
                formatted_text = response_text.replace("\n\n", "\n")  # Remove double newlines
                formatted_text = formatted_text.replace("\n", "\n\n")  # Add consistent spacing
                
                st.markdown(f"### Results for {uploaded_file.name}")
                st.markdown(formatted_text, unsafe_allow_html=True)

                # Update CSV incrementally
                with open("output.csv", "a") as f:
                    f.write(response_text)
                
                try:
                    # Read CSV and format it for better display
                    df = pd.read_csv("output.csv")
                    
                    # Format the text columns to wrap and be more readable
                    st.write("### Analysis Results")
                    st.dataframe(
                        df.style.set_properties(**{
                            'white-space': 'pre-wrap',
                            'text-align': 'left',
                            'word-wrap': 'break-word',
                            'max-width': '400px'
                        }),
                        height=600  # Set a fixed height to prevent scrolling
                    )
                except Exception as e:
                    st.error(f"Failed to load CSV for {uploaded_file.name}: {e}")
            except Exception as e:
                st.error(f"Error analyzing {uploaded_file.name}: {str(e)}")

        # Use ThreadPoolExecutor to process files concurrently
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(analyze_file, uploaded_file) for uploaded_file in uploaded_files]
            for future in as_completed(futures):
                future.result()