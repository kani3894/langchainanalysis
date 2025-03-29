# DocSift: Minimal Document Analyzer

DocSift is a minimal document analysis tool built with Streamlit that allows users to upload and analyze text (.txt) and PDF (.pdf) documents. The application sends the uploaded file to a FastAPI backend for analysis and displays the analysis results in a formatted code block.

## Features

- **File Upload:** Easily upload text or PDF files.
- **File Size Limits:** 
  - .txt files: less than 50 KB (~12,000 words)
  - .pdf files: maximum 500 KB
- **Analysis Display:** Results are shown in a well-formatted code block for clear viewing.
- **Backend Integration:** Sends files to a FastAPI endpoint (`/analyze_file`) for processing.
- **OpenAI Model:** Utilizes the gpt-3.5-turbo model for analysis.

## Prerequisites

- Python 3.7+
- [Streamlit](https://streamlit.io/)
- [Requests](https://pypi.org/project/requests/)
- FastAPI (for the backend API)

## Installation

1. **Clone the Repository:**

   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. **Set Up a Virtual Environment:**

   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```

3. **Install Dependencies:**

   Create a `requirements.txt` file (if not already provided) with:

   ```text
   streamlit
   requests
   fastapi
   uvicorn
   ```

   Then run:

   ```bash
   pip install -r requirements.txt
   ```

## Running the Application

### FastAPI Backend

Make sure your FastAPI backend is running and accessible. For example, if your backend is in `app.py`, you can start it with:

```bash
uvicorn app:app --reload
```

This should serve your backend at http://localhost:8000.

### Streamlit Frontend

Run the Streamlit application with:

```bash
streamlit run streamlit_app.py
```

This will open your app in your default web browser (usually at http://localhost:8501).

## Usage

1. **Upload a File:**

   In the Streamlit app, use the file uploader to select a .txt or .pdf file.

2. **Click the Analyze button.**

3. **View the Analysis:**

   The app sends a POST request to http://localhost:8000/analyze_file.

   The analysis result is returned and displayed in a formatted code block.

## Customization

- **File Size & Type Checks:**
  Adjust the file size limits or MIME type checks in `streamlit_app.py` as needed.

- **Backend URL:**
  If your FastAPI backend is running on a different host or port, update the URL in the POST request within `streamlit_app.py`.

## Contributing

Contributions and feedback are welcome! Please submit pull requests or open issues to help improve the project.

## License

This project is licensed under the MIT License.

## Acknowledgments

- Built using Streamlit for the frontend.
- Powered by FastAPI for the backend API.