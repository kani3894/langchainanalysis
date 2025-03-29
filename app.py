import logging
from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import JSONResponse
from langchain_community.chat_models import ChatOpenAI
import os
from dotenv import load_dotenv
import traceback
from PyPDF2 import PdfReader
import magic
from io import BytesIO

load_dotenv()

# Initialize FastAPI app
app = FastAPI()

# Configure logging
logging.basicConfig(level=logging.INFO)

# Initialize the OpenAI LLM model with gpt-3.5-turbo
llm = ChatOpenAI(openai_api_key=os.getenv('OPENAI_API_KEY'), model_name="gpt-3.5-turbo")

@app.middleware("http")
async def log_requests(request: Request, call_next):
    logging.info(f"Request method: {request.method}, URL: {request.url}")
    response = await call_next(request)
    return response

# @app.post("/analyze_simpletext")
# async def analyze_simpletext():
#     try:
#         # Analyze tone and extract key message
#         logging.info("Connecting to OpenAI...")
#         logging.info("Analyzing tone...")
#         simple_text = "This is a simple test sentence."
#         logging.info("Sending request to OpenAI: %s", "Analyze the tone of the following text: " + simple_text)
#         messages = [{"role": "user", "content": "Analyze the tone of the following text: " + simple_text}]
#         tone = llm.invoke(messages)
#         logging.info("Received response from OpenAI: %s", tone)
#         logging.info("Tone analysis complete. Tone: %s", tone)

#         logging.info("Extracting key message...")
#         logging.info("Sending request to OpenAI: %s", "Summarize the key message of the following text: " + simple_text)
#         messages = [{"role": "user", "content": "Summarize the key message of the following text: " + simple_text}]
#         key_message = llm.invoke(messages)
#         logging.info("Received response from OpenAI: %s", key_message)
#         logging.info("Key message extraction complete. Key Message: %s", key_message)

#         logging.info("Analysis complete: Tone - %s, Key Message - %s", tone, key_message)
#         logging.info("Preparing response...")
#         return {"tone": tone, "key_message": key_message}
#     except Exception as e:
#         logging.error("Error during analysis: %s", e)
#         logging.error("Traceback: %s", traceback.format_exc())
#         return JSONResponse(status_code=500, content={"error": "Internal server error."})

@app.post("/analyze_file")
async def analyze_file(file: UploadFile = File(...)):
    try:
        # Read file content
        logging.info("Receiving file...")
        content = await file.read()

        # Log detected file type
        mime_type = file.content_type or magic.from_buffer(content, mime=True)
        logging.info(f"Detected file type: {mime_type}")

        # Check if the file is PDF
        if mime_type in ['application/pdf', 'application/octet-stream'] or file.filename.endswith('.pdf'):
            logging.info("Detected PDF file.")
            if len(content) > 500 * 1024:  # 500 KB
                logging.error("PDF file size exceeds limit.")
                return JSONResponse(status_code=400, content={"error": "PDF file size exceeds 500 KB limit."})
            reader = PdfReader(BytesIO(content))
            simple_text = ""
            for page in reader.pages:
                simple_text += page.extract_text()
        elif mime_type == 'text/plain' or file.filename.endswith('.txt'):
            logging.info("Detected text file.")
            if len(content) > 50 * 1024:  # 50 KB
                logging.error("Text file size exceeds limit.")
                return JSONResponse(status_code=400, content={"error": "Text file size exceeds 50 KB limit."})
            try:
                simple_text = content.decode('utf-8')
                logging.info("File decoded using UTF-8.")
            except UnicodeDecodeError:
                logging.warning("UTF-8 decoding failed, trying latin-1.")
                simple_text = content.decode('latin-1')
                logging.info("File decoded using latin-1.")
        else:
            logging.error("Unsupported file type.")
            return JSONResponse(status_code=400, content={"error": "Unsupported file type."})

        # Analyze tone and extract key message
        logging.info("Analyzing tone...")
        messages = [{"role": "user", "content": "Analyze the tone of the following text: " + simple_text}]
        tone = llm.invoke(messages)

        logging.info("Extracting key message...")
        messages = [{"role": "user", "content": "Summarize the key message of the following text: " + simple_text}]
        key_message = llm.invoke(messages)

        # Prepare structured response
        response = f"""
        ### Tone Analysis
        **Content:**
        {tone.content}

        **Model:** {tone.response_metadata['model_name']}
        **Tokens Used:** {tone.response_metadata['token_usage']['total_tokens']}

        ### Key Message Extraction
        **Content:**
        {key_message.content}

        **Model:** {key_message.response_metadata['model_name']}
        **Tokens Used:** {key_message.response_metadata['token_usage']['total_tokens']}
        """
        logging.info("Analysis complete.")
        return response
    except Exception as e:
        logging.error("Error during analysis: %s", e)
        logging.error("Traceback: %s", traceback.format_exc())
        return JSONResponse(status_code=500, content={"error": "Internal server error."})
