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
import pandas as pd
from transformers import GPT2Tokenizer
import csv
import json
from typing import List

load_dotenv()

# Initialize FastAPI app
app = FastAPI()

# Configure logging
logging.basicConfig(level=logging.INFO)

# Initialize the OpenAI LLM model with gpt-3.5-turbo
llm = ChatOpenAI(openai_api_key=os.getenv('OPENAI_API_KEY'), model_name="gpt-3.5-turbo")

# Initialize tokenizer
TOKENIZER = GPT2Tokenizer.from_pretrained("gpt2")

# Function to split transcript into windows
async def split_into_windows(text, max_tokens=1500, overlap=200):
    tokens = TOKENIZER.encode(text)
    windows = []
    for i in range(0, len(tokens), max_tokens - overlap):
        window = tokens[i:i + max_tokens]
        windows.append(TOKENIZER.decode(window))
    return windows

# Function to process each window
async def process_window(window_text):
    prompt = f"""
    You are a transcript analyst. Your task is to split a transcript excerpt into topic-based chunks.

    You MUST:
    - Only use the provided transcript.
    - Do NOT infer or imagine any meaning not directly supported by the words.
    - Every chunk must include actual text from the transcript.
    - Base summaries and microtags only on what is said — do not guess.

    For each chunk, return:
    - A 1–2 sentence summary (based ONLY on what is in the chunk).
    - 3–6 microtags (short tags using exact words or very close synonyms).
    - Start and end line numbers (from the input).
    - The actual chunk text (copied from transcript).

    Use only the given content. Avoid hallucination. Stay grounded in what is explicitly said.

    Transcript:
    {window_text}
    """
    messages = [{"role": "user", "content": prompt}]
    response = llm.invoke(messages)
    # Log the response content
    logging.info(f"Raw LLM response:\n{response.content}")
    # Parse the response content manually
    lines = response.content.split('\n')
    chunks = []
    chunk = {}
    for line in lines:
        line = line.strip()
        logging.info(f"Processing line: {line}")
        if line.startswith("Chunk"):
            # If we already have a chunk collected, append it before starting new one
            if all(k in chunk for k in ('summary', 'microtags', 'start_line', 'end_line', 'chunk_text')):
                chunks.append(chunk)
                logging.info(f"Appended chunk: {chunk}")
            chunk = {}
        elif line.startswith("Summary:"):
            chunk['summary'] = line.replace("Summary:", "").strip()
        elif line.startswith("Microtags:"):
            chunk['microtags'] = line.replace("Microtags:", "").strip().split(', ')
        elif line.lower().startswith("start line"):
            value = line.split(":", 1)[-1].strip()
            chunk['start_line'] = value
            logging.info(f"Accepted start line as: {value}")
        elif line.lower().startswith("end line"):
            value = line.split(":", 1)[-1].strip()
            chunk['end_line'] = value
            logging.info(f"Accepted end line as: {value}")
        elif line.startswith("Text:"):
            chunk['chunk_text'] = line.replace("Text:", "").strip()
    # After loop ends, append the last chunk if complete
    if all(k in chunk for k in ('summary', 'microtags', 'start_line', 'end_line', 'chunk_text')):
        chunks.append(chunk)
        logging.info(f"Appended final chunk: {chunk}")
    return chunks

# Function to load and preprocess transcript
async def load_transcript(file_content, is_csv=False):
    lines = []
    if is_csv:
        reader = csv.reader(file_content.splitlines())
        next(reader)  # Skip header
        for i, row in enumerate(reader):
            lines.append((i + 1, ' '.join(row)))
    else:
        for i, line in enumerate(file_content.splitlines()):
            lines.append((i + 1, line))
    return lines

# Function to segment transcript
async def segment_transcript(lines):
    text = ' '.join(line[1] for line in lines)
    windows = await split_into_windows(text)
    results = []
    for window in windows:
        result = await process_window(window)
        for chunk in result:
            chunk["filename"] = "unknown"  # Temporary placeholder, will be overwritten later
        if result:  # Ensure result is not empty
            logging.info(f"Appending result: {result}")
            results.extend(result)  # Append each chunk to results
    logging.info(f"Final results: {results}")
    return results

# Function to save results to CSV
async def save_to_csv(results, filename="output.csv"):
    # Print results for debugging
    logging.info(f"Results: {results}")
    if not results:
        logging.error("No results to save.")
        return
    with open(filename, "w", newline="") as csvfile:
        fieldnames = ["Filename", "Chunk #", "Start Line", "End Line", "Summary", "Microtags", "Chunk Text"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for i, result in enumerate(results):
            logging.info(f"Writing row: {result}")
            writer.writerow({
                "Filename": result.get('filename', 'unknown'),
                "Chunk #": i + 1,
                "Start Line": str(result['start_line']),
                "End Line": str(result['end_line']),
                "Summary": result['summary'],
                "Microtags": ', '.join(result['microtags']),
                "Chunk Text": result['chunk_text']
            })

# Modify analyze_file to include new logic
@app.post("/analyze_file")
async def analyze_file(files: List[UploadFile] = File(...)):
    try:
        logging.info("Starting file analysis...")
        all_results = []
        if not files:
            logging.error("No files uploaded.")
            return JSONResponse(status_code=400, content={"error": "No files uploaded."})
        for file in files:
            logging.info(f"Receiving file: {file.filename}")
            content = await file.read()
            if not content:
                logging.error(f"File {file.filename} is empty.")
                return JSONResponse(status_code=400, content={"error": f"File {file.filename} is empty."})

            # Use python-magic for mime detection if content_type is missing
            if not file.content_type or file.content_type == 'application/octet-stream':
                mime_type = magic.from_buffer(content, mime=True)
            else:
                mime_type = file.content_type

            logging.info(f"Detected file type: {mime_type} for file: {file.filename}")

            # Handle different file types
            if mime_type == 'application/pdf' or file.filename.endswith('.pdf'):
                logging.info("Detected PDF file.")
                reader = PdfReader(BytesIO(content))
                text = "".join(page.extract_text() for page in reader.pages if page.extract_text())
            elif mime_type.startswith('text/') or file.filename.endswith(('.txt', '.csv')):
                logging.info("Detected text or CSV file.")
                text = content.decode('utf-8')
            else:
                logging.error(f"Unsupported file type: {mime_type}")
                return JSONResponse(status_code=400, content={"error": f"Unsupported file type: {mime_type}"})

            # Preprocess transcript
            logging.info("Preprocessing transcript...")
            lines = await load_transcript(text, is_csv=(mime_type == 'text/csv'))

            # Segment transcript
            logging.info("Segmenting transcript...")
            results = await segment_transcript(lines)
            for result in results:
                result['filename'] = file.filename
            all_results.extend(results)

        # Save results to CSV
        logging.info("Saving results to CSV...")
        await save_to_csv(all_results)

        logging.info("Processing complete. Results saved to output.csv.")
        return JSONResponse(status_code=200, content={"message": "Processing complete. Results saved to output.csv."})
    except Exception as e:
        logging.error(f"Error during analysis: {str(e)}")
        logging.error("Traceback:", exc_info=True)
        return JSONResponse(status_code=500, content={"error": "Internal server error."})

@app.middleware("http")
async def log_requests(request: Request, call_next):
    logging.info(f"Request method: {request.method}, URL: {request.url}")
    response = await call_next(request)
    return response
