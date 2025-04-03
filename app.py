import logging
from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import JSONResponse
from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv
from PyPDF2 import PdfReader
import magic
from io import BytesIO
import csv
import re
import nltk
nltk.data.path.append('/Users/kani/nltk_data')
from nltk.tokenize.punkt import PunktSentenceTokenizer
from sentence_transformers import SentenceTransformer
import numpy as np
from transformers import GPT2Tokenizer
from typing import List

# Init
load_dotenv()
app = FastAPI()
logging.basicConfig(level=logging.INFO)

# Model & NLP tools
llm = ChatOpenAI(api_key=os.getenv('OPENAI_API_KEY'), model="gpt-4o")
embedder = SentenceTransformer("all-MiniLM-L6-v2")
tokenizer = PunktSentenceTokenizer()
TOKENIZER = GPT2Tokenizer.from_pretrained("gpt2")

def sent_tokenize(text):
    return tokenizer.tokenize(text)

# Split transcript into semantic chunks
async def split_into_windows(text, max_sentences=10, similarity_threshold=0.8):
    logging.info("Splitting text into semantic windows...")
    sentences = sent_tokenize(text)
    sentences = [s.strip() for s in sentences if s.strip()]
    logging.info(f"Sentence count: {len(sentences)}")
    sentence_embeddings = embedder.encode(sentences)

    windows = []
    current_chunk = [sentences[0]]
    current_vector = sentence_embeddings[0]

    for i in range(1, len(sentences)):
        similarity = np.dot(current_vector, sentence_embeddings[i]) / (
            np.linalg.norm(current_vector) * np.linalg.norm(sentence_embeddings[i])
        )
        if similarity >= similarity_threshold and len(current_chunk) < max_sentences:
            current_chunk.append(sentences[i])
        else:
            windows.append(" ".join(current_chunk))
            current_chunk = [sentences[i]]
            current_vector = sentence_embeddings[i]

    if current_chunk:
        windows.append(" ".join(current_chunk))
    logging.info(f"Created {len(windows)} semantic windows.")
    return windows

# Batch multiple windows
async def batch_windows(windows, batch_size=3):
    return ["\n\n".join(windows[i:i+batch_size]) for i in range(0, len(windows), batch_size)]

# Process batched windows via GPT
async def process_window(window_text):
    logging.info("Sending window batch to LLM...")
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

Transcript:
{window_text}
"""
    messages = [{"role": "user", "content": prompt}]
    response = llm.invoke(messages)
    if not response or not hasattr(response, "content"):
        logging.error("No LLM response content.")
        return []

    with open("raw_llm_output.txt", "a") as f:
        f.write("\n--- GPT RESPONSE ---\n")
        f.write(response.content + "\n")
    logging.info("Appended raw LLM output to raw_llm_output.txt")

    # Parse markdown-style GPT output
    lines = response.content.split('\n')
    chunks = []
    current_chunk = {
        "summary": "",
        "microtags": [],
        "start_line": None,
        "end_line": None,
        "chunk_text": ""
    }

    for line in lines:
        line = line.strip()
        logging.debug(f"Processing line: {line}")

        if re.match(r"^#+\s*Chunk", line, re.IGNORECASE):
            if any(current_chunk.values()):
                chunks.append(current_chunk)
                logging.info(f"Appended chunk: {current_chunk}")
            current_chunk = {"summary": "", "microtags": [], "start_line": None, "end_line": None, "chunk_text": ""}
        elif re.search(r"summary", line.lower()) and ':' in line:
            current_chunk["summary"] = line.split(":", 1)[-1].strip()
        elif re.search(r"microtags", line.lower()) and ':' in line:
            tags = line.split(":", 1)[-1].strip()
            current_chunk["microtags"] = [tag.strip() for tag in tags.split(",")]
        elif re.search(r"start.*line", line.lower()) and re.search(r"end.*line", line.lower()):
            nums = re.findall(r"\d+", line)
            if len(nums) >= 2:
                current_chunk["start_line"] = int(nums[0])
                current_chunk["end_line"] = int(nums[1])
        elif re.search(r"chunk\s*text", line.lower()):
            current_chunk["chunk_text"] = ""
        elif line.startswith("```") or not line:
            continue
        else:
            current_chunk["chunk_text"] += " " + line

    # Append the last chunk
    if any(current_chunk.values()):
        chunks.append(current_chunk)

    return chunks

# Transcript loader
async def load_transcript(file_content, is_csv=False):
    lines = []
    if is_csv:
        reader = csv.reader(file_content.splitlines())
        next(reader)
        for i, row in enumerate(reader):
            lines.append((i + 1, ' '.join(row)))
    else:
        for i, line in enumerate(file_content.splitlines()):
            lines.append((i + 1, line))
    return lines

# Full segmentation pipeline
async def segment_transcript(lines):
    text = ' '.join(line[1] for line in lines)
    windows = await split_into_windows(text)
    batched_windows = await batch_windows(windows)
    results = []
    for window in batched_windows:
        chunks = await process_window(window)
        for chunk in chunks:
            chunk["filename"] = "unknown"
        results.extend(chunks)
    return results

# CSV writer
async def save_to_csv(results, filename="output.csv"):
    if not results:
        logging.warning("No results to write.")
        return
    with open(filename, "w", newline="") as csvfile:
        fieldnames = ["Filename", "Chunk #", "Start Line", "End Line", "Summary", "Microtags", "Chunk Text"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for i, result in enumerate(results):
            writer.writerow({
                "Filename": result.get('filename', 'unknown'),
                "Chunk #": i + 1,
                "Start Line": str(result['start_line']),
                "End Line": str(result['end_line']),
                "Summary": result['summary'],
                "Microtags": ', '.join(result['microtags']),
                "Chunk Text": result['chunk_text']
            })
            logging.info(f"Wrote chunk {i + 1} to CSV: {result.get('summary', '')[:60]}...")

# Endpoint
@app.post("/analyze_file")
async def analyze_file(files: List[UploadFile] = File(...)):
    try:
        all_results = []
        for file in files:
            logging.info(f"Processing file: {file.filename}")
            content = await file.read()
            if not content:
                return JSONResponse(status_code=400, content={"error": f"{file.filename} is empty"})

            mime_type = magic.from_buffer(content, mime=True)
            if mime_type == 'application/pdf':
                reader = PdfReader(BytesIO(content))
                text = "".join(p.extract_text() for p in reader.pages if p.extract_text())
            else:
                text = content.decode('utf-8')

            lines = await load_transcript(text, is_csv=(mime_type == 'text/csv'))
            results = await segment_transcript(lines)
            for r in results:
                r['filename'] = file.filename
            all_results.extend(results)

        await save_to_csv(all_results)
        return JSONResponse(status_code=200, content={"message": "Processing complete. Saved to output.csv."})
    except Exception as e:
        logging.error("Error in processing", exc_info=True)
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.middleware("http")
async def log_requests(request: Request, call_next):
    logging.info(f"{request.method} {request.url}")
    return await call_next(request)
