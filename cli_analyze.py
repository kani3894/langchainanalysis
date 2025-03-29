import requests
import sys

if len(sys.argv) != 2:
    print("Usage: python cli_analyze.py <path_to_txt_file>")
    sys.exit(1)

file_path = sys.argv[1]

# Check if the input file is being read correctly
try:
    with open(file_path, 'r') as file:
        content = file.read()
    print("File content successfully read. First 100 characters:")
    print(content[:100])  # Print the first 100 characters for verification
except Exception as e:
    print(f"Error reading file: {e}")

try:
    with open(file_path, 'rb') as f:
        files = {'file': f.read()}
        response = requests.post('http://localhost:8000/analyze', files=files)

    if response.status_code == 200:
        result = response.json()
        print("Tone:", result["tone"])
        print("Key Message:", result["key_message"])
    else:
        print("Error:", response.json().get("error", "Unknown error"))
except FileNotFoundError:
    print(f"File not found: {file_path}")
except Exception as e:
    print(f"An error occurred: {e}")
