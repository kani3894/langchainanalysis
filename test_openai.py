import openai

# Set the API key directly
openai.api_key = "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"

# Test a simple request to the OpenAI API
try:
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello, how are you?"}
        ],
        temperature=0.9
    )
    print("Response:", response.choices[0].message['content'].strip())
except Exception as e:
    print("Error:", e)