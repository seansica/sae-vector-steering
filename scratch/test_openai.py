import os
from pathlib import Path
from dotenv import load_dotenv
import openai

# Load the API key from the .env file
load_dotenv()

# Initialize the OpenAI API client
client = openai.OpenAI(
    # This is the default and can be omitted
    api_key=os.environ.get("OPENAI_API_KEY"),
)

# Make a request to GPT-4
chat_completion = client.chat.completions.create(
    messages=[
        {
            "role": "user",
            "content": "Write a Python function that calculates the factorial of a number.",
        }
    ],
    model="gpt-4o",
)

# Print the response
print(chat_completion.choices[0].message.content)


client.files.create(
    file=Path("input.jsonl"),
    purpose="fine-tune",
)