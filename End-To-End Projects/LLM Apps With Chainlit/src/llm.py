from groq import Groq
from src.prompt import system_instruction

import os
from dotenv import load_dotenv

load_dotenv()

os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")


client = Groq()

messages = [
    {"role": "system", "content": system_instruction}
]

def ask_order(messages, model="llama-3.3-70b-versatile", temperature=0):
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature
    )

    return response.choices[0].message.content


