#!/usr/bin/env python
'''
Smith Agent
'''
import os

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from pathlib import Path
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_ollama import ChatOllama

from base_agent import BaseAgent, ChatRequest


load_dotenv()

AGENT_NAME = os.getenv("AGENT_NAME", "smith")
PORT = int(os.getenv("PORT", "6603"))

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

# AI! instead of OLLAMA_HOST and OLLAMA_PORT use OLLAMA_BASE_URL
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "localhost")
OLLAMA_PORT = int(os.getenv("OLLAMA_PORT", "11434"))

# providers:
#  'openai', 'anthropic', 'ollama'
# models:
#  'gpt-4o', 'gpt-4o-mini', 'claude-3-5-sonnet-20240620',
#  'medragondot/Sky-T1-32B-Preview',
#  'llama3.3', 'gemma', 'mistral', 'tinyllama'
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")

if LLM_PROVIDER == "openai":
    chat_llm = ChatOpenAI(model=MODEL_NAME, api_key=OPENAI_API_KEY)
elif LLM_PROVIDER == "anthropic":
    chat_llm = ChatAnthropic(model=MODEL_NAME, api_key=ANTHROPIC_API_KEY)
elif LLM_PROVIDER == "ollama":
    chat_llm = ChatOllama(
        model=MODEL_NAME,
        base_url=f"http://{OLLAMA_HOST}:{OLLAMA_PORT}"
    )
else:
    raise ValueError(f"Unknown LLM provider: {LLM_PROVIDER}")


class SmithAgent(BaseAgent):
    async def chat(self, request: ChatRequest):
        try:
            prompt = request.prompt
            print('prompt:', prompt)
            ai_msg = chat_llm.invoke(prompt)
            print("ai_msg:", ai_msg.content)
            return JSONResponse(
                content={
                    "result": "ok",
                    "agent": AGENT_NAME,
                    "content": ai_msg.content,
                },
                status_code=200
            )
        except Exception as err:
            print('Chat error:', err)
            return JSONResponse(
                content={
                    "result": "error",
                    'error': str(err),
                },
                status_code=500
            )


# Create a single instance of the agent
agent = SmithAgent(
    agent_name=AGENT_NAME,
    port=PORT,
)

# Export the FastAPI app instance
app = agent.get_app()

if __name__ == "__main__":
    agent.run()
