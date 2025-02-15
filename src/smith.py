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
# from langchain_openai import ChatOpenAI
# from langchain_anthropic import ChatAnthropic
# from langchain_ollama import ChatOllama
from langchain.chat_models import init_chat_model

from base_agent import BaseAgent, ChatRequest


load_dotenv()

AGENT_NAME = os.getenv("AGENT_NAME", "smith")


# providers:
#  'openai', 'anthropic', 'ollama'
# models:
#  'gpt-4o', 'gpt-4o-mini', 'claude-3-5-sonnet-20240620',
#  'medragondot/Sky-T1-32B-Preview',
#  'llama3.3', 'gemma', 'mistral', 'tinyllama'
MODEL_PROVIDER = os.getenv("MODEL_PROVIDER", "openai")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")


model = None

env_requirements = {
    "groq": ["GROQ_API_KEY"],
    "openai": ["OPENAI_API_KEY", "GROQ_API_KEY_1"]
}
# AI! For the env_requirements dict object above, create an algorithm that check all the required environment variables with keys to replace the code block below. Replace it with the loop that select dict key as env_requirements[MODEL_PROVIDER] and then iterates through the environment variables to check if they are set and if not raises ValueError as in the code block below: 
if MODEL_PROVIDER == 'groq':
    if not os.environ.get("GROQ_API_KEY"):
        raise ValueError("GROQ_API_KEY is not set")
if MODEL_PROVIDER == 'openai':
    if not os.environ.get("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY is not set")
    if not os.environ.get("OPENAI_API_KEY_1"):
        raise ValueError("OPENAI_API_KEY_1 is not set")
else:
    print(f"Unknown LLM provider: {MODEL_PROVIDER}")


model = init_chat_model(MODEL_NAME, model_provider=MODEL_PROVIDER)



OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")


# model = None
# if LLM_PROVIDER == "openai":
#     model = ChatOpenAI(model=MODEL_NAME, api_key=OPENAI_API_KEY)
# elif LLM_PROVIDER == "anthropic":
#     model = ChatAnthropic(model=MODEL_NAME, api_key=ANTHROPIC_API_KEY)
# elif LLM_PROVIDER == "ollama":
#     model = ChatOllama(model=MODEL_NAME, base_url=OLLAMA_BASE_URL)
# else:
#     raise ValueError(f"Unknown LLM provider: {LLM_PROVIDER}")


class SmithAgent(BaseAgent):
    async def chat(self, request: ChatRequest):
        try:
            prompt = request.prompt
            print('prompt:', prompt)
            ai_msg = model.invoke(prompt)
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
agent = SmithAgent(agent_name=AGENT_NAME)

# Export the FastAPI app instance
app = agent.get_app()

if __name__ == "__main__":
    agent.run()
