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
    "openai": ["OPENAI_API_KEY"],

}
if MODEL_PROVIDER not in env_requirements:
    print(f"Unknown LLM provider: {MODEL_PROVIDER}")
for env_var in env_requirements[MODEL_PROVIDER]:
    if not os.environ.get(env_var):
        raise ValueError(f"{env_var} is not set")


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
