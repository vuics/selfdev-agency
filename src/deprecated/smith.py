#!/usr/bin/env python
'''
Smith Agent
'''
import os

from dotenv import load_dotenv
from fastapi.responses import JSONResponse

from langchain_core.messages import HumanMessage, SystemMessage

from base_agent import BaseAgent, ChatRequest
from base_model import init_model


load_dotenv()

AGENT_NAME = os.getenv("AGENT_NAME", "smith")
MODEL_PROVIDER = os.getenv("MODEL_PROVIDER", "openai")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
SYSTEM_MESSAGE = os.getenv("SYSTEM_MESSAGE", "")


try:
  model = init_model(model_provider=MODEL_PROVIDER,
                     model_name=MODEL_NAME)
except Exception as e:
  print("Error initializing model:", e)


class SmithAgent(BaseAgent):
  async def chat(self, request: ChatRequest):
    print('request:', request)
    try:
      prompt = request.prompt
      print('prompt:', prompt)
      ai_msg = model.invoke([
        SystemMessage(SYSTEM_MESSAGE),
        HumanMessage(prompt)
      ])
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
