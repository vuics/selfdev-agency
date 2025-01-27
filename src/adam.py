#!/usr/bin/env python
'''
Adam Agent
'''
import os
from dotenv import load_dotenv
from fastapi.responses import JSONResponse

from base_agent import BaseAgent, ChatRequest

load_dotenv()

AGENT_NAME = os.getenv("AGENT_NAME", "adam")
PORT = int(os.getenv("PORT", "6601"))


class AdamAgent(BaseAgent):
    async def chat(self, request: ChatRequest):
        try:
            prompt = request.prompt
            print('prompt:', prompt)
            return JSONResponse(
                content={
                    "result": "ok",
                    "agent": self.agent_name,
                    "content": prompt,
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
agent = AdamAgent(
    agent_name=AGENT_NAME,
    port=PORT,
)

# Export the FastAPI app instance
app = agent.get_app()

if __name__ == "__main__":
    agent.run()
