#!/usr/bin/env python
'''
Eve Agent
'''
import os
from dotenv import load_dotenv
from fastapi.responses import JSONResponse

from base_agent import BaseAgent, ChatRequest

load_dotenv()

AGENT_NAME = os.getenv("AGENT_NAME", "eve")
PORT = int(os.getenv("PORT", "6602"))


class EveAgent(BaseAgent):
    async def chat(self, request: ChatRequest):
        try:
            prompt = request.prompt
            print('prompt:', prompt)
            raise Exception('I raised an exception because I am exceptional!')
            return JSONResponse(
                content={
                    "result": "ok",
                    "agent": self.agent_name,
                    "content": prompt.upper(),
                },
                status_code=200
            )
        except Exception as err:
            print('Chat error:', err)
            return JSONResponse(
                content={
                    "result": "error",
                    "error": str(err),
                },
                status_code=500
            )


# Create a single instance of the agent
agent = EveAgent(
    agent_name=AGENT_NAME,
    port=PORT,
)

# Export the FastAPI app instance
app = agent.get_app()

if __name__ == "__main__":
    agent.run()
