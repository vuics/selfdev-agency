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
AGENCY_URL = os.getenv("AGENCY_URL", "http://localhost:6600/v1")
HEARTBEAT_INTERVAL = int(os.getenv("HEARTBEAT_INTERVAL", "60"))
MAX_REGISTRATION_RETRIES = int(os.getenv("MAX_REGISTRATION_RETRIES", "5"))
INITIAL_RETRY_DELAY = int(os.getenv("INITIAL_RETRY_DELAY", "2"))

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
    agency_url=AGENCY_URL,
    heartbeat_interval=HEARTBEAT_INTERVAL,
    max_registration_retries=MAX_REGISTRATION_RETRIES,
    initial_retry_delay=INITIAL_RETRY_DELAY
)

# Export the FastAPI app instance
app = agent.get_app()

if __name__ == "__main__":
    agent.run()
