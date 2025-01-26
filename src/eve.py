#!/usr/bin/env python
'''
Selfdev Agency
'''
import os

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from pathlib import Path

# AI! The eve.py agent is very similar to adam.py agent. They are different in the chat function. Add the code for self registering, heartbeats to this eve agent from adam. Do not repeat yourself. If you need move the common code to a new file called agent_class.py

load_dotenv()

AGENT_NAME = os.getenv("AGENT_NAME", "eve")
PORT = int(os.getenv("PORT", "6602"))

app = FastAPI()


class ChatRequest(BaseModel):
    prompt: str


@app.post("/v1/chat")
async def chat(request: ChatRequest):
    try:
        prompt = request.prompt
        print('prompt:', prompt)
        return JSONResponse(
            content={
                "result": "ok",
                "agent": AGENT_NAME,
                "content": prompt.upper(),
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


@app.on_event("shutdown")
async def shutdown_event():
    pass


if __name__ == "__main__":
    import uvicorn
    module = Path(__file__).stem
    print('Start module:', module)
    uvicorn.run(f"{module}:app", host="0.0.0.0", port=PORT, reload=True)
#!/usr/bin/env python
'''
Eve Agent
'''
import os
from dotenv import load_dotenv
from fastapi.responses import JSONResponse
from agent_class import BaseAgent, ChatRequest

load_dotenv()

AGENT_NAME = os.getenv("AGENT_NAME", "eve")
PORT = int(os.getenv("PORT", "6602"))
AGENCY_URL = os.getenv("AGENCY_URL", "http://localhost:6600/v1")
HEARTBEAT_INTERVAL = int(os.getenv("HEARTBEAT_INTERVAL", "60"))
MAX_REGISTRATION_RETRIES = int(os.getenv("MAX_REGISTRATION_RETRIES", "5"))
INITIAL_RETRY_DELAY = int(os.getenv("INITIAL_RETRY_DELAY", "2"))

class EveAgent(BaseAgent):
    async def chat(self, request: ChatRequest):
        try:
            prompt = request.prompt
            print('prompt:', prompt)
            # Eve's specific chat implementation here
            response = f"Eve's response to: {prompt}"
            return JSONResponse(
                content={
                    "result": "ok",
                    "agent": self.agent_name,
                    "content": response,
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

if __name__ == "__main__":
    agent = EveAgent(
        agent_name=AGENT_NAME,
        port=PORT,
        agency_url=AGENCY_URL,
        heartbeat_interval=HEARTBEAT_INTERVAL,
        max_registration_retries=MAX_REGISTRATION_RETRIES,
        initial_retry_delay=INITIAL_RETRY_DELAY
    )
    agent.run()
