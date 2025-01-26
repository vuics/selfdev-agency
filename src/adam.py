#!/usr/bin/env python
'''
Selfdev Agency
'''
import os
import asyncio
import httpx
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from pathlib import Path

load_dotenv()

AGENT_NAME = os.getenv("AGENT_NAME", "adam")
PORT = int(os.getenv("PORT", "6601"))
AGENCY_URL = os.getenv("AGENCY_URL", "http://localhost:6600/v1")
HEARTBEAT_INTERVAL = int(os.getenv("HEARTBEAT_INTERVAL", "10"))  # seconds

app = FastAPI()
http_client = httpx.AsyncClient()


async def send_heartbeats():
    """Periodically send heartbeats to the agency"""
    while True:
        try:
            response = await http_client.post(f"{AGENCY_URL}/heartbeat/{AGENT_NAME}")
            if response.status_code != 200:
                print(f"Heartbeat failed: {response.text}")
        except Exception as e:
            print(f"Failed to send heartbeat: {e}")
        await asyncio.sleep(HEARTBEAT_INTERVAL)


@app.on_event("startup")
async def startup_event():
    """Register with the agency on startup and start heartbeats"""
    try:
        response = await http_client.post(
            f"{AGENCY_URL}/register",
            json={
                "name": AGENT_NAME,
                "url": f"http://localhost:{PORT}/v1",
                "version": "1.0",
                "description": "Adam agent for testing"
            }
        )
        print(f"Registration response: {response.status_code}")
        if response.status_code != 200:
            print(f"Registration failed: {response.text}")
        if response.status_code == 200:
            # Start sending heartbeats after successful registration
            asyncio.create_task(send_heartbeats())
    except Exception as e:
        print(f"Failed to register with agency: {e}")
# AI! I am getting the following log below. Fix the failing registering error. Add registering retry with exponential backoff.
"""
self-developing-selfdev-adam-prod-1  | INFO:     Started server process [12]
self-developing-selfdev-adam-prod-1  | INFO:     Waiting for application startup.
self-developing-selfdev-adam-prod-1  | Failed to register with agency: All connection attempts failed
self-developing-selfdev-adam-prod-1  | INFO:     Application startup complete.
"""


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


@app.on_event("shutdown")
async def shutdown_event():
    """Unregister from the agency on shutdown"""
    try:
        response = await http_client.delete(f"{AGENCY_URL}/unregister/{AGENT_NAME}")
        print(f"Unregistration response: {response.status_code}")
        await http_client.aclose()
    except Exception as e:
        print(f"Failed to unregister from agency: {e}")


if __name__ == "__main__":
    import uvicorn
    module = Path(__file__).stem
    print('Start module:', module)
    uvicorn.run(f"{module}:app", host="0.0.0.0", port=PORT, reload=True)
