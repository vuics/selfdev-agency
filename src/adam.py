#!/usr/bin/env python
'''
Selfdev Agency
'''
import os
import asyncio
import socket
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
HEARTBEAT_INTERVAL = int(os.getenv("HEARTBEAT_INTERVAL", "60"))  # seconds
MAX_REGISTRATION_RETRIES = int(os.getenv("MAX_REGISTRATION_RETRIES", "5"))
INITIAL_RETRY_DELAY = int(os.getenv("INITIAL_RETRY_DELAY", "2"))

# Get hostname automatically - will be container name in Docker
HOST = socket.gethostname()
SERVICE_URL = f"http://{HOST}:{PORT}/v1"

app = FastAPI()
http_client = httpx.AsyncClient(timeout=30.0)  # 30 second timeout


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


# AI! Fix the following DeprecationWarnings:
"""
self-developing-selfdev-adam-prod-1  | /opt/app/src/adam.py:39: DeprecationWarning:
self-developing-selfdev-adam-prod-1  |         on_event is deprecated, use lifespan event handlers instead.
self-developing-selfdev-adam-prod-1  |
self-developing-selfdev-adam-prod-1  |
self-developing-selfdev-adam-prod-1  |         Read more about it in the
self-developing-selfdev-adam-prod-1  |         [FastAPI docs for Lifespan Events](https://fastapi.tiangolo.com/advanced/events/).
self-developing-selfdev-adam-prod-1  |
self-developing-selfdev-adam-prod-1  |   @app.on_event("startup")
self-developing-selfdev-adam-prod-1  | /opt/app/src/adam.py:111: DeprecationWarning:
self-developing-selfdev-adam-prod-1  |         on_event is deprecated, use lifespan event handlers instead.
self-developing-selfdev-adam-prod-1  |
self-developing-selfdev-adam-prod-1  |
self-developing-selfdev-adam-prod-1  |         Read more about it in the
self-developing-selfdev-adam-prod-1  |         [FastAPI docs for Lifespan Events](https://fastapi.tiangolo.com/advanced/events/).
self-developing-selfdev-adam-prod-1  |
self-developing-selfdev-adam-prod-1  |   @app.on_event("shutdown")

"""
@app.on_event("startup")
async def startup_event():
    """Register with the agency on startup and start heartbeats"""
    retry_delay = INITIAL_RETRY_DELAY
    response = None

    for attempt in range(MAX_REGISTRATION_RETRIES):
        try:
            print(f'Attempting to register with agency ({attempt + 1}/{MAX_REGISTRATION_RETRIES})')
            print(f'Registering with URL: {SERVICE_URL}')
            response = await http_client.post(
                f"{AGENCY_URL}/register",
                json={
                    "name": AGENT_NAME,
                    "url": SERVICE_URL,
                    "version": "1.0",
                    "description": "Adam agent for testing"
                },
                timeout=10.0
            )
            print('response.status_code:', response.status_code)

            if response.status_code == 200:
                print(f"Successfully registered with agency after {attempt + 1} attempts")
                # Start sending heartbeats after successful registration
                asyncio.create_task(send_heartbeats())
                return
            else:
                print(f"Registration attempt {attempt + 1} failed with status {response.status_code}: {response.text}")

        except httpx.ConnectError as e:
            print('response.status_code:', response.status_code)
            print(f"Connection error on attempt {attempt + 1}: {str(e)}")
        except httpx.TimeoutError as e:
            print(f"Timeout error on attempt {attempt + 1}: {str(e)}")
        except Exception as e:
            print(f"Unexpected error on attempt {attempt + 1}: {str(e)}")

        if attempt < MAX_REGISTRATION_RETRIES - 1:
            print(f"Retrying in {retry_delay} seconds...")
            await asyncio.sleep(retry_delay)
            retry_delay *= 2  # Exponential backoff

    print(f"Failed to register with agency after {MAX_REGISTRATION_RETRIES} attempts")


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
