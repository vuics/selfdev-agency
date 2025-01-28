#!/usr/bin/env python
import os
import asyncio
import httpx

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from contextlib import asynccontextmanager
from abc import ABC, abstractmethod
from pathlib import Path
import urllib.parse


load_dotenv()

AGENCY_URL = os.getenv("AGENCY_URL", "http://localhost:6600/v1")
AGENT_URL = os.getenv("AGENT_URL", "http://localhost:6601/v1")
HEARTBEAT_INTERVAL = int(os.getenv("HEARTBEAT_INTERVAL", "60"))
MAX_REGISTRATION_RETRIES = int(os.getenv("MAX_REGISTRATION_RETRIES", "5"))
INITIAL_RETRY_DELAY = int(os.getenv("INITIAL_RETRY_DELAY", "2"))
HTTP_CLIENT_TIMEOUT = int(os.getenv("HTTP_CLIENT_TIMEOUT", "300"))


class ChatRequest(BaseModel):
    prompt: str


class BaseAgent(ABC):
    def __init__(
        self,
        agent_name: str,
    ):
        self.agent_name = agent_name
        self.http_client = httpx.AsyncClient(timeout=HTTP_CLIENT_TIMEOUT)
        self.app = FastAPI(lifespan=self.lifespan)
        self.setup_routes()

    def setup_routes(self):
        """Setup the FastAPI routes"""
        self.app.post("/v1/chat")(self.chat)

    def get_app(self):
        """Return the FastAPI app instance"""
        return self.app

    @asynccontextmanager
    async def lifespan(self, app: FastAPI):
        """Lifespan context manager for startup/shutdown events"""
        # Startup
        retry_delay = INITIAL_RETRY_DELAY

        for attempt in range(MAX_REGISTRATION_RETRIES):
            try:
                print(f'Attempting to register with agency ({attempt + 1}/{MAX_REGISTRATION_RETRIES})')
                print(f'Registering with URL: {AGENT_URL}')
                response = await self.http_client.post(
                    f"{AGENCY_URL}/register",
                    json={
                        "name": self.agent_name,
                        "url": AGENT_URL,
                        "version": "1.0",
                        "description": f"{self.agent_name} agent"
                    },
                    timeout=10.0
                )
                print('response.status_code:', response.status_code)

                if response.status_code == 200:
                    print(f"Successfully registered with agency after {attempt + 1} attempts")
                    # Start sending heartbeats after successful registration
                    asyncio.create_task(self.send_heartbeats())
                    break
                else:
                    print(f"Registration attempt {attempt + 1} failed with status {response.status_code}: {response.text}")

            except httpx.ConnectError as e:
                print(f"Connection error on attempt {attempt + 1}: {str(e)}")
            except httpx.TimeoutError as e:
                print(f"Timeout error on attempt {attempt + 1}: {str(e)}")
            except Exception as e:
                print(f"Unexpected error on attempt {attempt + 1}: {str(e)}")

            if attempt < MAX_REGISTRATION_RETRIES - 1:
                print(f"Retrying in {retry_delay} seconds...")
                await asyncio.sleep(retry_delay)
                retry_delay *= 2  # Exponential backoff
            else:
                print(f"Failed to register with agency after {MAX_REGISTRATION_RETRIES} attempts")

        yield  # Server is running

        # Shutdown
        try:
            response = await self.http_client.delete(f"{AGENCY_URL}/unregister/{self.agent_name}")
            print(f"Unregistration response: {response.status_code}")
            await self.http_client.aclose()
        except Exception as e:
            print(f"Failed to unregister from agency: {e}")

    async def send_heartbeats(self):
        """Periodically send heartbeats to the agency"""
        while True:
            try:
                response = await self.http_client.post(f"{AGENCY_URL}/heartbeat/{self.agent_name}")
                if response.status_code != 200:
                    print(f"Heartbeat failed: {response.text}")
            except Exception as e:
                print(f"Failed to send heartbeat: {e}")
            await asyncio.sleep(HEARTBEAT_INTERVAL)

    @abstractmethod
    async def chat(self, request: ChatRequest):
        """Chat implementation must be provided by child classes"""
        pass

    def run(self):
        """Run the agent"""
        import uvicorn
        print(f'Start agent: {self.agent_name}')
        host = "0.0.0.0"

# AI! I am getting those errors:
# self-developing-selfdev-smith-dev-1  |     parsed = urllib.parse(AGENT_URL)
# self-developing-selfdev-smith-dev-1  |              ^^^^^^^^^^^^^^^^^^^^^^^
# self-developing-selfdev-smith-dev-1  | TypeError: 'module' object is not callable
        parsed = urllib.parse(AGENT_URL)
        print(f'Agent URL: {AGENT_URL} => Listen on {host}:{parsed.port}')
        uvicorn.run(self.app, host=host, port=parsed.port)
