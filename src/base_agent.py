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


load_dotenv()


class ChatRequest(BaseModel):
    prompt: str


class BaseAgent(ABC):
    def __init__(
        self,
        agent_name: str,
        port: int,
    ):
        # Load common environment variables in base class
        self.agency_url = os.getenv("AGENCY_URL", "http://localhost:6600/v1")
        self.heartbeat_interval = int(os.getenv("HEARTBEAT_INTERVAL", "60"))
        self.max_registration_retries = int(os.getenv("MAX_REGISTRATION_RETRIES", "5"))
        self.initial_retry_delay = int(os.getenv("INITIAL_RETRY_DELAY", "2"))

        self.agent_name = agent_name
        self.port = port

        # Get service name from environment variables, with fallbacks for different environments
        # K8s sets POD_NAME and SERVICE_NAME, Docker Compose sets HOSTNAME
        self.host = (
            os.getenv('SERVICE_NAME') or  # K8s service name
            os.getenv('POD_NAME') or      # K8s pod name
            os.getenv('HOSTNAME') or      # Docker compose service name
            f'selfdev-{agent_name}-prod'  # Local development fallback
        )

        # Use service URL from environment if provided, otherwise construct it
        self.service_url = os.getenv('SERVICE_URL') or f"http://{self.host}:{self.port}/v1"

        self.http_client = httpx.AsyncClient(timeout=30.0)
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
        retry_delay = self.initial_retry_delay

        for attempt in range(self.max_registration_retries):
            try:
                print(f'Attempting to register with agency ({attempt + 1}/{self.max_registration_retries})')
                print(f'Registering with URL: {self.service_url}')
                response = await self.http_client.post(
                    f"{self.agency_url}/register",
                    json={
                        "name": self.agent_name,
                        "url": self.service_url,
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

            if attempt < self.max_registration_retries - 1:
                print(f"Retrying in {retry_delay} seconds...")
                await asyncio.sleep(retry_delay)
                retry_delay *= 2  # Exponential backoff
            else:
                print(f"Failed to register with agency after {self.max_registration_retries} attempts")

        yield  # Server is running

        # Shutdown
        try:
            response = await self.http_client.delete(f"{self.agency_url}/unregister/{self.agent_name}")
            print(f"Unregistration response: {response.status_code}")
            await self.http_client.aclose()
        except Exception as e:
            print(f"Failed to unregister from agency: {e}")

    async def send_heartbeats(self):
        """Periodically send heartbeats to the agency"""
        while True:
            try:
                response = await self.http_client.post(f"{self.agency_url}/heartbeat/{self.agent_name}")
                if response.status_code != 200:
                    print(f"Heartbeat failed: {response.text}")
            except Exception as e:
                print(f"Failed to send heartbeat: {e}")
            await asyncio.sleep(self.heartbeat_interval)

    @abstractmethod
    async def chat(self, request: ChatRequest):
        """Chat implementation must be provided by child classes"""
        pass

    def run(self):
        """Run the agent"""
        import uvicorn
        print('Start agent:', self.agent_name)
        # AI! the folling code produce an error: the self.agent_name is set to 'ollama' but file name is 'smith.py'. When the self.agent_name was 'smith', it worked well. Now I get an error: `ERROR:    Error loading ASGI app. Attribute "app" not found in module "ollama".`
        module = Path(__file__).parent / Path(self.agent_name + ".py")
        uvicorn.run(f"{module.stem}:app", host="0.0.0.0", port=self.port, reload=True)
