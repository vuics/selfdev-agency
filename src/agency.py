#!/usr/bin/env python
'''
Selfdev Agency
'''
import os
import asyncio
import json
import re
import pickle
from typing import Dict, Optional, Any

import aioredis
from aioredis import Redis
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import httpx
from pathlib import Path

from helpers import str_to_bool


load_dotenv()

AGENCY_NAME = os.getenv("AGENCY_NAME", "agency")
PORT = int(os.getenv("PORT", "6600"))
DEBUG = str_to_bool(os.getenv("DEBUG", 'False'))
# Dynamic agent registry
DEFAULT_AGENTS = json.loads(os.getenv("DEFAULT_AGENTS", '["smith"]'))
HTTP_TIMEOUT = int(os.getenv("HTTP_TIMEOUT", "60"))
HEARTBEAT_TIMEOUT = int(os.getenv("HEARTBEAT_TIMEOUT", "30"))  # seconds
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
REDIS_PREFIX = os.getenv("REDIS_PREFIX", "agency:")


class AgentStore:
    def __init__(self, redis: Redis):
        self.redis = redis
        self.prefix = REDIS_PREFIX

    async def set_agent(self, name: str, data: dict):
        key = f"{self.prefix}agent:{name}"
        await self.redis.set(key, pickle.dumps(data))
        # Set expiration to 2 * HEARTBEAT_TIMEOUT
        await self.redis.expire(key, 2 * HEARTBEAT_TIMEOUT)

    async def get_agent(self, name: str) -> Optional[dict]:
        key = f"{self.prefix}agent:{name}"
        data = await self.redis.get(key)
        return pickle.loads(data) if data else None

    async def delete_agent(self, name: str):
        key = f"{self.prefix}agent:{name}"
        await self.redis.delete(key)

    async def get_all_agents(self) -> Dict[str, Any]:
        agents = {}
        async for key in self.redis.scan_iter(f"{self.prefix}agent:*"):
            name = key.decode().split(':')[-1]
            data = await self.get_agent(name)
            if data:
                agents[name] = data
        return agents


app = FastAPI()
http_client = httpx.AsyncClient(timeout=HTTP_TIMEOUT)

redis: Redis = None
agent_store: AgentStore = None


# Background task for checking agent heartbeats
async def check_agent_heartbeats():
    while True:
        current_time = asyncio.get_event_loop().time()
        agents = await agent_store.get_all_agents()
        for agent_name, agent_data in agents.items():
            last_heartbeat = agent_data.get("last_heartbeat", 0)
            if current_time - last_heartbeat > HEARTBEAT_TIMEOUT:
                print(f"Agent {agent_name} timed out, unregistering")
                await agent_store.delete_agent(agent_name)
        await asyncio.sleep(HEARTBEAT_TIMEOUT // 2)

class ChatRequest(BaseModel):
    prompt: str


@app.post("/v1/chat")
async def chat(request: ChatRequest):
    try:
        prompt = request.prompt
        print('prompt:', prompt)

        mentioned_agents = re.findall(r"@(\w+)", prompt)
        print('mentioned_agents:', mentioned_agents)
        agents = await agent_store.get_all_agents()
        if 'everyone' in mentioned_agents or 'all' in mentioned_agents:
            call_agents = agents.keys()
        else:
            call_agents = set(mentioned_agents) & set(agents.keys())
            print('call_agents 1:', call_agents)
            if len(call_agents) == 0:
                call_agents = DEFAULT_AGENTS
        print('call_agents 2:', call_agents)

        print('call urls:', [f"{agents[agent_name]['url']}/chat" for agent_name in call_agents])
        responses = await asyncio.gather(
            *[http_client.post(
                f"{agents[agent_name]['url']}/chat", json={"prompt": prompt}
            ) for agent_name in call_agents]
        )
        print('responses:', responses)

        content = ''
        for response in responses:
            obj = response.json()
            print('obj:', obj)
            content += f'@{obj["agent"]}: {obj["content"]}\n\n'
        content = content.rstrip()
        return JSONResponse(
            content={
                "result": "ok",
                "content": content
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


class AgentRegistration(BaseModel):
    name: str
    url: str
    version: Optional[str] = "1.0"
    description: Optional[str] = None


@app.on_event("startup")
async def startup_event():
    """Initialize Redis connection and start the heartbeat checker"""

    # AI! Improve the connection to Redis. Handle connection to Redis exceptions. 
    # Print important debug information. I want to get this connection to Redis work without errors.
    """
AI! Currently I get this errors
self-developing-selfdev-agency-prod-1  | Traceback (most recent call last):
self-developing-selfdev-agency-prod-1  |   File "/opt/app/src/agency.py", line 12, in <module>
self-developing-selfdev-agency-prod-1  |     import aioredis
self-developing-selfdev-agency-prod-1  |   File "/usr/local/lib/python3.11/site-packages/aioredis/__init__.py", line 1, in <module>
self-developing-selfdev-agency-prod-1  |     from aioredis.client import Redis, StrictRedis
self-developing-selfdev-agency-prod-1  |   File "/usr/local/lib/python3.11/site-packages/aioredis/client.py", line 32, in <module>
self-developing-selfdev-agency-prod-1  |     from aioredis.connection import (
self-developing-selfdev-agency-prod-1  |   File "/usr/local/lib/python3.11/site-packages/aioredis/connection.py", line 33, in <module>
self-developing-selfdev-agency-prod-1  |     from .exceptions import (
self-developing-selfdev-agency-prod-1  |   File "/usr/local/lib/python3.11/site-packages/aioredis/exceptions.py", line 14, in <module>
self-developing-selfdev-agency-prod-1  |     class TimeoutError(asyncio.TimeoutError, builtins.TimeoutError, RedisError):
self-developing-selfdev-agency-prod-1  | TypeError: duplicate base class TimeoutError
    """
    global redis, agent_store
    print('Connecting to Redis at:', REDIS_URL)
    redis = await aioredis.from_url(REDIS_URL, decode_responses=False)
    print('redis:', redis)
    agent_store = AgentStore(redis)
    asyncio.create_task(check_agent_heartbeats())


@app.post("/v1/register")
async def register_agent(registration: AgentRegistration):
    """Register a new agent with the agency"""
    try:
        agent_data = {
            "url": registration.url,
            "version": registration.version,
            "description": registration.description,
            "last_heartbeat": asyncio.get_event_loop().time()
        }
        await agent_store.set_agent(registration.name, agent_data)
        print(f"Registered agent: {registration.name} at {registration.url}")
        return JSONResponse(
            content={
                "result": "ok",
                "message": f"Successfully registered agent {registration.name}"
            },
            status_code=200
        )
    except Exception as err:
        print('Registration error:', err)
        return JSONResponse(
            content={
                "result": "error",
                "error": str(err)
            },
            status_code=500
        )


@app.delete("/v1/unregister/{agent_name}")
async def unregister_agent(agent_name: str):
    """Unregister an agent from the agency"""
    try:
        if await agent_store.get_agent(agent_name):
            await agent_store.delete_agent(agent_name)
            return JSONResponse(
                content={
                    "result": "ok",
                    "message": f"Successfully unregistered agent {agent_name}"
                },
                status_code=200
            )
        return JSONResponse(
            content={
                "result": "error",
                "error": f"Agent {agent_name} not found"
            },
            status_code=404
        )
    except Exception as err:
        print('Unregistration error:', err)
        return JSONResponse(
            content={
                "result": "error",
                "error": str(err)
            },
            status_code=500
        )

@app.post("/v1/heartbeat/{agent_name}")
async def agent_heartbeat(agent_name: str):
    """Record a heartbeat from an agent"""
    agent_data = await agent_store.get_agent(agent_name)
    if not agent_data:
        return JSONResponse(
            content={
                "result": "error",
                "error": f"Agent {agent_name} not registered"
            },
            status_code=404
        )

    agent_data["last_heartbeat"] = asyncio.get_event_loop().time()
    await agent_store.set_agent(agent_name, agent_data)
    return JSONResponse(
        content={
            "result": "ok"
        },
        status_code=200
    )


@app.on_event("shutdown")
async def shutdown_event():
    await http_client.aclose()
    if redis:
        await redis.close()


if __name__ == "__main__":
    import uvicorn
    module = Path(__file__).stem
    print('Start module:', module)
    uvicorn.run(f"{module}:app", host="0.0.0.0", port=PORT, reload=True)
