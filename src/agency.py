#!/usr/bin/env python
'''
Selfdev Agency
'''
import os
import asyncio
import json
import re
from typing import Dict, Optional

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
AGENTS: Dict[str, dict] = {}
DEFAULT_AGENTS = json.loads(os.getenv("DEFAULT_AGENTS", '["smith"]'))
HTTP_TIMEOUT = int(os.getenv("HTTP_TIMEOUT", "60"))

app = FastAPI()
http_client = httpx.AsyncClient(timeout=HTTP_TIMEOUT)


class ChatRequest(BaseModel):
    prompt: str


@app.post("/v1/chat")
async def chat(request: ChatRequest):
    try:
        prompt = request.prompt
        print('prompt:', prompt)

        mentioned_agents = re.findall(r"@(\w+)", prompt)
        print('mentioned_agents:', mentioned_agents)
        if 'everyone' in mentioned_agents or 'all' in mentioned_agents:
            call_agents = AGENTS.keys()
        else:
            call_agents = set(mentioned_agents) & set(AGENTS.keys())
            print('call_agents 1:', call_agents)
            if len(call_agents) == 0:
                call_agents = DEFAULT_AGENTS
        print('call_agents 2:', call_agents)

        print('call urls:', [f"{AGENTS[agent_name]['url']}/chat" for agent_name in call_agents])
        responses = await asyncio.gather(
            *[http_client.post(
                f"{AGENTS[agent_name]['url']}/chat", json={"prompt": prompt}
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

@app.post("/v1/register")
async def register_agent(registration: AgentRegistration):
    """Register a new agent with the agency"""
    try:
        AGENTS[registration.name] = {
            "url": registration.url,
            "version": registration.version,
            "description": registration.description
        }
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
        if agent_name in AGENTS:
            del AGENTS[agent_name]
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

# AI! Add a route for heartbeat so that the adam.py agent will be sending this heartbeat. If heartbeat did not come during the timeout period, then the agency unregisters the agent. Modify adam.py too to send the heartbeats.
#

@app.on_event("shutdown")
async def shutdown_event():
    await http_client.aclose()


if __name__ == "__main__":
    import uvicorn
    module = Path(__file__).stem
    print('Start module:', module)
    uvicorn.run(f"{module}:app", host="0.0.0.0", port=PORT, reload=True)
