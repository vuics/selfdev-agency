#!/usr/bin/env python
'''
Selfdev Agency
'''
import os
import asyncio
import json
import re

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
AGENTS = json.loads(os.getenv("AGENTS", '''{
"adam": {"url": "http://localhost:6601/v1"},
"eve": {"url": "http://localhost:6602/v1"},
"smith": {"url": "http://localhost:6603/v1"},
"rag": {"url": "http://localhost:6604/v1"}
}'''))
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

# AI! Add a new API endpoint. Make adam.py agent self register with the agency.py through HTTP. So that the agency will have an API endpoint that allows agents to self-register. Then the agency can call each agent.
"""
AI!
Designing a self-registering mechanism for agents in a microservice architecture involves several considerations to ensure the agency can effectively track and manage these agents. Here are some guidelines and best practices:

### Self-Registration Information

When an agent self-registers with the agency, it should provide comprehensive information about its capabilities and contact details. Consider including the following data:

1. **Agent Identification**
   - Unique ID or name for the agent.
   - Version number of the agent.

2. **Endpoint Information**
   - URL and port where the agent is accessible.
   - Supported API methods (GET, POST, etc.).
   - Any specific endpoints or paths that the agent exposes.

3. **Capabilities**
   - List of services or operations the agent provides.
   - Description of available resources.
   - Supported content types (JSON, XML, etc.).

4. **Resource Utilization**
   - Current capacity or load (if applicable).
   - Maximum number of concurrent requests it can handle.

5. **Security**
   - Authentication and authorization requirements.
   - Public keys or certificates for secure communication.
   
6. **Metadata**
   - Tags or labels for categorization.
   - Optional description or documentation URLs.

### Keeping Track of Active Agents

To ensure that the agency maintains an updated list of active agents, consider implementing the following mechanisms:

1. **Heartbeat Mechanism**
   - Agents periodically send heartbeat messages to the agency to confirm their availability.
   - Specify a fixed interval for these heartbeat messages.
   - If the agency does not receive a heartbeat within a certain timeframe, it can mark the agent as inactive.

2. **Timeouts and Expiry**
   - Apply timeouts to consider an agent inactive if no communication is detected within a defined period.
   - Use a cleanup routine that regularly checks for and purges inactive agents.

### Unregistering Agents

Agents need to be able to unregister themselves from the agency in a clean and consistent manner:

1. **Unregister Endpoint**
   - Provide an API endpoint for agents to request unregistration.
   - Upon receiving the unregistration request, remove the agent from the active list and perform any necessary cleanup.

2. **Graceful Shutdown**
   - Implement a mechanism for agents to notify the agency before they shut down.
   - This could involve informing about planned maintenance or shutdowns, allowing the agency to redistribute workloads.

### Additional Considerations

1. **Retry Logic**
   - Implement retry logic in both registration and heartbeat mechanisms to handle transient network failures.

2. **Concurrent Monitoring**
   - Consider a separate monitoring service to perform health checks on each agent if more reliability is needed.

3. **Alerts and Notifications**
   - Set up notifications or alerts for when agents go offline or fail to register properly.

4. **Version Control and Compatibility**
   - Maintain compatibility with agents running different versions through proper versioning and possibly an adapter pattern to translate calls.

5. **Documentation and Discovery**
   - Use a service registry for the agency to expose the available agents and their methods for easy discovery by clients or other services.

6. **Audit and Logging**
   - Keep logs of all registration, unregistration, and heartbeat communication for auditing and troubleshooting purposes.

Designing a robust self-registration system ensures that agent services are discoverable, manageable, and maintained dynamically by the agency, leading to a more resilient microservice architecture.

---

Here's a Mermaid diagram illustrating the self-registering mechanism for agents in a microservice architecture:

```mermaid
sequenceDiagram
    participant Agent
    participant Agency
    participant Registry
    participant Monitor

    Note over Agent,Monitor: Registration Process
    Agent->>Agency: Register (Agent Info)
    Agency->>Registry: Store Agent Details
    Agency-->>Agent: Registration Confirmation
    
    Note over Agent,Monitor: Active Monitoring
    loop Heartbeat (Periodic)
        Agent->>Agency: Heartbeat Signal
        Agency->>Registry: Update Last Seen
        Monitor->>Agent: Health Check
        Monitor->>Agency: Status Update
    end

    alt Graceful Shutdown
        Agent->>Agency: Unregister Request
        Agency->>Registry: Remove Agent
        Agency-->>Agent: Unregister Confirmation
    else Failure Detection
        Monitor->>Agency: Agent Not Responding
        Agency->>Registry: Mark Agent Inactive
        Agency->>Monitor: Alert/Notification
    end

    Note over Agent,Monitor: Agent Information
    rect rgb(200, 200, 200)
        Note right of Agent: Agent ID & Version
        Note right of Agent: Endpoint Details
        Note right of Agent: Capabilities
        Note right of Agent: Resource Usage
        Note right of Agent: Security Credentials
        Note right of Agent: Metadata
    end
```

This diagram shows:
1. Initial registration process
2. Ongoing heartbeat mechanism
3. Health monitoring
4. Shutdown/failure scenarios
5. Key information maintained for each agent

The different components interact to maintain an up-to-date registry of active agents and their capabilities.
"""

@app.on_event("shutdown")
async def shutdown_event():
    await http_client.aclose()


if __name__ == "__main__":
    import uvicorn
    module = Path(__file__).stem
    print('Start module:', module)
    uvicorn.run(f"{module}:app", host="0.0.0.0", port=PORT, reload=True)
