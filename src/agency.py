#!/usr/bin/env python
'''
Selfdev Agency
'''
import os
import asyncio

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
ADAM_URL = os.getenv("ADAM_URL", "http://localhost:6601/v1")
EVE_URL = os.getenv("EVE_URL", "http://localhost:6602/v1")
SMITH_URL = os.getenv("SMITH_URL", "http://localhost:6603/v1")

app = FastAPI()
http_client = httpx.AsyncClient()


class ChatRequest(BaseModel):
    prompt: str


@app.post("/v1/chat")
async def chat(request: ChatRequest):
    try:
        prompt = request.prompt
        print('prompt:', prompt)

        # TODO: send messages to agents
        #
        # at_name_regex = r'@[^ \n]*[ \n]?'
        # matches = re.findall(at_name_regex, prompt)
        # matches = [s.replace('@', '').replace(',', '').replace('\n', '').replace(' ', '').strip() for s in matches]
        # n_matches = len(matches)
        # content = prompt
        # print('at name matches:', matches, ', n_matches:', n_matches)
        # print('content:', content)

        responses = await asyncio.gather(
            http_client.post(f"{ADAM_URL}/chat", json={"prompt": prompt}),
            http_client.post(f"{EVE_URL}/chat", json={"prompt": prompt}),
            http_client.post(f"{SMITH_URL}/chat", json={"prompt": prompt}),
        )
        content = ''
        for response in responses:
            content += response.json()["content"] + "\n\n"
        # content = content.rstrip()
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


@app.on_event("shutdown")
async def shutdown_event():
    await http_client.aclose()


if __name__ == "__main__":
    import uvicorn
    module = Path(__file__).stem
    print('Start module:', module)
    uvicorn.run(f"{module}:app", host="0.0.0.0", port=PORT, reload=True)
