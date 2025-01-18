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

load_dotenv()

AGENT_NAME = os.getenv("AGENT_NAME", "smith")
PORT = int(os.getenv("PORT", "6603"))

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
                "content": f"{AGENT_NAME} echoes: {prompt}"
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
