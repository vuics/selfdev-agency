#!/usr/bin/env python3
'''
Main module
'''
import os
import asyncio
import threading
from dataclasses import dataclass

from dotenv import load_dotenv
from flask import Flask, request
from flask_cors import CORS
from autogen_core import DefaultTopicId, MessageContext, RoutedAgent, default_subscription, message_handler
from autogen_ext.runtimes.grpc import GrpcWorkerAgentRuntime
# import requests
# import json
from waitress import serve


load_dotenv()
PORT = os.getenv("PORT", 6699)
DEBUG = os.getenv("DEBUG", False)
HOST_ADDRESS = os.getenv("HOST_ADDRESS", "localhost:50051")


SHARED_WORKER = None
SHARED_LOCK = threading.Lock()


@dataclass
class MyMessage:
    ''' MyMessage '''
    content: str


@default_subscription
class MyAgent(RoutedAgent):
    ''' MyAgent '''
    def __init__(self, name: str) -> None:
        super().__init__("My agent")
        self._name = name
        self._counter = 0

    @message_handler
    async def my_message_handler(self, message: MyMessage,
                                 ctx: MessageContext) -> None:
        print(f"Received message: {message}")
        print(f"Received message content: {message.content}")
        self._counter += 1
        if self._counter > 5:
            return
        content = f"{self._name}: Hello x {self._counter}"
        print(content)
        await self.publish_message(MyMessage(content=content),
                                   DefaultTopicId())


try:
    app = Flask(__name__)
    CORS(app, supports_credentials=True)
except Exception as err:
    print('Flask setup error:', err, ', type:', type(err))


@app.route("/")
def hello_world():
    ''' About '''
    return "<p>Selfdev Dispatcher API v1</p>"


@app.route('/v1/available')
def available():
    ''' Available '''
    return {
        "name": "Selfdev Dispatcher API v1",
        "status": "available",
    }


@app.route('/v1/ask')
async def ask():
    ''' Ask '''
    try:
        prompt = request.args.get("prompt")
        with SHARED_LOCK:
            if SHARED_WORKER is not None:
                print('publish_message:', prompt)
                r = await SHARED_WORKER.publish_message(
                  MyMessage(content=prompt),
                  DefaultTopicId()
                )
                print('r:', r)
        # FIXME: change res to the actual response from the agent
        res = '(TODO)'
        print('res:', res)
        return res
    except Exception as err:
        print('Ask error:', err, ', type:', type(err))
        return {
            'result': 'error',
            'error': str(err),
        }


async def worker_executor():
    ''' worker_executor '''
    global SHARED_WORKER
    print('sleep 5')
    await asyncio.sleep(5)
    try:
        print(f'init sender on {HOST_ADDRESS}')
        worker = GrpcWorkerAgentRuntime(host_address=HOST_ADDRESS)
        print('sender connected')
        worker.start()
        print('sender started')
        await MyAgent.register(worker, "sender", lambda: MyAgent("worker"))
        print('sender registered')
        with SHARED_LOCK:
            SHARED_WORKER = worker
    except Exception as err:
        print('sender setup error:', err)

    # await worker.stop()
    await worker.stop_when_signal()
    print('sender stopped')


def run_flask_app():
    ''' run flask app '''
    if DEBUG:
        print('Using Flask Web Server')
        app.run(host='0.0.0.0', port=PORT, debug=DEBUG)
    else:
        print('Using Waitress Web Server')
        serve(app, host='0.0.0.0', port=PORT)
    print('Start web server on port:', PORT)


def main():
    ''' main '''
    flask_thread = threading.Thread(target=run_flask_app)
    flask_thread.daemon = True
    flask_thread.start()

    asyncio.run(worker_executor())
    print("exit")


if __name__ == '__main__':
    main()
