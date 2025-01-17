#!/usr/bin/env python3
'''
Main module
'''
import os
import asyncio
import threading
import queue
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


PUBLISH_QUEUE = queue.Queue()


@dataclass
class MyMessage:
    ''' MyMessage '''
    content: str


@default_subscription
class MyAgent(RoutedAgent):
    ''' MyAgent '''
    def __init__(self, name: str) -> None:
        super().__init__(name)
        self._name = name

    @message_handler
    async def my_message_handler(self, message: MyMessage,
                                 ctx: MessageContext) -> None:
        print(f"Received message: {message}")
        print(f"Received message content: {message.content}")
        # print(content)
        # await self.publish_message(MyMessage(content=content),
        #                            DefaultTopicId())


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
        PUBLISH_QUEUE.put(prompt)

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
    print('sleep 5')
    await asyncio.sleep(5)
    try:
        print(f'init sender on {HOST_ADDRESS}')
        worker = GrpcWorkerAgentRuntime(host_address=HOST_ADDRESS)
        worker.start()
        await MyAgent.register(worker, "agency", lambda: MyAgent("worker"))
        print('sender registered')
    except Exception as err:
        print('sender setup error:', err)

    print('Worker is listening for messages in the queue')
    while True:
        item = PUBLISH_QUEUE.get()
        print('item:', item)
        print('worker:', worker)
        await worker.publish_message(
          MyMessage(content=item),
          DefaultTopicId()
        )
        print('published message:', item)
        PUBLISH_QUEUE.task_done()
        await asyncio.sleep(0.01)

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
    flask_thread = threading.Thread(target=run_flask_app, daemon=True)
    # flask_thread.daemon = True
    flask_thread.start()

    asyncio.run(worker_executor())

    print("Block until all tasks are done.")
    PUBLISH_QUEUE.join()
    print("exit")


if __name__ == '__main__':
    main()
