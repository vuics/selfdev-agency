#!/usr/bin/env python3
'''
Main module
'''
import os
import sys
import asyncio
import threading
import queue
from dataclasses import dataclass

from dotenv import load_dotenv
from flask import Flask, request
from flask_cors import CORS
from autogen_core import DefaultTopicId, MessageContext, RoutedAgent, default_subscription, message_handler
from autogen_ext.runtimes.grpc import GrpcWorkerAgentRuntime
from waitress import serve
import uuid
# import requests
# import json

# from messages import AgenticMessage


def str_to_bool(val):
    """Convert a string representation of truth to true (1) or false (0).
    True values are 'y', 'yes', 't', 'true', 'on', and '1'; false values
    are 'n', 'no', 'f', 'false', 'off', and '0'.  Raises ValueError if
    'val' is anything else.
    """
    val = val.lower()
    if val in ('y', 'yes', 't', 'true', 'on', '1'):
        return 1
    elif val in ('n', 'no', 'f', 'false', 'off', '0'):
        return 0
    else:
        raise ValueError("invalid truth value %r" % (val,))


load_dotenv()

PORT = int(os.getenv("PORT", "6699"))
DEBUG = str_to_bool(os.getenv("DEBUG", 'False'))
HOST_ADDRESS = os.getenv("HOST_ADDRESS", "localhost:50051")
INIT_SLEEP = int(os.getenv("INIT_SLEEP", "5"))


PUBLISH_QUEUE = queue.Queue()
EXPECT_QUEUE = queue.Queue()


@dataclass
class AgenticMessage:
    ''' AgenticMessage '''
    content: str
    id: str
    id_replied: str


@default_subscription
class MyAgent(RoutedAgent):
    ''' MyAgent '''
    def __init__(self, name: str) -> None:
        super().__init__(name)
        self._name = name

    @message_handler
    async def my_message_handler(self, message: AgenticMessage,
                                 ctx: MessageContext) -> None:
        print(f"Put the queue received message: {message}")
        EXPECT_QUEUE.put(message)


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
        id = str(uuid.uuid4())
        PUBLISH_QUEUE.put(
          AgenticMessage(
              id=id,
              id_replied='',
              content=prompt,
          ),
        )

        res = '(None)'
        message = None
        print('Worker is expecting messages from the queue')
        while True:
            message = EXPECT_QUEUE.get()
            if id == message.id_replied:
                print('Queue received message:', message)
                res = message.content
                EXPECT_QUEUE.task_done()
                break
            elif id == message.id:
                print('Remove from the queue:', message)
                EXPECT_QUEUE.task_done()
            else:
                print('Queue skipped message:', message)
            await asyncio.sleep(0.001)

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
    if INIT_SLEEP > 0:
        print(f"Initially, sleeping for {INIT_SLEEP} seconds")
        await asyncio.sleep(INIT_SLEEP)

    try:
        print('Connecting to gRPC Agent Runtime on host:', HOST_ADDRESS)
        worker = GrpcWorkerAgentRuntime(host_address=HOST_ADDRESS)
        worker.start()
        await MyAgent.register(worker, "agency", lambda: MyAgent("worker"))
        print('Worker registered')
    except Exception as err:
        print('Worker setup error:', err)

    print('Worker is listening for messages in the queue')
    while True:
        message = PUBLISH_QUEUE.get()
        await worker.publish_message(
            message,
            DefaultTopicId()
        )
        print('Published message:', message)
        PUBLISH_QUEUE.task_done()
        await asyncio.sleep(0.01)

    await worker.stop_when_signal()
    print('Sender stopped')


def run_flask_app():
    ''' run flask app '''
    if DEBUG:
        app.run(host='0.0.0.0', port=PORT, debug=DEBUG)
        print('Started Flask Web Server on port:', PORT)
    else:
        serve(app, host='0.0.0.0', port=PORT)
        print('Started Waitress Web Server on port:', PORT)


def main():
    ''' main '''
    flask_thread = threading.Thread(target=run_flask_app, daemon=True)
    flask_thread.start()

    asyncio.run(worker_executor())

    print("Block until all queue tasks are done.")
    PUBLISH_QUEUE.join()
    EXPECT_QUEUE.join()
    print("Exited")


if __name__ == '__main__':
    main()
