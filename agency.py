#!/usr/bin/env python3
'''
Main module
'''
from dotenv import load_dotenv
import os
import requests
from flask import Flask, request
from flask_cors import CORS
from waitress import serve
import json
from dataclasses import dataclass
from autogen_core import DefaultTopicId, MessageContext, RoutedAgent, default_subscription, message_handler
import asyncio
from autogen_ext.runtimes.grpc import GrpcWorkerAgentRuntime


load_dotenv()
port = os.getenv("PORT", 6699)
debug = os.getenv("DEBUG", False)
host_address = os.getenv("HOST_ADDRESS", "localhost:50051")


data = {
  'worker': None
}


@dataclass
class MyMessage:
    content: str


@default_subscription
class MyAgent(RoutedAgent):
    def __init__(self, name: str) -> None:
        super().__init__("My agent")
        self._name = name
        self._counter = 0

    @message_handler
    async def my_message_handler(self, message: MyMessage,
                                 ctx: MessageContext) -> None:
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
def ask():
    ''' Ask '''
    try:
        prompt = request.args.get("prompt")
        print('data[worker]:', data['worker'])
        if (data['worker'] is not None):
            print('publish_message:', prompt)
            r = await data['worker'].publish_message(
              MyMessage(content=prompt),
              DefaultTopicId()
            )
            print('r:', r)
        # FIXME: change res to the actual response from the agent
        res = prompt
        print('res:', res)
        return res
    except Exception as err:
        print('Ask error:', err, ', type:', type(err))
        return {
            'result': 'error',
            'error': str(err),
        }


async def executor():
    print('sleep 5')
    await asyncio.sleep(5)
    try:
        print('init sender')
        worker = GrpcWorkerAgentRuntime(host_address=host_address)
        print('sender connected')
        worker.start()
        print('sender started')
        await MyAgent.register(worker, "sender", lambda: MyAgent("worker"))
        print('sender registered')
        data['worker'] = worker
    except Exception as err:
        print('sender setup error:', err)

    # TODO: Stopy by signal
    # await worker.stop()
    #
    print('sender stopped')


if __name__ == '__main__':
    # if debug:
    #     print('Using Flask Web Server')
    #     app.run(debug=debug, port=port, host='localhost')
    # else:
    #     print('Using Waitress Web Server')
    #     serve(app, host="0.0.0.0", port=port)

    print('start web server on port:', port)
    # app.run(debug=debug, port=port, host='localhost')
    serve(app, host="0.0.0.0", port=port)

    # print("start")
    # loop = asyncio.get_event_loop()
    # loop.run_until_complete(asyncio.gather(
    #     # serve(app, host="0.0.0.0", port=port),
    #     # executor(),
    # ))
    #
    print("exit")
