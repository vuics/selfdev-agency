'''
Eva agent was second agent fully copied from adam.
She is completely like an Adam agent, but different and beautiful.
She is a girl of Adam.
'''
import os
from dataclasses import dataclass
import asyncio

from autogen_core import DefaultTopicId, MessageContext, RoutedAgent, default_subscription, message_handler
from autogen_ext.runtimes.grpc import GrpcWorkerAgentRuntime
from dotenv import load_dotenv
import uuid
# import json

# from message_types import AgenticMessage

load_dotenv()

HOST_ADDRESS = os.getenv("HOST_ADDRESS", "localhost:50051")
INIT_SLEEP = int(os.getenv("INIT_SLEEP", "5"))
AGENT_NAME = os.getenv("AGENT_NAME", "eve")


@dataclass
class AgenticMessage:
    ''' AgenticMessage '''
    id: str
    id_replied: str
    content: str
    to: list[str]
    fr: str


@default_subscription
class MyAgent(RoutedAgent):
    def __init__(self, name: str) -> None:
        super().__init__(name)
        self._name = name

    @message_handler
    async def my_message_handler(self, message: AgenticMessage,
                                 ctx: MessageContext) -> None:
        if message.id_replied != '':
            return
        if message.to and AGENT_NAME not in message.to:
            print(f'Agent @{AGENT_NAME} not in message.to: {message.to}')
            return
        print(f"Received message: {message}")
        await self.publish_message(
            AgenticMessage(
                id=str(uuid.uuid4()),
                id_replied=message.id,
                to=[message.fr],
                fr=AGENT_NAME,
                content=f'👩🏻 {AGENT_NAME} 🗣️: {message.content}',
            ),
            DefaultTopicId()
        )


async def main():
    if INIT_SLEEP > 0:
        print(f"Initially, sleeping for {INIT_SLEEP} seconds")
        await asyncio.sleep(INIT_SLEEP)

    try:
        print('Connecting to gRPC Agent Runtime on host:', HOST_ADDRESS)
        worker = GrpcWorkerAgentRuntime(host_address=HOST_ADDRESS)
        worker.start()
        await MyAgent.register(worker, AGENT_NAME, lambda: MyAgent(AGENT_NAME))
        print('Worker registered')
    except Exception as err:
        print('Worker setup error:', err)

    await worker.stop_when_signal()
    print('Worker stopped')


if __name__ == '__main__':
    asyncio.run(main())
