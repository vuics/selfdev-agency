from dataclasses import dataclass
from autogen_core import DefaultTopicId, MessageContext, RoutedAgent, default_subscription, message_handler
import asyncio
from autogen_ext.runtimes.grpc import GrpcWorkerAgentRuntime
from dotenv import load_dotenv
import os
import json

load_dotenv()
host_address = os.getenv("HOST_ADDRESS", "localhost:50051")


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
        print(f"Received message: {message}")
        print(f"Received message content: {message.content}")
        self._counter += 1
        if self._counter > 5:
            return
        content = f"{self._name}: Hello x {self._counter}"
        print(content)
        await self.publish_message(MyMessage(content=content),
                                   DefaultTopicId())


async def main():
    print('sleep 5')
    await asyncio.sleep(5)
    try:
        print('init receiver')
        worker1 = GrpcWorkerAgentRuntime(host_address=host_address)
        print('receiver connected')
        worker1.start()
        print('receiver started')
        await MyAgent.register(worker1, "receiver", lambda: MyAgent("worker1"))
        print('receiver registered')
    except Exception as err:
        print('sender setup error:', err)

    await worker1.stop_when_signal()
    print('receiver stopped')


if __name__ == '__main__':
    asyncio.run(main())
