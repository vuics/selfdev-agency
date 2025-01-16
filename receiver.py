from dataclasses import dataclass
from autogen_core import DefaultTopicId, MessageContext, RoutedAgent, default_subscription, message_handler
import asyncio
from autogen_ext.runtimes.grpc import GrpcWorkerAgentRuntime


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
    async def my_message_handler(self, message: MyMessage, ctx: MessageContext) -> None:
        self._counter += 1
        if self._counter > 5:
            return
        content = f"{self._name}: Hello x {self._counter}"
        print(content)
        await self.publish_message(MyMessage(content=content), DefaultTopicId())


async def main():
    worker1 = GrpcWorkerAgentRuntime(host_address="localhost:50051")
    worker1.start()
    await MyAgent.register(worker1, "receiver", lambda: MyAgent("worker1"))

    # worker2 = GrpcWorkerAgentRuntime(host_address="localhost:50051")
    # worker2.start()
    # await MyAgent.register(worker2, "sender2", lambda: MyAgent("worker2"))

    # await worker2.publish_message(MyMessage(content="Hello!"), DefaultTopicId())

    # Let the agents run for a while.
    # await asyncio.sleep(60)

    # await worker1.stop()
    # await worker2.stop()

    # To keep the worker running until a termination signal is received (e.g., SIGTERM).
    await worker1.stop_when_signal()


if __name__ ==  '__main__':
    asyncio.run(main())


