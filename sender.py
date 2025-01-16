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
    # print('sleep 5')
    # await asyncio.sleep(5)
    try:
      print('init sender')
      # worker2 = GrpcWorkerAgentRuntime(host_address="localhost:50051")
      worker2 = GrpcWorkerAgentRuntime(host_address="selfdev-agency-prod:50051")
      print('sender connected')
      worker2.start()
      print('sender started')
      await MyAgent.register(worker2, "sender", lambda: MyAgent("worker2"))
      print('sender registered')
    except Exception as err:
      print('sender setup error:', err)

    await worker2.publish_message(MyMessage(content="Hello1!"), DefaultTopicId())
    print('message published 1')
    await asyncio.sleep(1)
    await worker2.publish_message(MyMessage(content="Hello2!"), DefaultTopicId())
    print('message published 2')
    await asyncio.sleep(2)
    await worker2.publish_message(MyMessage(content="Hello3!"), DefaultTopicId())
    print('message published 3')
    await asyncio.sleep(4)
    await worker2.publish_message(MyMessage(content="Hello4!"), DefaultTopicId())
    print('message published 4')
    await asyncio.sleep(8)
    await worker2.publish_message(MyMessage(content="Hello5!"), DefaultTopicId())
    print('message published 5')

    # await asyncio.sleep(60)

    await worker2.stop()
    # await worker1.stop_when_signal()
    print('sender stopped')


if __name__ == '__main__':
    asyncio.run(main())

