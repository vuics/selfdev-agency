'''
Adam agent was the first in a history of agents. It was created by the Creator.
'''
import os
from dataclasses import dataclass
import asyncio

from autogen_core import DefaultTopicId, MessageContext, RoutedAgent, default_subscription, message_handler
from autogen_ext.runtimes.grpc import GrpcWorkerAgentRuntime
from dotenv import load_dotenv
import uuid
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_ollama import ChatOllama
# import json

# from message_types import AgenticMessage

load_dotenv()

HOST_ADDRESS = os.getenv("HOST_ADDRESS", "localhost:50051")
INIT_SLEEP = int(os.getenv("INIT_SLEEP", "5"))
AGENT_NAME = os.getenv("AGENT_NAME", "smith")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

# providers:
#  'openai', 'anthropic', 'ollama'
# models:
#  'gpt-4o', 'gpt-4o-mini', 'claude-3-5-sonnet-20240620', 
#  'medragondot/Sky-T1-32B-Preview', 
#  'llama3.3', 'gemma', 'mistral', 'tinyllama'
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")

if LLM_PROVIDER == "openai":
    chat = ChatOpenAI(model=MODEL_NAME, api_key=OPENAI_API_KEY)
elif LLM_PROVIDER == "anthropic":
    chat = ChatAnthropic(model=MODEL_NAME, api_key=ANTHROPIC_API_KEY)
elif LLM_PROVIDER == "ollama":
    chat = ChatOllama(model=MODEL_NAME)
else:
    raise ValueError(f"Unknown LLM provider: {LLM_PROVIDER}")


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
        if (message.id_replied != ''):
            return
        if message.to and AGENT_NAME not in message.to:
            print(f'Agent @{AGENT_NAME} not in message.to: {message.to}')
            return
        print(f"Received message: {message}")
        ai_msg = chat.invoke(message.content)
        print(f"ai_msg: {ai_msg.content}")
        await self.publish_message(
            AgenticMessage(
                id=str(uuid.uuid4()),
                id_replied=message.id,
                to=[message.fr],
                fr=AGENT_NAME,
                content=ai_msg.content,
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
