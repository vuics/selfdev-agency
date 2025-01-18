'''
Description: This is the main entry point for the Paradise runtime.
'''
import os
import asyncio
import logging

from dotenv import load_dotenv
from autogen_ext.runtimes.grpc import GrpcWorkerAgentRuntimeHost

load_dotenv()


ADDRESS = os.getenv("ADDRESS", "0.0.0.0:50051")


async def main() -> None:
    ''' Main entry point '''
    print('Start gRPC Worket Agent Runtime Host on:', ADDRESS)
    service = GrpcWorkerAgentRuntimeHost(address=ADDRESS)
    service.start()
    await service.stop_when_signal()


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    # logging.getLogger("autogen_core").setLevel(logging.INFO)
    logging.getLogger("autogen_core").setLevel(logging.WARNING)
    asyncio.run(main())
