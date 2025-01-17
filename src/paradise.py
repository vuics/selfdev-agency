'''
Description: This is the main entry point for the Paradise runtime.
'''
import asyncio
import logging

from autogen_ext.runtimes.grpc import GrpcWorkerAgentRuntimeHost


async def main() -> None:
    ''' Main entry point '''
    service = GrpcWorkerAgentRuntimeHost(address="0.0.0.0:50051")
    service.start()
    await service.stop_when_signal()


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    # logging.getLogger("autogen_core").setLevel(logging.INFO)
    logging.getLogger("autogen_core").setLevel(logging.WARNING)
    asyncio.run(main())
