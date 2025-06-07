#!/usr/bin/env python
'''
XMPP Agency - Manages and runs XMPP agents based on MongoDB configuration

Architect Instructions for AI:
---

Read all the instruction carefully in this file and program accordingly with deep understanding of all details.
Create an agency application on Python 3.12 (or higher) using asyncio in this file src/agents.py.
Use file requirements.txt to define the requirements and dependencies.
The agency runs several agents like AliceAgent defined in src/alice.py and BobAgent defined in src/bob.py.
Both AliceAgent and BobAgent derived from the base class XmppAgent defined in src/xmpp_agent.py.
What agent should we run depend on the value of the field called `protoAgent` defined in MongoDB collection 'agents'.
The agency should be scalable and the load could be distributed between several docker containers that runs the same agency defined in Dockerfile.
The agency should be connected to the MongoDB and take configuration of each agent from the documents defined in the 'users' collection.
You can add other options to the schema if needed.
We run MongoDB with `DB_URL=mongodb://mongo.dev.local:27017/selfdev`, in which the collection `agents` exists.
The agency should read all the documents in the 'agents' collection.
Only run the agent if the agent is deployed (`deployed===True`).

---

Read all the instructions carefully in this comment and program accordingly with deep understanding of all details.
Develop this agents.py as a scalable microservice running in docker containers. Parallel the load, if one container runs the agent, the other container should not run it. So the load with agents should be distributed between containers. Only one container should run the agent. All the valid agents should run. The validation is already implemented by checking: deployed===True with the agent are defined as documents stored in the agents collection in the MongoDB.
To ensure that only one container runs a particular agent while others remain idle, you can implement a **leader election** algorithm. Here’s how you might approach this:
   - Use a Distributed Lock. Develop a consensus system based on Redis with a locking mechanism. Every time an agent attempts to start, it tries to acquire a lock; only the container that successfully acquires the lock will run the agent, while others will remain passive.
   - Agent Coordination. When the active agent finishes its task, it should release the lock, allowing another container to take the lead.

'''

import asyncio
import random
import atexit
import logging
import os
import json
import signal
import time
import traceback

import uuid
from typing import Dict, Any, List, Optional, Type
from dotenv import load_dotenv
from motor.motor_asyncio import AsyncIOMotorClient
from pymongo.errors import ConnectionFailure
import redis.asyncio as redis
from box import Box
import hvac

from helpers import str_to_bool

# Import agent classes
from xmpp_agent import XmppAgent
from chat_v1 import ChatV1
from rag_v1 import RagV1
from notebook_v1 import NotebookV1
from command_v1 import CommandV1
from langflow_v1 import LangflowV1
from nodered_v1 import NoderedV1
from quantum_v1 import QuantumV1
from storage_v1 import StorageV1
from code_v1 import CodeV1
from imagegen_v1 import ImagegenV1

# Load environment variables
load_dotenv()

# MongoDB connection settings
DB_URL = os.getenv("DB_URL", "mongodb://mongo.dev.local:27017/selfdev")

# XMPP default settings
XMPP_HOST = os.getenv("XMPP_HOST", "selfdev-prosody.dev.local")
XMPP_PASSWORD = os.getenv("XMPP_PASSWORD", "123")
XMPP_MUC_HOST = os.getenv("XMPP_MUC_HOST", f"conference.{XMPP_HOST}")

# Agent monitoring settings
MONITOR_SECONDS = int(os.getenv("MONITOR_SECONDS", "60"))

# Redis settings for distributed locks
REDIS_URL = os.getenv("REDIS_URL", "redis://redis.dev.local:6379/0")
REDIS_SOCKET_TIMEOUT = int(os.getenv("REDIS_SOCKET_TIMEOUT", "10"))  # 10s
REDIS_CONNECT_TIMEOUT = int(os.getenv("REDIS_CONNECT_TIMEOUT", "15"))  # 15s
LOCK_TIMEOUT = int(os.getenv("LOCK_TIMEOUT", "120"))  # 2m
LOCK_REFRESH = int(os.getenv("LOCK_REFRESH", "30"))   # 30s

# Generate a unique ID for this container instance
CONTAINER_ID = os.getenv("CONTAINER_ID", os.getenv("HOSTNAME", str(uuid.uuid4())))

FILTER_ARCHETYPES = json.loads(os.getenv('FILTER_ARCHETYPES', '[ ]'))

VAULT_ENABLE = str_to_bool(os.getenv("VAULT_ENABLE", "false"))
VAULT_ADDR = os.getenv("VAULT_ADDR", "http://127.0.0.1:8200")
VAULT_TOKEN = os.getenv("VAULT_TOKEN", "(not-set)")
VAULT_UNSEAL = str_to_bool(os.getenv("VAULT_UNSEAL", "false"))
VAULT_UNSEAL_KEYS = os.getenv("VAULT_UNSEAL_KEYS", "(not-set),(not-set),(not-set),(not-set),(not-set)").split(',')


# Configure logging
logging.basicConfig(
  # level=logging.INFO,
  level=logging.DEBUG,
  format='%(levelname)s: %(message)s'
)
logger = logging.getLogger("agents")

logging.getLogger("pymongo").setLevel(logging.INFO)
logging.getLogger("motor").setLevel(logging.INFO)
# logging.getLogger("slixmpp").setLevel(logging.INFO)
logging.getLogger("httpx").setLevel(logging.INFO)
logging.getLogger("httpcore").setLevel(logging.INFO)

# Map of agent class names to their actual classes
ARCHETYPE_CLASSES = {
  "chat-v1.0": ChatV1,
  "rag-v1.0": RagV1,
  "notebook-v1.0": NotebookV1,
  "command-v1.0": CommandV1,
  "langflow-v1.0": LangflowV1,
  "nodered-v1.0": NoderedV1,
  "quantum-v1.0": QuantumV1,
  "storage-v1.0": StorageV1,
  "code-v1.0": CodeV1,
  "imagegen-v1.0": ImagegenV1,
}

# Running agents registry
running_agents = {}

# Redis client for distributed locks
redis_client = None

# HashiCorp Vault client for secrets ({ valueFromVault: "vaultKey" })
vault_client = None


class AgentConfig:
  """Python representation of the MongoDB agent schema"""
  def __init__(self, doc: Dict[str, Any], user):
    self.doc = doc
    self.user = user
    self.id = str(doc.get('_id'))
    self.userId = str(doc.get('userId'))

    self.deployed = doc.get('deployed', False)
    self.archetype = doc.get('archetype', None)
    self.options = Box(doc.get('options', {}))
    self.updatedAt = doc.get('updatedAt', None)
    self.name = self.options.name
    self.joinRooms = self.options.joinRooms
    self.replace_vault_values(self.options)

  def is_valid(self) -> bool:
    """Check if the agent configuration is valid and should be deployed"""
    return bool(self.deployed and self.name and
                self.archetype in ARCHETYPE_CLASSES and
                (True if not FILTER_ARCHETYPES else self.archetype in FILTER_ARCHETYPES))

  def __str__(self) -> str:
    # return f"""{self.name}({self.archetype}, {self.deployed and 'deployed' or 'undeployed'}, {self.is_valid() and 'valid' or 'invalid'}) in {self.joinRooms}"""
    return f"{self.name}({self.archetype})"

  def __repr__(self) -> str:
    return self.__str__()

  def get_vault_value(self, vault_key: str) -> str:
    if not vault_client:
      return ''
    read_response = vault_client.secrets.kv.read_secret_version(path=f'user_{self.userId}')
    vault_value = read_response['data']['data'][vault_key]
    return vault_value

  def replace_vault_values(self, obj: Any):
    if not vault_client:
      return ''

    if isinstance(obj, Box):
      for key in list(obj.keys()):
        value = obj[key]
        # Check for {'valueFromVault': 'vaultKey'} pattern
        if isinstance(value, dict) and set(value.keys()) == {"valueFromVault"}:
          vault_key = value["valueFromVault"]
          obj[key] = self.get_vault_value(vault_key)
        else:
          self.replace_vault_values(value)
    elif isinstance(obj, dict):
      for key in list(obj.keys()):
        value = obj[key]
        if isinstance(value, dict) and set(value.keys()) == {"valueFromVault"}:
          vault_key = value["valueFromVault"]
          obj[key] = self.get_vault_value(vault_key)
        else:
          self.replace_vault_values(value)
    elif isinstance(obj, list):
      for item in obj:
        self.replace_vault_values(item)


async def connect_to_redis():
  """Connect to Redis for distributed locks"""
  global redis_client
  try:
    redis_client = await redis.Redis.from_url(
      REDIS_URL,
      decode_responses=False,
      retry_on_timeout=True,
      socket_timeout=REDIS_SOCKET_TIMEOUT,
      socket_connect_timeout=REDIS_CONNECT_TIMEOUT,
    )
    await redis_client.ping()
    logger.info(f"Connected to Redis at {REDIS_URL}")
    return redis_client
  except Exception as e:
    logger.error(f"Failed to connect to Redis: {e}")
    raise


async def connect_to_mongodb() -> AsyncIOMotorClient:
  """Connect to MongoDB and return the client"""
  try:
    client = AsyncIOMotorClient(DB_URL)
    # Verify connection is working
    await client.admin.command('ping')
    logger.info(f"Connected to MongoDB at {DB_URL}")
    return client
  except ConnectionFailure as e:
    logger.error(f"Failed to connect to MongoDB: {e}")
    raise


async def connect_to_vault(retries: int = 20, base_delay: float = 1.0, max_delay: float = 60.0):
  """Connect to Vault with adaptive exponential backoff delay and jitter."""
  global vault_client

  if not VAULT_ENABLE:
    logger.warning("Vault is disabled.")
    return

  for attempt in range(1, retries + 1):
    try:
      vault_client = hvac.Client(url=VAULT_ADDR, token=VAULT_TOKEN)

      if VAULT_UNSEAL:
        is_sealed = vault_client.sys.is_sealed()
        logger.info(f"Vault is_sealed (before): {is_sealed}")
        if is_sealed:
          for i, key in enumerate(VAULT_UNSEAL_KEYS):
            if not key or key.strip().lower() == "(not-set)":
              logger.warning(f"Vault unseal key {i + 1} is not set. Skipping.")
              continue
            try:
              response = vault_client.sys.submit_unseal_key(key.strip())
              logger.debug(f"Vault unseal response {i + 1}: {response}")
              if not response.get("sealed", True):
                logger.info(f"Vault unsealed after {i + 1} keys.")
                break
            except Exception as e:
              logger.error(f"Error submitting unseal key {i + 1}: {e}")
          is_sealed = vault_client.sys.is_sealed()
          logger.info(f"Vault is_sealed (after): {is_sealed}")
        else:
          logger.info("Vault was already unsealed.")
      else:
        logger.info("Vault unsealing is disabled by VAULT_UNSEAL=false")

      if vault_client.is_authenticated():
        logger.info(f"Connected to Vault at {VAULT_ADDR}, is_authenticated: True")
        return vault_client
      else:
        logger.warning(f"Attempt {attempt}: Authentication failed.")
    except Exception as e:
      logger.error(f"Attempt {attempt}: Exception during Vault connection: {e}")

    if attempt < retries:
      # Exponential backoff with jitter
      delay = min(base_delay * (2 ** (attempt - 1)), max_delay)
      jitter = random.uniform(0.5, 1.5)
      adaptive_delay = delay * jitter
      logger.info(f"Retrying in {adaptive_delay:.2f} seconds...")
      await asyncio.sleep(adaptive_delay)

  logger.critical("All attempts to connect to Vault failed.")
  raise ConnectionError("Failed to authenticate with Vault after multiple attempts.")


async def get_agent_configs(db) -> List[AgentConfig]:
  """Retrieve all agent configurations from MongoDB"""
  try:
    cursor = db.agents.find({})
    configs = []

    async for doc in cursor:
      user = await db.users.find_one({"_id": doc.get('userId')})
      # logger.debug(f'user: {user}')
      config = AgentConfig(doc, user)
      configs.append(config)

    logger.info(f"Retrieved {len(configs)} agent configurations")
    return configs
  except Exception as e:
    logger.error(f"Error retrieving agent configurations: {e}")
    return []


async def check_and_clear_stale_lock(agent_name: str) -> bool:
  """
  Check if a lock exists but is stale (no heartbeat updates)
  Returns True if lock was cleared or doesn't exist, False if lock is valid
  """
  if not redis_client:
    logger.error("Redis client not initialized")
    return False

  lock_key = f"agent_lock:{agent_name}"
  heartbeat_key = f"agent_heartbeat:{agent_name}"

  try:
    # Check if lock exists
    lock_owner = await redis_client.get(lock_key)
    if not lock_owner:
      # No lock exists
      return True

    # Check if heartbeat exists and is recent
    last_heartbeat = await redis_client.get(heartbeat_key)
    if not last_heartbeat:
      # No heartbeat, lock is stale
      logger.warning(f"Found stale lock for agent {agent_name} with no heartbeat, clearing")
      await redis_client.delete(lock_key)
      return True

    # Check heartbeat timestamp
    try:
      heartbeat_time = float(last_heartbeat.decode())
      current_time = time.time()
      if current_time - heartbeat_time > LOCK_TIMEOUT:
        # Heartbeat is too old, lock is stale
        logger.warning(f"Found stale lock for agent {agent_name} with old heartbeat, clearing")
        await redis_client.delete(lock_key)
        await redis_client.delete(heartbeat_key)
        return True
    except (ValueError, TypeError):
      # Invalid heartbeat format, consider lock stale
      logger.warning(f"Found lock with invalid heartbeat format for agent {agent_name}, clearing")
      await redis_client.delete(lock_key)
      await redis_client.delete(heartbeat_key)
      return True

    # Lock exists and has a recent heartbeat
    return False
  except Exception as e:
    logger.error(f"Error checking stale lock for agent {agent_name}: {e}")
    return False


async def acquire_lock(agent_name: str) -> bool:
  """
  Acquire a distributed lock for an agent to ensure only one container runs it

  Returns True if lock was acquired, False otherwise
  """
  if not redis_client:
    logger.error("Redis client not initialized")
    return False

  lock_key = f"agent_lock:{agent_name}"
  heartbeat_key = f"agent_heartbeat:{agent_name}"

  try:
    # Check if we already own this lock
    lock_owner = await redis_client.get(lock_key)
    if lock_owner and lock_owner.decode() == CONTAINER_ID:
      logger.debug(f"Already own lock for agent {agent_name}")
      # Update heartbeat
      await redis_client.set(heartbeat_key, str(time.time()), ex=LOCK_TIMEOUT * 2)
      return True

    # Check for and clear stale locks
    lock_cleared = await check_and_clear_stale_lock(agent_name)
    if not lock_cleared:
      logger.debug(f"Lock for agent {agent_name} is owned by {lock_owner}")
      return False

    # Try to acquire the lock with our container ID
    acquired = await redis_client.set(
      lock_key,
      CONTAINER_ID,
      nx=True,  # Only set if key doesn't exist
      ex=LOCK_TIMEOUT
    )

    if acquired:
      # Set initial heartbeat
      await redis_client.set(heartbeat_key, str(time.time()), ex=LOCK_TIMEOUT * 2)
      logger.info(f"Acquired lock for agent {agent_name}")
      # Start a background task to refresh the lock
      asyncio.create_task(refresh_lock(agent_name))
      return True
    else:
      logger.debug(f"Race condition: Failed to acquire lock for agent {agent_name}")
      return False
  except Exception as e:
    logger.error(f"Error acquiring lock for agent {agent_name}: {e}")
    return False


async def refresh_lock(agent_name: str):
  """Periodically refresh the lock to maintain ownership"""
  lock_key = f"agent_lock:{agent_name}"
  heartbeat_key = f"agent_heartbeat:{agent_name}"

  while agent_name in running_agents:
    try:
      # Only refresh if we still own the lock
      lock_owner = await redis_client.get(lock_key)
      if lock_owner and lock_owner.decode() == CONTAINER_ID:
        # Refresh lock expiration
        await redis_client.expire(lock_key, LOCK_TIMEOUT)
        # Update heartbeat timestamp
        await redis_client.set(heartbeat_key, str(time.time()), ex=LOCK_TIMEOUT * 2)
        logger.debug(f"Refreshed lock for agent {agent_name}")
      else:
        logger.warning(f"Lost lock ownership for agent {agent_name}")
        # We lost the lock, stop the agent
        await stop_agent(agent_name)
        break
    except Exception as e:
      logger.error(f"Error refreshing lock for agent {agent_name}: {e}")

    # Wait before refreshing again
    await asyncio.sleep(LOCK_REFRESH)


async def release_lock(agent_name: str):
  """Release the distributed lock for an agent"""
  if not redis_client:
    return

  lock_key = f"agent_lock:{agent_name}"
  heartbeat_key = f"agent_heartbeat:{agent_name}"

  try:
    # Only delete the lock if we own it
    lock_owner = await redis_client.get(lock_key)
    if lock_owner and lock_owner.decode() == CONTAINER_ID:
      await redis_client.delete(lock_key)
      await redis_client.delete(heartbeat_key)
      logger.info(f"Released lock for agent {agent_name}")
  except Exception as e:
    logger.error(f"Error releasing lock for agent {agent_name}: {e}")


async def start_agent(config: AgentConfig) -> Optional[XmppAgent]:
  """Start an agent based on its configuration"""
  try:
    if not config.is_valid():
      logger.warning(f"Invalid agent configuration: {config}")
      return None

    # Try to acquire a distributed lock for this agent
    lock_acquired = await acquire_lock(config.name)
    if not lock_acquired:
      logger.info(f"Agent {config.archetype}({config.name}) is already running in another container")
      return None

    # Get the appropriate agent class
    agent_class: Type[XmppAgent] = ARCHETYPE_CLASSES[config.archetype]

    # logger.debug(f"config.user: {config.user}")
    # logger.debug(f"config.user.xmpp: {config.user.get('xmpp')}")
    # logger.debug(f"config.user.xmpp.user: {config.user.get('xmpp')['user']}")

    # Create and start the agent
    agent = agent_class(
      host=XMPP_HOST,
      user=config.name,  # Use agent name as XMPP username
      password=XMPP_PASSWORD,
      muc_host=XMPP_MUC_HOST,
      join_rooms=config.joinRooms,
      nick=config.name,  # Use agent name as XMPP nickname
      config=config,
      ownername=(config.user.get('xmpp')['user']).lower(),
    )
    asyncio.create_task(agent.start())

    logger.info(f"Started agent: {config.archetype} ({config.name})")
    return agent
  except Exception as e:
    # Release the lock if we failed to start the agent
    await release_lock(config.name)
    logger.error(f"Error starting agent {config.name}: {e}")
    traceback.print_exc()
    return None


async def stop_agent(agent_name: str):
  """Stop a running agent"""
  if agent_name in running_agents:
    try:
      agent = running_agents[agent_name]
      await agent.disconnect()
      del running_agents[agent_name]
      logger.info(f"Stopped agent: {agent_name}")

      # Release the distributed lock
      await release_lock(agent_name)
    except Exception as e:
      logger.error(f"Error stopping agent {agent_name}: {e}")


async def sync_agents(db):
  """Synchronize running agents with the database configuration"""
  configs = await get_agent_configs(db)
  logger.debug('sync_agents():')
  # logger.debug(f'🗂️ Configs: {configs}')
  logger.debug(f'🥚 BEFORE: running_agents: {running_agents}')

  # Track which agents should be running
  should_run = {}

  # Start new agents or update existing ones
  for config in configs:
    if config.is_valid():
      should_run[config.name] = config

      if config.name not in running_agents:
        # Start new agent
        logger.info(f'▶️ Start agent: {config.name}, config: {config}')
        agent = await start_agent(config)
        if agent:
          running_agents[config.name] = agent
      else:
        # Check if config was updated, then need to restart the agent
        if config.updatedAt != running_agents[config.name].config.updatedAt:
          logger.info(f'🔄 Restart agent: {config.name}, config: {config} because its config got updated: old updatedAt: {config.updatedAt}, new updatedAt: {running_agents[config.name].config.updatedAt}.')
          await stop_agent(config.name)
          agent = await start_agent(config)
          if agent:
            running_agents[config.name] = agent

  # Stop agents that should no longer be running
  for agent_name in list(running_agents.keys()):
    if agent_name not in should_run:
      logger.info(f'⏹️ Stop agent: {config.name}')
      await stop_agent(agent_name)
  logger.debug(f'🐣 AFTER: running_agents: {running_agents}')


async def monitor_agents(db):
  """Periodically check for changes in agent configurations"""
  while True:
    try:
      await sync_agents(db)
    except Exception as e:
      logger.error(f"Error in agent monitoring: {e}")

    # Wait before checking again
    logger.debug(f"Waiting {MONITOR_SECONDS} seconds before next check")
    await asyncio.sleep(MONITOR_SECONDS)  # Check based on environment variable


async def cleanup_locks():
  """Clean up all locks owned by this container on shutdown"""
  if not redis_client:
    return

  try:
    # Get all lock keys
    lock_pattern = "agent_lock:*"
    cursor = 0
    while True:
      cursor, keys = await redis_client.scan(cursor, match=lock_pattern, count=100)
      for key in keys:
        try:
          key_str = key.decode()
          lock_owner = await redis_client.get(key)
          if lock_owner and lock_owner.decode() == CONTAINER_ID:
            agent_name = key_str.split(':')[1]
            await release_lock(agent_name)
        except Exception as e:
          logger.error(f"Error cleaning up lock {key}: {e}")

      if cursor == 0:
        break
  except Exception as e:
    logger.error(f"Error in cleanup_locks: {e}")


def sync_cleanup_locks():
  """Synchronous version of cleanup_locks for atexit registration"""
  if running_agents:
    logger.info(f"Emergency cleanup of locks for container {CONTAINER_ID}")
    # Create a new event loop for the cleanup
    loop = asyncio.new_event_loop()
    try:
      loop.run_until_complete(cleanup_locks())
    except Exception as e:
      logger.error(f"Error in emergency cleanup: {e}")
    finally:
      loop.close()


async def shutdown():
  """Gracefully shut down the application"""
  logger.info("Shutting down XMPP agency...")
  # Stop all running agents
  for agent_name in list(running_agents.keys()):
    await stop_agent(agent_name)

  # Clean up any remaining locks
  await cleanup_locks()

  # Close Redis connection
  if redis_client:
    await redis_client.close()
  logger.info("Shutdown complete")


async def main():
  """Main entry point for the XMPP agency"""
  try:
    sleep_time_sec = random.randint(0, 3000) / 1000
    logger.info(f"Sleep randomly for {sleep_time_sec:.3f} seconds)")
    await asyncio.sleep(sleep_time_sec)

    # Connect to MongoDB and Redis
    mongo_client = await connect_to_mongodb()
    db_name = DB_URL.split('/')[-1]
    db = mongo_client[db_name]

    await connect_to_redis()

    await connect_to_vault()

    logger.info(f"Starting XMPP agency with container ID: {CONTAINER_ID}")

    # Register cleanup with atexit to handle unexpected shutdowns
    atexit.register(sync_cleanup_locks)

    # Set up signal handlers for graceful shutdown
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
      loop.add_signal_handler(sig, lambda: asyncio.create_task(shutdown()))

    # Start monitoring for changes
    monitor_task = asyncio.create_task(monitor_agents(db))

    # Keep the application running
    await monitor_task
  except Exception as e:
    logger.error(f"Fatal error in XMPP agency: {e}")
  finally:
    await shutdown()


if __name__ == "__main__":
  asyncio.run(main())
