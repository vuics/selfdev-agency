#!/usr/bin/env python
'''
XMPP Agency - Manages and runs XMPP agents based on MongoDB configuration

# Architect Instructions for AI

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
Only run the agent if its `options.schemaVersion==='0.1'` and the agent is deployed (`options.deployed===True`).
The schema of each agent defined in agent.js in another Node.js repo with API server (I copy-pasted it below).
Translate the schema defined on Node.js to Python:
```js
export default mongoose.model(
  'Agent', // translates to 'agents' by Mongoose driver of MongoDB
  mongoose.Schema({
    userId: {
      type: ObjectId,
      required: true,
      ref: 'User'
    },

    deployed: false, // only run the agents with deployed===true

    options: {
      schemaVersion: String, // current schemaVersion==='0.1'

      name: String, // unique name
      description: String,

      systemMessage: String, // SystemMessage(SYSTEM_MESSAGE) to pass to LLM on LangChain

      protoAgent: String, // 'AliceAgent' or 'BobAgent' class on Python
      joinRooms: [ String ], // XMPP rooms to join,

      model: {
        provider: String, // Name of the LLM model provider such as 'openai' or 'anthropic'
        name: String, // Name of the LLM such as 'gpt-4o-mini' or 'claude-3-5-sonnet-20240620'
      },

      // other options will be defined here
    },
  })
    .plugin(mongooseTimestamp)
)
```

The XMPP settings will be taken from the environment variables:
```python
XMPP_HOST = os.getenv("XMPP_HOST", "selfdev-prosody.dev.local")
XMPP_PASSWORD = os.getenv("XMPP_PASSWORD", "123")
XMPP_MUC_HOST = os.getenv("XMPP_MUC_HOST", f"conference.{XMPP_HOST}")
XMPP_JOIN_ROOMS = json.loads(os.getenv('XMPP_JOIN_ROOMS', '[ "team", "a-suite", "agents" ]'))
```
Except of the XMPP_USER and XMPP_NICK, they should be redefined from the 'agents' collection document based on the `options.name`.
XMPP_USER = os.getenv("XMPP_USER", AGENT_NAME)
XMPP_NICK = os.getenv("XMPP_NICK", AGENT_NAME)
```
'''

import asyncio
import logging
import os
import json
from typing import Dict, Any, List, Optional, Type
from dotenv import load_dotenv
from motor.motor_asyncio import AsyncIOMotorClient
from pymongo.errors import ConnectionFailure

# Import agent classes
from alice import AliceAgent
# FIXME: enable back
# from bob import BobAgent
from xmpp_agent import XmppAgent
from base_model import init_model



# AI! Read all the instructions carefully in this comment and program accordingly with deep understanding of all details.
# Develop this agents.py as a scalable microservice running in docker containers. Parallel the load, if one container runs the agent, the other container should not run it. So the load with agents should be distributed between containers. Only one container should run the agent. All the valid agents should run. The validation is already implemented by checking: deployed===True and schemaVersion===‘0.1’ with the agent are defined as documents stored in the agents collection in the MongoDB.
# To ensure that only one container runs a particular agent while others remain idle, you can implement a **leader election** algorithm. Here’s how you might approach this:
#    - Use a Distributed Lock. Develop a consensus system based on Redis with a locking mechanism. Every time an agent attempts to start, it tries to acquire a lock; only the container that successfully acquires the lock will run the agent, while others will remain passive.
#    - Agent Coordination. When the active agent finishes its task, it should release the lock, allowing another container to take the lead.
# The redis service defined in docker-compose.yml as:
# ```yaml
#   redis:
#     hostname: redis.dev.local
#     image: redis:6-alpine
#     command: redis-server --appendonly yes
#     ports:
#     - 6379:6379
#     volumes:
#     - redis-volume:/data
#     networks:
#     - dev-network
# ```




# Load environment variables
load_dotenv()

# MongoDB connection settings
DB_URL = os.getenv("DB_URL", "mongodb://mongo.dev.local:27017/selfdev")
DB_NAME = DB_URL.split('/')[-1]

# XMPP default settings
XMPP_HOST = os.getenv("XMPP_HOST", "selfdev-prosody.dev.local")
XMPP_PASSWORD = os.getenv("XMPP_PASSWORD", "123")
XMPP_MUC_HOST = os.getenv("XMPP_MUC_HOST", f"conference.{XMPP_HOST}")
XMPP_JOIN_ROOMS = json.loads(os.getenv('XMPP_JOIN_ROOMS', '[ "team", "a-suite", "agents" ]'))

# Agent monitoring settings
MONITOR_SECONDS = int(os.getenv("MONITOR_SECONDS", "60"))

# Configure logging
logging.basicConfig(
  # level=logging.INFO,
  level=logging.DEBUG,
  format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("agents")

# Map of agent class names to their actual classes
AGENT_CLASSES = {
  "AliceAgent": AliceAgent,
  # FIXME: enable back
  # "BobAgent": BobAgent,
}

# Running agents registry
running_agents = {}

class AgentConfig:
  """Python representation of the MongoDB agent schema"""
  def __init__(self, doc: Dict[str, Any]):
    self.id = str(doc.get('_id'))
    self.user_id = str(doc.get('userId'))
    self.deployed = doc.get('deployed', False)

    options = doc.get('options', {})
    self.schema_version = options.get('schemaVersion')
    self.name = options.get('name')
    self.description = options.get('description')
    self.system_message = options.get('systemMessage', '')
    self.proto_agent = options.get('protoAgent')
    self.join_rooms = options.get('joinRooms', [])

    model = options.get('model', {})
    self.model_provider = model.get('provider')
    self.model_name = model.get('name')

  def is_valid(self) -> bool:
    """Check if the agent configuration is valid and should be deployed"""
    return bool(
      self.deployed and
      self.schema_version == '0.1' and
      self.name and
      self.proto_agent and
      self.proto_agent in AGENT_CLASSES and
      self.model_provider and
      self.model_name
    )

  def __str__(self) -> str:
    return f"Agent(name={self.name}, proto={self.proto_agent}, deployed={self.deployed}, model={self.model_provider}/{self.model_name} => valid={self.is_valid()})"

  def __repr__(self) -> str:
    return self.__str__()

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


async def get_agent_configs(db) -> List[AgentConfig]:
  """Retrieve all agent configurations from MongoDB"""
  try:
    agents_collection = db.agents
    cursor = agents_collection.find({})
    configs = []

    async for doc in cursor:
      config = AgentConfig(doc)
      configs.append(config)

    logger.info(f"Retrieved {len(configs)} agent configurations")
    return configs
  except Exception as e:
    logger.error(f"Error retrieving agent configurations: {e}")
    return []


async def start_agent(config: AgentConfig) -> Optional[XmppAgent]:
  """Start an agent based on its configuration"""
  try:
    if not config.is_valid():
      logger.warning(f"Invalid agent configuration: {config}")
      return None

    # Get the appropriate agent class
    agent_class: Type[XmppAgent] = AGENT_CLASSES[config.proto_agent]

    # Initialize the model for the agent
    model = init_model(
      model_provider=config.model_provider,
      model_name=config.model_name
    )

    # FIXME: Do the opposite way from os.environ assign to config (by default if not set)
    # Set environment variables for the agent
    os.environ["AGENT_NAME"] = config.name
    os.environ["SYSTEM_MESSAGE"] = config.system_message
    os.environ["MODEL_PROVIDER"] = config.model_provider
    os.environ["MODEL_NAME"] = config.model_name

    # Create and start the agent
    agent = agent_class(
      host=XMPP_HOST,
      user=config.name,  # Use agent name as XMPP username
      password=XMPP_PASSWORD,
      muc_host=XMPP_MUC_HOST,
      join_rooms=config.join_rooms or XMPP_JOIN_ROOMS,
      nick=config.name,  # Use agent name as XMPP nickname
    )

    logger.info(f"Started agent: {config.name} ({config.proto_agent})")
    return agent
  except Exception as e:
    logger.error(f"Error starting agent {config.name}: {e}")
    return None


async def stop_agent(agent_name: str):
  """Stop a running agent"""
  if agent_name in running_agents:
    try:
      agent = running_agents[agent_name]
      agent.disconnect()
      del running_agents[agent_name]
      logger.info(f"Stopped agent: {agent_name}")
    except Exception as e:
      logger.error(f"Error stopping agent {agent_name}: {e}")


async def sync_agents(db):
  """Synchronize running agents with the database configuration"""
  configs = await get_agent_configs(db)
  logger.debug('sync_agents():')
  logger.debug(f'  configs: {configs}')
  logger.debug(f'  (before) running_agents: {running_agents}')

  # Track which agents should be running
  should_run = {}

  # Start new agents or update existing ones
  for config in configs:
    if config.is_valid():
      should_run[config.name] = config

      if config.name not in running_agents:
        # Start new agent
        logger.debug(f'  start agent: {config.name}, config: {config}')
        agent = await start_agent(config)
        if agent:
          running_agents[config.name] = agent

  # Stop agents that should no longer be running
  for agent_name in list(running_agents.keys()):
    if agent_name not in should_run:
      logger.debug(f'  stop agent: {config.name}')
      await stop_agent(agent_name)
  logger.debug(f'  (after) running_agents: {running_agents}')


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


async def main():
  """Main entry point for the XMPP agency"""
  try:
    # Connect to MongoDB
    mongo_client = await connect_to_mongodb()
    db = mongo_client[DB_NAME]

    # Start monitoring for changes
    monitor_task = asyncio.create_task(monitor_agents(db))

    # Keep the application running
    await asyncio.gather(monitor_task)
  except Exception as e:
    logger.error(f"Fatal error in XMPP agency: {e}")
  finally:
    # Ensure all agents are stopped on shutdown
    for agent_name in list(running_agents.keys()):
      await stop_agent(agent_name)


if __name__ == "__main__":
  asyncio.run(main())
