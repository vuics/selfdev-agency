'''
StorageV1 Agent Archetype
'''
import os
import logging
import re
from urllib.parse import urlparse
from datetime import datetime

from dotenv import load_dotenv
from pymongo import AsyncMongoClient
from bson import ObjectId

from xmpp_agent import XmppAgent

logger = logging.getLogger("StorageV1")

# Load environment variables
load_dotenv()

# MongoDB connection settings
DB_URL = os.getenv("DB_URL", "mongodb://mongo.dev.local:27017/selfdev")


class StorageV1(XmppAgent):
  '''
  StorageV1 provides storage with a database
  '''
  async def start(self):
    await super().start()
    try:
      # logger.debug(f'self.config: {self.config}')
      self.storage = self.config.options.storage
      # logger.debug(f'self.storage: {self.storage}')

      self.regex_list = re.compile(self.storage.commands["list"])
      self.regex_get = re.compile(self.storage.commands["get"])
      self.regex_set = re.compile(self.storage.commands["set"])
      self.regex_delete = re.compile(self.storage.commands["delete"])
      # logger.debug(f'regex_list: {self.regex_list}')
      # logger.debug(f'regex_get: {self.regex_get}')
      # logger.debug(f'regex_set: {self.regex_set}')
      # logger.debug(f'regex_delete: {self.regex_delete}')

      if self.storage.driver == "mongodb":
        self.client = AsyncMongoClient(DB_URL)
        # self.client = await AsyncMongoClient(DB_URL).aconnect()
        parsed_db_url = urlparse(DB_URL)
        database_name = parsed_db_url.path.lstrip('/')
        logger.info(f"database_name: {database_name}")
        self.db = self.client[database_name]
        self.storages = self.db["storages"]
      else:
        raise Exception('Unknown storage driver')

      logger.debug(f"self.client: {self.client}")
    except Exception as e:
      logger.error(f"Error initializing model: {e}")

  async def chat(self, *, prompt, reply_func=None):
    try:
      logger.debug(f"prompt: {prompt}")
      # logger.debug(f'self.config.options: {self.config.options}')
      content = ''

      if re.match(self.regex_list, prompt):
        logger.debug("LIST command")
        cursor = self.storages.find({
          "userId": ObjectId(self.config.userId),
          "namespace": self.storage.namespace,
        })
        items = await cursor.to_list(length=None)
        keys = [item["key"] for item in items]
        if self.storage.verbose == 2:
          content = "Keys: " + ", ".join(keys) if keys else "No keys found."
        elif self.storage.verbose == 1:
          content = "\n".join(keys) if keys else "NOT_FOUND"
        else:
          content = "\n".join(keys) if keys else " "

      elif match := re.match(self.regex_get, prompt):
        key = match.group("key")
        default = match.groupdict().get("default")
        logger.debug(f"GET command with key: {key}, default: {default}")
        doc = await self.storages.find_one({
          "userId": ObjectId(self.config.userId),
          "namespace": self.storage.namespace,
          "key": key,
        })
        if doc:
          content = doc["value"]
        elif default is not None:
          content = default
        else:
          if self.storage.verbose == 2:
            content = f"No entry found for key '{key}'."
          elif self.storage.verbose == 1:
            content = "NOT_FOUND"
          else:
            content = " "

      elif match := re.match(self.regex_set, prompt):
        key = match.group("key")
        value = match.group("value")
        logger.debug(f"SET command with key: {key}, value: {value}")
        now = datetime.utcnow()
        await self.storages.update_one({
          "userId": ObjectId(self.config.userId),
          "namespace": self.storage.namespace,
          "key": key,
        }, {
          "$set": {
            "value": value,
            "updatedAt": now,          # update every time
          },
          "$setOnInsert": {
            "createdAt": now           # only set on insert
          }
        }, upsert=True)
        if self.storage.verbose == 2:
          content = f"Set key '{key}' to value '{value}'."
        elif self.storage.verbose == 1:
          content = value
        else:
          content = " "

      elif match := re.match(self.regex_delete, prompt):
        key = match.group("key")
        logger.debug(f"DELETE command with key: {key}")
        result = await self.storages.delete_one({
          "userId": ObjectId(self.config.userId),
          "namespace": self.storage.namespace,
          "key": key,
        })
        if result.deleted_count > 0:
          if self.storage.verbose == 2:
            content = f"Deleted key '{key}'."
          elif self.storage.verbose == 1:
            content = "OK"
          else:
            content = " "
        else:
          if self.storage.verbose == 2:
            content = f"No entry found for key '{key}'."
          elif self.storage.verbose == 1:
            content = "NOT_FOUND"
          else:
            content = " "

      else:
        logger.warning("No valid command match")
        if self.storage.verbose == 2:
          content = 'Command not found'
        elif self.storage.verbose == 1:
          content = "UNKNOWN_COMMAND"
        else:
          content = " "

      return content
    except Exception as e:
      logger.error(f"Storage error: {e}")
      return f'Error: {str(e)}'

  async def disconnect(self):
    """
    Release resources
    """
    try:
      await super().disconnect()
      logger.debug("Disconnecting from MongoDB...")
      await self.client.close()
      logger.info("Disconnected from MongoDB")
    except Exception as e:
      logger.error(f"Disconnect error: {e}")
