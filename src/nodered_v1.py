'''
NoderedV1 Agent Archetype
'''
import os
import logging
import json
import copy

import httpx

from dotenv import load_dotenv

from xmpp_agent import XmppAgent
from helpers import extract_and_parse_json

logger = logging.getLogger("NoderedV1")

load_dotenv()

NODERED_URL = os.getenv("NODERED_URL", "http://localhost:1880")


class NoderedV1(XmppAgent):
  '''
  NoderedV1 provides execution of Langflow workflows
  '''
  async def start(self):
    await super().start()

  async def chat(self, *, prompt, reply_func=None):
    try:
      logger.debug(f"prompt: {prompt}")
      logger.debug(f'self.config.options: {self.config.options}')

      webhook = self.config.options.webhook
      url = f"{NODERED_URL}{webhook.route}"
      logger.debug(f"url: {url}")
      headers = {
        "Content-Type": "application/json"
      }

      payload = {}
      if hasattr(webhook, 'payload'):
        payload = copy.deepcopy(webhook.payload)
      if hasattr(webhook, 'parseJson') and webhook.parseJson:
        try:
          parsed_json = extract_and_parse_json(prompt)
          logger.debug(f'parsed_json: {parsed_json}')
          payload.update(parsed_json)
        except Exception:
          pass
      if hasattr(webhook, 'promptKey'):
        payload[webhook.promptKey] = prompt
      logger.debug(f"payload: {payload}")

      async with httpx.AsyncClient() as client:
        allowed_methods = {"get", "post", "put", "patch", "delete"}
        method_lower = webhook.method.lower()
        if method_lower not in allowed_methods:
          raise ValueError(f"Unsupported method: {webhook.method}")

        client_method_func = getattr(client, webhook.method.lower())
        response = await client_method_func(url, json=payload, headers=headers)
        response.raise_for_status()  # Raise exception for bad status codes

      logger.debug(f"response.text: {response.text}")
      # return self.extract_response(response.text)
      return response.text

    except httpx.RequestError as e:
      logger.error(f"Error making API request: {e}")
      return f'Error making API request: {str(e)}'

    except httpx.HTTPStatusError as e:
      logger.error(f"HTTP error response: {e.response.status_code} - {e.response.text}")
      return f'HTTP error response: {e.response.status_code} - {e.response.text}'

    except ValueError as e:
      logger.error(f"Error parsing response: {e}")
      return f'Error parsing response: {str(e)}'

    except Exception as e:
      logger.error(f"Chat error: {e}")
      return f'Error: {str(e)}'

  # def extract_response(self, response_text: str) -> str:
  #   try:
  #     extracted = json.loads(response_text)
  #     logger.debug(f'Extracted: {extracted}')
  #     return extracted
  #   except (KeyError, IndexError, TypeError, json.JSONDecodeError) as e:
  #     return f"Error extracting message: {str(e)}"
