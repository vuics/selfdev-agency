#!/usr/bin/env python
'''
Alice Agent
'''

import logging
import os
import json
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
import asyncio

from base_model import init_model
from xmpp_agent import XmppAgent

load_dotenv()


AGENT_NAME = os.getenv("AGENT_NAME", "alice")
SYSTEM_MESSAGE = os.getenv("SYSTEM_MESSAGE", "")

XMPP_HOST = os.getenv("XMPP_HOST", "selfdev-prosody.dev.local")
XMPP_USER = os.getenv("XMPP_USER", AGENT_NAME)
XMPP_PASSWORD = os.getenv("XMPP_PASSWORD", "123")
XMPP_MUC_HOST = os.getenv("XMPP_MUC_HOST", f"conference.{XMPP_HOST}")
XMPP_JOIN_ROOMS = json.loads(os.getenv('XMPP_JOIN_ROOMS',
                                       '[ "team", "a-suite", "agents" ]'))
XMPP_NICK = os.getenv("XMPP_NICK", AGENT_NAME)

MODEL_PROVIDER = os.getenv("MODEL_PROVIDER", "openai")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")

try:
  model = init_model(model_provider=MODEL_PROVIDER,
                     model_name=MODEL_NAME)
except Exception as e:
  print("Error initializing model:", e)


class AliceAgent(XmppAgent):
  # def __init__(self, *, host, user, password, room, nick):
  #   XmppAgent.__init__(self, host, user, password, room, nick)

  def chat(self, prompt):
    try:
      print('prompt:', prompt)
      ai_msg = model.invoke([
        SystemMessage(SYSTEM_MESSAGE),
        HumanMessage(prompt)
      ])
      print("ai_msg:", ai_msg.content)
      return ai_msg.content
    except Exception as err:
      print('chat error:', err)
      return 'Error: {str(err)}'


if __name__ == '__main__':
  logging.basicConfig(
    level=logging.INFO,  # DEBUG, ERROR, INFO,
    format='%(levelname)-8s %(message)s'
  )

  agent = AliceAgent(
    host=XMPP_HOST,
    user=XMPP_USER,
    password=XMPP_PASSWORD,
    muc_host=XMPP_MUC_HOST,
    join_rooms=XMPP_JOIN_ROOMS,
    nick=XMPP_NICK,
  )
  asyncio.get_event_loop().run_forever()
