#!/usr/bin/env python
'''
Alice Agent
'''

# Based on
# https://slixmpp.readthedocs.io/en/latest/getting_started/muc.html
# Slixmpp is an MIT licensed XMPP library for Python 3.7+,

import logging
import os
import ssl
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
import asyncio
import slixmpp

from base_model import init_model

load_dotenv()


AGENT_NAME = os.getenv("AGENT_NAME", "alice")
MODEL_PROVIDER = os.getenv("MODEL_PROVIDER", "openai")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
SYSTEM_MESSAGE = os.getenv("SYSTEM_MESSAGE", "")

XMPP_JID = os.getenv("XMPP_JID", f"{AGENT_NAME}@selfdev-prosody.dev.local")
XMPP_PASSWORD = os.getenv("XMPP_PASSWORD", "123")
XMPP_ROOM = os.getenv("XMPP_ROOM", "team@conference.selfdev-prosody.dev.local")
XMPP_NICK = os.getenv("XMPP_NICK", AGENT_NAME)


try:
  model = init_model(model_provider=MODEL_PROVIDER,
                     model_name=MODEL_NAME)
except Exception as e:
  print("Error initializing model:", e)


class AliceAgent(slixmpp.ClientXMPP):

    """
    A simple Slixmpp bot that will greets those
    who enter the room, and acknowledge any messages
    that mentions the bot's nickname.
    """

    def __init__(self, jid, password, room, nick):
        slixmpp.ClientXMPP.__init__(self, jid, password)

        # Allow insecure certificates
        #
        # Configure SSL context
        self.ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        self.ssl_context.check_hostname = False
        self.ssl_context.verify_mode = ssl.CERT_NONE
        # Enable all available protocols
        self.ssl_context.minimum_version = ssl.TLSVersion.MINIMUM_SUPPORTED
        self.ssl_context.maximum_version = ssl.TLSVersion.MAXIMUM_SUPPORTED
        # Register event handlers
        self.add_event_handler('ssl_invalid_cert', self.ssl_invalid_cert)

        self.room = room
        self.nick = nick

        # The session_start event will be triggered when
        # the bot establishes its connection with the server
        # and the XML streams are ready for use. We want to
        # listen for this event so that we we can initialize
        # our roster.
        self.add_event_handler("session_start", self.start)

        # The groupchat_message event is triggered whenever a message
        # stanza is received from any chat room. If you also also
        # register a handler for the 'message' event, MUC messages
        # will be processed by both handlers.
        self.add_event_handler("groupchat_message", self.muc_message)

        # The groupchat_presence event is triggered whenever a
        # presence stanza is received from any chat room, including
        # any presences you send yourself. To limit event handling
        # to a single room, use the events muc::room@server::presence,
        # muc::room@server::got_online, or muc::room@server::got_offline.
        self.add_event_handler("muc::%s::got_online" % self.room,
                               self.muc_online)


    def ssl_invalid_cert(self, pem_cert):
        print("Warning: Invalid SSL certificate received")
        return True


    async def start(self, event):
        """
        Process the session_start event.

        Typical actions for the session_start event are
        requesting the roster and broadcasting an initial
        presence stanza.

        Arguments:
            event -- An empty dictionary. The session_start
                     event does not provide any additional
                     data.
        """
        await self.get_roster()
        self.send_presence()
        self.plugin['xep_0045'].join_muc(self.room,
                                         self.nick,
                                         # If a room password is needed, use:
                                         # password=the_room_password,
                                         )

    def muc_message(self, msg):
        """
        Process incoming message stanzas from any chat room. Be aware
        that if you also have any handlers for the 'message' event,
        message stanzas may be processed by both handlers, so check
        the 'type' attribute when using a 'message' event handler.

        Whenever the bot's nickname is mentioned, respond to
        the message.

        IMPORTANT: Always check that a message is not from yourself,
                   otherwise you will create an infinite loop responding
                   to your own messages.

        This handler will reply to messages that mention
        the bot's nickname.

        Arguments:
            msg -- The received message stanza. See the documentation
                   for stanza objects and the Message stanza to see
                   how it may be used.
        """
        # print('msg:', msg)
        if msg['mucnick'] != self.nick and self.nick in msg['body']:
            try:
              prompt = msg['body']
              print('prompt:', prompt)
              ai_msg = model.invoke([
                SystemMessage(SYSTEM_MESSAGE),
                HumanMessage(prompt)
              ])
              print("ai_msg:", ai_msg.content)
              self.send_message(mto=msg['from'].bare,
                                mbody=ai_msg.content,
                                mtype='groupchat')
            except Exception as err:
              print('Chat error:', err)
              self.send_message(mto=msg['from'].bare,
                                mbody=f'Error: {str(err)}',
                                mtype='groupchat')
        # elif msg['mucnick'] != self.nick:
        #     self.send_message(mto=msg['from'].bare,
        #                       mbody=f"Echo: {msg['body']}",
        #                       mtype='groupchat')

    def muc_online(self, presence):
        """
        Process a presence stanza from a chat room. In this case,
        presences from users that have just come online are
        handled by sending a welcome message that includes
        the user's nickname and role in the room.

        Arguments:
            presence -- The received presence stanza. See the
                        documentation for the Presence stanza
                        to see how else it may be used.
        """
        # print('presense:', presence)
        if presence['muc']['nick'] != self.nick:
            self.send_message(mto=presence['from'].bare,
                              mbody="Hello, %s %s" % (presence['muc']['role'],
                                                      presence['muc']['nick']),
                              mtype='groupchat')


if __name__ == '__main__':

    logging.basicConfig(
      level=logging.DEBUG,
      # level=logging.ERROR,
      # level=logging.INFO,
      format='%(levelname)-8s %(message)s'
    )

    xmpp = AliceAgent(XMPP_JID, XMPP_PASSWORD, XMPP_ROOM, XMPP_NICK)
    xmpp.register_plugin('xep_0030')  # Service Discovery
    xmpp.register_plugin('xep_0045')  # Multi-User Chat
    xmpp.register_plugin('xep_0199')  # XMPP Ping

    xmpp.connect()

    asyncio.get_event_loop().run_forever()
