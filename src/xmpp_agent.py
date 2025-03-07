'''
Base XMPP Agent
'''
import ssl
from slixmpp import ClientXMPP
import httpx


class XmppAgent(ClientXMPP):
  """
  A Base XMPP Agent
  """

  def __init__(self, *, host, user, password, muc_host, join_rooms, nick, options):
    jid = f"{user}@{host}"
    ClientXMPP.__init__(self, jid, password)

    self.host = host
    self.user = user
    self.jid = jid
    self.password = password
    self.muc_host = muc_host
    self.join_rooms = join_rooms
    self.nick = nick
    self.join_room_jids = [f"{room}@{muc_host}" for room in self.join_rooms]
    self.options = options

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

    self.add_event_handler('failed_auth', self.failed_auth)

    # The session_start event will be triggered when
    # the bot establishes its connection with the server
    # and the XML streams are ready for use. We want to
    # listen for this event so that we we can initialize
    # our roster.
    self.add_event_handler("session_start", self.session_start)

    # The message event is triggered whenever a message
    # stanza is received. Be aware that that includes
    # MUC messages and error messages.
    self.add_event_handler("message", self.message)

    # The groupchat_message event is triggered whenever a message
    # stanza is received from any chat room. If you also also
    # register a handler for the 'message' event, MUC messages
    # will be processed by both handlers.
    self.add_event_handler("groupchat_message", self.groupchat_message)

    # The groupchat_presence event is triggered whenever a
    # presence stanza is received from any chat room, including
    # any presences you send yourself. To limit event handling
    # to a single room, use the events muc::room@server::presence,
    # muc::room@server::got_online, or muc::room@server::got_offline.
    for room_jid in self.join_room_jids:
      self.add_event_handler("muc::%s::got_online" % room_jid,
                             self.muc_online)

    self.add_event_handler("groupchat_direct_invite",
                           self.groupchat_direct_invite)

    self.register_plugin('xep_0030')  # Service Discovery
    self.register_plugin('xep_0045')  # Multi-User Chat
    self.register_plugin('xep_0199')  # XMPP Ping
    self.register_plugin('xep_0004')  # Data Forms
    self.register_plugin('xep_0060')  # PubSub
    self.register_plugin('xep_0249')  # Direct MUC Invitations

    self.connect()

  def start(self):
    pass

  def ssl_invalid_cert(self, pem_cert):
    print("Warning: Invalid SSL certificate received")
    return True

  async def failed_auth(self, event):
    print(f'{self.jid} Failed auth:', event)
    condition = event.get_condition()
    text = event._get_sub_text('text')
    print(f'{self.jid} Failed auth> condition:', condition, ', text:', text)
    # elem = event.xml.find('{urn:ietf:params:xml:ns:xmpp-sasl}text')
    # print('elem.text:', elem.text)
    if text == "Unable to authorize you with the authentication credentials you've sent.":
      print('I can register the user')
      registered = await self.register_user()
      if registered:
        print('User registered. Reconnect')
        self.connect()

  async def register_user(self):
    print('Register a new XMPP user with credentials> user:', self.user,
          ', password:', self.password)
    try:
      async with httpx.AsyncClient() as client:
        response = await client.get(
          f"http://{self.host}:8387/register",
          params={
            "user": self.user,
            "password": self.password,
            "host": self.host,
          },
          headers={"Content-Type": "application/json"}
        )
      print('XMPP Registration Status Code:', response.status_code)
      print('XMPP Registration Data:', response.text)
      return True
    except Exception as err:
      print('XMPP registration error:', err)
      return False

  async def session_start(self, event):
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
    print('Session started')
    for room_jid in self.join_room_jids:
      print('  Join room> jid:', room_jid, ', nick:', self.nick)
      self.plugin['xep_0045'].join_muc(room_jid, self.nick)

  def chat(self, prompt):
    print("WARNING: XmppAgent.chat() should be defined in child class")

  def message(self, msg):
    """
    Process incoming message stanzas. Be aware that this also
    includes MUC messages and error messages. It is usually
    a good idea to check the messages's type before processing
    or sending replies.

    Arguments:
      msg -- The received message stanza. See the documentation
           for stanza objects and the Message stanza to see
           how it may be used.
    """
    # print('msg:', msg)
    # print('msg type:', msg['type'])
    # print('msg body:', msg['body'])
    # print('msg mucnick:', msg['mucnick'])
    # print('msg from:', msg['from'])
    if msg['type'] in ('chat', 'normal'):
      try:
        content = self.chat(msg['body'])
        # msg.reply(content).send()
        self.send_message(mto=msg['from'].bare,
                          mbody=content,
                          mtype='chat')
      except Exception as err:
        print('message error:', err)

  def groupchat_message(self, msg):
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
    # print('msg type:', msg['type'])
    # print('msg body:', msg['body'])
    # print('msg mucnick:', msg['mucnick'])
    # print('msg from:', msg['from'])
    if msg['mucnick'] != self.nick and self.nick in msg['body']:
      try:
        content = self.chat(msg['body'])
        self.send_message(mto=msg['from'].bare,
                          mbody=content,
                          mtype='groupchat')
      except Exception as err:
        print('groupchat_message error:', err)
    # elif msg['mucnick'] != self.nick:
    #   self.send_message(mto=msg['from'].bare,
    #                     mbody=f"Echo: {msg['body']}",
    #                     mtype='groupchat')

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

  async def groupchat_direct_invite(self, msg):
    """
    Handler for direct MUC invitations.
    """
    print('Groupchat Invite msg:', msg)
    print(f"  Inviter: {msg['from']}")
    print(f"  Invitee: {msg['to']}")
    invite = msg.xml.find('.//{jabber:x:conference}x')
    room_jid = invite.get('jid')
    print(f"  Room: {room_jid}")
    print(f"  Reason: {invite.get('reason')}")

    print('Joining the room> room_jid:', room_jid)
    self.add_event_handler("muc::%s::got_online" % room_jid, self.muc_online)
    self.plugin['xep_0045'].join_muc(room_jid, self.nick)
