# AI! Read all the instruction carefully in this file and program accordingly with deep understanding of all details.
# Create an agency application on Python 3.12 (or higher) using asyncio in this file src/xmpp_agency.py.
# Use file requirements.txt to define the requirements and dependencies.
# The agency runs several agents like AliceAgent defined in src/alice.py and BobAgent defined in src/bob.py.
# Both AliceAgent and BobAgent derived from the base class XmppAgent defined in src/xmpp_agent.py.
# What agent should we run depend on the value of the field called `protoAgent` defined in MongoDB collection 'agents'.
# The agency should be scalable and the load could be distributed between several docker containers that runs the same agency defined in Dockerfile.
# The agency should be connected to the MongoDB and take configuration of each agent from the documents defined in the 'users' collection.
# You can add other options to the schema if needed.
# We run MongoDB with `DB_URL=mongodb://mongo.dev.local:27017/selfdev`, in which the collection `agents` exists.
# The agency should read all the documents in the 'agents' collection.
# Only run the agent if its `options.schemaVersion==='0.1'` and the agent is deployed (`options.deployed===True`).
# The schema of each agent defined in agent.js in another Node.js repo with API server (I copy-pasted it below).
# Translate the schema defined on Node.js to Python:
# ```js
# export default mongoose.model(
#   'Agent', // translates to 'agents' by Mongoose driver of MongoDB
#   mongoose.Schema({
#     userId: {
#       type: ObjectId,
#       required: true,
#       ref: 'User'
#     },

#     deployed: false, // only run the agents with deployed===true

#     options: {
#       schemaVersion: String, // current schemaVersion==='0.1'

#       name: String, // unique name
#       description: String,

#       systemMessage: String, // SystemMessage(SYSTEM_MESSAGE) to pass to LLM on LangChain

#       protoAgent: String, // 'AliceAgent' or 'BobAgent' class on Python
#       joinRooms: [ String ], // XMPP rooms to join,

#       model: {
#         provider: String, // Name of the LLM model provider such as 'openai' or 'anthropic'
#         name: String, // Name of the LLM such as 'gpt-4o-mini' or 'claude-3-5-sonnet-20240620'
#       },

#       // other options will be defined here
#     },
#   })
#     .plugin(mongooseTimestamp)
# )
# ```

# The XMPP settings will be taken from the environment variables:
# ```python
# XMPP_HOST = os.getenv("XMPP_HOST", "selfdev-prosody.dev.local")
# XMPP_PASSWORD = os.getenv("XMPP_PASSWORD", "123")
# XMPP_MUC_HOST = os.getenv("XMPP_MUC_HOST", f"conference.{XMPP_HOST}")
# XMPP_JOIN_ROOMS = json.loads(os.getenv('XMPP_JOIN_ROOMS', '[ "team", "a-suite", "agents" ]'))
# ```
# Except of the XMPP_USER and XMPP_NICK, they should be redefined from the 'agents' collection document based on the `options.name`.
# XMPP_USER = os.getenv("XMPP_USER", AGENT_NAME)
# XMPP_NICK = os.getenv("XMPP_NICK", AGENT_NAME)
# ```

