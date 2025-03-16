#!/usr/bin/env python
import os
import multiprocessing

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_ollama import ChatOllama
from langchain_openai import AzureChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ibm import ChatWatsonx
from databricks_langchain import ChatDatabricks
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_ai21 import ChatAI21
from langchain_upstage import ChatUpstage
from langchain_xai import ChatXAI
from langchain_community.chat_models import ChatPerplexity
# from langchain_community.chat_models import ChatLlamaCpp

from langchain_openai import OpenAIEmbeddings
from langchain_ollama import OllamaEmbeddings

# providers:
#  'openai', 'anthropic', 'ollama'
# models:
#  'gpt-4o', 'gpt-4o-mini', 'claude-3-5-sonnet-20240620',
#  'medragondot/Sky-T1-32B-Preview',
#  'llama3.3', 'gemma', 'mistral', 'tinyllama'

# The following provider_list is based on the Featured Providers table:
# https://python.langchain.com/docs/integrations/chat/#featured-providers
provider_list = {
  "anthropic": {
    "requires_envs": ["ANTHROPIC_API_KEY"],
    "tool_calling": True,
    "structure_output": True,
    "json_mode": False,
    "local": False,
    "multimodal": True,
  },
  "mistralai": {
    "requires_envs": ["MISTRAL_API_KEY"],
    "tool_calling": True,
    "structure_output": True,
    "json_mode": False,
    "local": False,
    "multimodal": False,
  },
  "fireworks": {
    "requires_envs": ["FIREWORKS_API_KEY"],
    "tool_calling": True,
    "structure_output": True,
    "json_mode": True,
    "local": False,
    "multimodal": False,
  },
  "azure": {
    "requires_envs": ["AZURE_OPENAI_API_KEY",
                      "AZURE_OPENAI_ENDPOINT",
                      "AZURE_OPENAI_API_VERSION"],
    "tool_calling": True,
    "structure_output": True,
    "json_mode": True,
    "local": False,
    "multimodal": True,
  },
  "openai": {
    "requires_envs": ["OPENAI_API_KEY"],
    "tool_calling": True,
    "structure_output": True,
    "json_mode": True,
    "local": False,
    "multimodal": True,
  },
  "together": {
    "requires_envs": ["TOGETHER_API_KEY"],
    "tool_calling": True,
    "structure_output": True,
    "json_mode": True,
    "local": False,
    "multimodal": False,
  },
  "google_vertexai": {
    "requires_envs": [],
    "tool_calling": True,
    "structure_output": True,
    "json_mode": False,
    "local": False,
    "multimodal": True,
  },
  "google_genai": {
    "requires_envs": ["GOOGLE_API_KEY"],
    "tool_calling": True,
    "structure_output": True,
    "json_mode": False,
    "local": False,
    "multimodal": True,
  },
  "groq": {
    "requires_envs": ["GROQ_API_KEY"],
    "tool_calling": True,
    "structure_output": True,
    "json_mode": True,
    "local": False,
    "multimodal": False,
  },
  "cohere": {
    "requires_envs": ["COHERE_API_KEY"],
    "tool_calling": True,
    "structure_output": True,
    "json_mode": False,
    "local": False,
    "multimodal": False,
  },
  "bedrock_converse": {
    "requires_envs": [],
    "tool_calling": True,
    "structure_output": True,
    "json_mode": False,
    "local": False,
    "multimodal": False,
  },
  "huggingface": {
    "requires_envs": ["HUGGINGFACEHUB_API_TOKEN"],
    "tool_calling": True,
    "structure_output": True,
    "json_mode": False,
    "local": True,
    "multimodal": False,
  },
  "nvidia": {
    "requires_envs": ["NVIDIA_API_KEY"],
    "tool_calling": True,
    "structure_output": True,
    "json_mode": True,
    "local": True,
    "multimodal": True,
  },
  "ollama": {
    "requires_envs": [],
    "tool_calling": True,
    "structure_output": True,
    "json_mode": True,
    "local": True,
    "multimodal": False,
  },
  "ai21": {
    "requires_envs": ["AI21_API_KEY"],
    "tool_calling": True,
    "structure_output": True,
    "json_mode": False,
    "local": False,
    "multimodal": False,
  },
  "upstage": {
    "requires_envs": ["UPSTAGE_API_KEY"],
    "tool_calling": True,
    "structure_output": True,
    "json_mode": False,
    "local": False,
    "multimodal": False,
  },
  "databricks": {
    "requires_envs": ["DATABRICKS_TOKEN", "DATABRICKS_HOST"],
    "tool_calling": True,
    "structure_output": True,
    "json_mode": False,
    "local": False,
    "multimodal": False,
  },
  "ibm": {
    "requires_envs": ["WATSONX_APIKEY", "IBM_URL", "IBM_PROJECT_ID"],
    "tool_calling": True,
    "structure_output": True,
    "json_mode": True,
    "local": False,
    "multimodal": False,
  },
  "xai": {
    "requires_envs": ["XAI_API_KEY"],
    "tool_calling": True,
    "structure_output": True,
    "json_mode": False,
    "local": False,
    "multimodal": False,
  },
  "perplexity": {
    "requires_envs": ["PPLX_API_KEY"],
    "tool_calling": True,
    "structure_output": True,
    "json_mode": True,
    "local": False,
    "multimodal": True,
  },
  # "llamacpp": {
  #   "requires_envs": [],
  #   "tool_calling": True,
  #   "structure_output": True,
  #   "json_mode": False,
  #   "local": True,
  #   "multimodal": False,
  # },
}


def init_model(*, model_provider, model_name):
  """ Initialize langchain chat model """
  print('model_provider:', model_provider)
  print('model_name:', model_name)
  if model_provider not in provider_list:
    print(f"Unknown LLM provider: {model_provider}")

  print('provider attributes:', provider_list[model_provider])
  print('provider requires envs:', provider_list[model_provider]['requires_envs'])
  for env_var in provider_list[model_provider]['requires_envs']:
    if not os.getenv(env_var):
      raise ValueError(f"For {model_provider}, {env_var} is not set")

  model = None
  if model_provider == "ollama":
    model = ChatOllama(
      model=model_name,
      temperature=0,
      base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
    )
  elif model_provider == "azure":
    model = AzureChatOpenAI(
      azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
      azure_deployment=model_name,
      openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    )
  elif model_provider == "google_genai":
    model = ChatGoogleGenerativeAI(
        model=model_name,
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2,
    )
  elif model_provider == "ibm":
    model = ChatWatsonx(
      model_id=model_name,
      url=os.getenv("IBM_URL"),
      project_id=os.getenv("IBM_PROJECT_ID")
    )
  elif model_provider == "databricks":
    model = ChatDatabricks(endpoint=model_name)
  elif model_provider == "huggingface":
    llm = HuggingFaceEndpoint(
      repo_id=model_name,
      task="text-generation",
      max_new_tokens=512,
      do_sample=False,
      repetition_penalty=1.03,
    )
    model = ChatHuggingFace(llm=llm)
  elif model_provider == "ai21":
    model = ChatAI21(model="jamba-instruct", temperature=0)
  elif model_provider == "upstage":
    model = ChatUpstage()
  elif model_provider == "xai":
    model = ChatXAI(
      model=model_name,
      temperature=0,
      max_tokens=None,
      timeout=None,
      max_retries=2,
    )
  elif model_provider == "perplexity":
    model = ChatPerplexity(
      model=model_name,
      temperature=0,
    )
  # elif model_provider == "llamacpp":
  #   model = ChatLlamaCpp(
  #     temperature=0.5,
  #     model_path=model_name,
  #     n_ctx=10000,
  #     n_gpu_layers=8,
  #     n_batch=300,  # Should be between 1 and n_ctx, consider the amount of VRAM in your GPU.
  #     max_tokens=512,
  #     n_threads=multiprocessing.cpu_count() - 1,
  #     repeat_penalty=1.5,
  #     top_p=0.5,
  #     verbose=True,
  #   )
  # elif model_provider == "openai":
  #   model = ChatOpenAI(model=model_name, api_key=os.getenv("OPENAI_API_KEY"))
  # elif model_provider == "anthropic":
  #   model = ChatAnthropic(model=model_name, api_key=os.getenv("ANTHROPIC_API_KEY"))
  else:
    model = init_chat_model(model_name, model_provider=model_provider)

  print('model:', model)
  return model


def init_embeddings(*, model_provider, embeddings_name):
  """ Initialize langchain chat model """
  print('embeddings_name:', embeddings_name)

  embeddings = None
  if model_provider == "openai":
    embeddings = OpenAIEmbeddings(model=embeddings_name)
  elif model_provider == "ollama":
    embeddings = OllamaEmbeddings(model=embeddings_name,
                                  base_url=os.getenv("OLLAMA_BASE_URL",
                                                     "http://localhost:11434"))
  else:
    raise ValueError(f"Unknown Embeddings model provider: {model_provider}")

  return embeddings
