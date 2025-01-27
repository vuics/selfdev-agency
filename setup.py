'''
Selfdev Agency Setup
'''
import setuptools

setuptools.setup(
  name='selfdev-agency',
  version='1.0.0',
  author='Artem Arakcheev',
  author_email='artarakcheev@gmail.com',
  description='Selfdev Agency microservices',
  url='https://github.com/ai-az1/selfdev-agency.git',
  packages=setuptools.find_packages(),
  classifiers=[
    'Programming Language :: Python :: 3',
    'Operating System :: OS Independent',
  ],
  python_requires='>=3.10',
  install_requires=[
    'python-dotenv==1.0.0',
    'fastapi==0.115.6',
    'pydantic==2.10.5',
    'uvicorn==0.34.0',
    'httpx==0.27.2',
    'waitress==2.1.2',
    'redis==5.2.1',
    'langchain==0.3.14',
    'langchain-core==0.3.30',
    'langchain-community==0.3.14',
    'langgraph==0.2.64',
    'langchain-openai==0.3.0',
    'langchain-ollama==0.2.2',
    'langchain-openai==0.3.0',
    'langchain-anthropic==0.3.3',
    'langchain-text-splitters==0.3.5',
    'langchain-community==0.3.14',
    'langchain-google-community[drive]==2.0.4',
    'langchain-googledrive==0.3.1',
    'tiktoken==0.8.0',
    'unstructured==0.16.13',
    'unstructured-client==0.29.0',
    'langchain-unstructured==0.1.6',
    'unstructured[all-docs]==0.16.13',
    'libmagic==1.0',
    'faiss-cpu==1.9.0.post1',
    'google-api-python-client==2.159.0',
    'google-auth-httplib2==0.2.0',
    'google-auth-oauthlib==1.2.1',

    'python-magic==0.4.27',

    # 'autogen-core==0.4.2',
    # 'autogen-agentchat==0.4.2',
    # 'autogen-ext[openai]==0.4.2',
    # 'autogen-ext[grpc]==0.4.2',
    #
    # 'openai==1.58.1',
  ],
)
