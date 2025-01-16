'''
Selfdev Agency Setup
'''
import setuptools

with open('README.md', 'r') as fh:
  long_description = fh.read()

setuptools.setup(
  name='selfdev-agency',
  version='1.0.0',
  author='Artem Arakcheev',
  author_email='artarakcheev@gmail.com',
  description='Selfdev Agency microservices',
  long_description=long_description,
  long_description_content_type='text/markdown',
  url='https://github.com/ai-az1/selfdev-agency.git',
  packages=setuptools.find_packages(),
  classifiers=[
    'Programming Language :: Python :: 3',
    'Operating System :: OS Independent',
  ],
  python_requires='>=3.10',
  install_requires=[
    'python-dotenv==1.0.0',
    # 'Flask==3.0.0',
    # 'Flask-Cors==4.0.0',
    # 'requests==2.31.0',
    # 'six==1.16.0',
    'watchdog==3.0.0',
    'watchdog[watchmedo]>=0.10.2',
    'safety==2.3.5',
    # 'openai==1.58.1',
    # 'waitress==2.1.2',
    'autogen-core==0.4.2'
    'autogen-agentchat==0.4.2',
    'autogen-ext[openai]==0.4.2',
    'autogen-ext[grpc]==0.4.2',
  ],
)
