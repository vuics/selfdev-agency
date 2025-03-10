'''
NotebookV1 Agent Archetype
'''
import os
import logging
import uuid
import sys
import asyncio
import copy

import papermill as pm
import tempfile
import nbconvert
import nbformat
import io
import concurrent.futures

from xmpp_agent import XmppAgent
from helpers import extract_and_parse_json

logger = logging.getLogger("NotebookV1")


class NotebookV1(XmppAgent):
  '''
  NotebookV1 provides exectution of Jupyter Notebooks and their outputs
  '''
  async def start(self):
    pass

  async def run_papermill(self, *, notebook_path, parameters=None,
                          kernel_name='python3'):
    """
    Runs a Jupyter notebook with papermill and nbconvert Python APIs
    asynchronously and captures print outputs

    Args:
      notebook_path: Path to the .ipynb file
      parameters: Dictionary of parameters to pass to the notebook
    """
    if parameters is None:
      parameters = {}
    logger.debug(f'parameters: {parameters}')

    # Create temporary file for the executed notebook
    # temp_output = os.path.join(tempfile.gettempdir(), "output.ipynb")
    temp_output = os.path.join(tempfile.gettempdir(), f"temp_output_{str(uuid.uuid4())}.ipynb")
    logger.debug(f'temp_output: {temp_output}')

    # Execute papermill in a thread pool to avoid blocking
    loop = asyncio.get_event_loop()
    with concurrent.futures.ThreadPoolExecutor() as pool:
      await loop.run_in_executor(
        pool,
        lambda: pm.execute_notebook(
          notebook_path,
          temp_output,
          kernel_name=kernel_name,
          parameters=parameters
        )
      )

    # Read the notebook file in a non-blocking way
    with open(temp_output, 'r', encoding='utf-8') as f:
      notebook_content = f.read()
    logger.debug(f'notebook_content: {notebook_content}')

    notebook = nbformat.reads(notebook_content, as_version=4)
    logger.debug(f'notebook: {notebook}')

    # Create a Python exporter
    python_exporter = nbconvert.PythonExporter()

    # Convert the notebook to Python (run in thread pool)
    with concurrent.futures.ThreadPoolExecutor() as pool:
      python_code, _ = await loop.run_in_executor(
        pool,
        lambda: python_exporter.from_notebook_node(notebook)
      )

    # Execute the Python code in a separate thread to avoid blocking
    def execute_and_capture():
      # Redirect stdout to capture print outputs
      old_stdout = sys.stdout
      new_stdout = io.StringIO()
      sys.stdout = new_stdout

      logger.debug(f'python_code: {python_code}')
      try:
        exec(python_code)
        return new_stdout.getvalue()
      finally:
        # Restore stdout
        sys.stdout = old_stdout

    with concurrent.futures.ThreadPoolExecutor() as pool:
      output = await loop.run_in_executor(pool, execute_and_capture)
    logger.debug(f'output: {output}')

    # Clean up the temporary file
    try:
      os.remove(temp_output)
    except Exception as e:
      logger.warning(f'Clean up the temporary file {temp_output} error: {e}')

    return output

  async def chat(self, *, prompt):
    try:
      logger.debug(f"prompt: {prompt}")
      logger.debug(f'self.options: {self.options}')
      logger.debug(f"self.options.notebook: {self.options.notebook}")
      parameters = {}
      if hasattr(self.options.notebook, 'parameters'):
        parameters = copy.deepcopy(self.options.notebook.parameters)
      if hasattr(self.options.notebook, 'parseJson') and self.options.notebook.parseJson:
        try:
          parsed_json = extract_and_parse_json(prompt)
          logger.debug(f'parsed_json: {parsed_json}')
          parameters.update(parsed_json)
        except Exception:
          pass
      if hasattr(self.options.notebook, 'promptKey'):
        parameters.update({self.options.notebook.promptKey: prompt})
      output = await self.run_papermill(
        notebook_path=self.options.notebook.filePath,
        parameters=parameters,
        kernel_name=self.options.notebook.kernelName if hasattr(self.options.notebook, 'kernelName') else None,
      )

      # TODO: Integrate with Scrapbook?
      # https://pypi.org/project/scrapbook/

      logger.debug(f"output: {output}")
      return output
    except Exception as e:
      logger.error(f"Chat error: {e}")
      return f'Error: {str(e)}'
