'''
FileManager

The file manager downloads file from urls every time it receives it in prompt method.
If the prompt contains only url, then the file manager would download the file
and store it in the memory together with other downloaded files.
If there is a message in prompt and not the url, then the chat would send all
the files in human message from the file manager memory and clear the memory.
So next time the user will send new downloaded files with the new message.
'''
import logging
import re
import os
import base64

from dotenv import load_dotenv
import httpx
import magic

logger = logging.getLogger("FileManager")

load_dotenv()

SHARE_URL_PREFIX = os.getenv("SHARE_URL_PREFIX", "https://selfdev-prosody.dev.local:5281/file_share")
SHARE_URL_REGEX = re.compile(f"^{re.escape(SHARE_URL_PREFIX)}")


class FileManager:
  '''
  Manages files downloaded from URLs in memory
  '''
  def __init__(self):
    self.files = []

  def is_file_url(self, prompt):
    try:
      if re.match(SHARE_URL_REGEX, prompt):
        logger.debug("Prompt is a file URL.")
        # logger.debug("Prompt is a file URL; downloading and storing.")
        # self.file_manager.add_file_from_url(prompt)
        return True
      return False
    except Exception as e:
      logger.error(f"Error checking prompt {prompt} to match url pattern: {e}")

  def add_file_from_url(self, url):
    try:
      logger.debug(f"Downloading file from {url}")
      # FIXME: Set verify=True to check the certificates
      response = httpx.get(url, verify=False)
      response.raise_for_status()
      file_bytes = response.content

      mime = magic.Magic(mime=True)
      mime_type = mime.from_buffer(file_bytes)
      type_part = mime_type.split("/")[0]
      if type_part in ["image", "audio", "text"]:
        type = type_part
      else:
        type = "file"

      file_base64 = base64.b64encode(file_bytes).decode("utf-8")
      file_info = {
        "type": type,
        "source_type": "base64",
        "data": file_base64,
        "mime_type": mime_type,
        "url": url,
      }
      self.files.append(file_info)
      logger.debug(f"File downloaded and added, MIME type: {mime_type}")
    except Exception as e:
      logger.error(f"Error downloading file from {url}: {e}")

  def get_all_files(self):
    return self.files

  def clear(self):
    self.files = []
