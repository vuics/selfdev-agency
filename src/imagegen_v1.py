'''
ImagegenV1 Agent Archetype
'''
# import os
import logging
# import base64

from openai import OpenAI

# from base_model import init_model
from xmpp_agent import XmppAgent
# from file_manager import FileManager

logger = logging.getLogger("ImagegenV1")


class ImagegenV1(XmppAgent):
  '''
  ImagegenV1 provides image generation
  '''
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    # self.file_manager = FileManager()

  async def start(self):
    await super().start()
    try:
      if self.config.options.imagegen.model.provider == 'openai':
        self.client = OpenAI(
          api_key=self.config.options.imagegen.model.apiKey or None,
        )
      else:
        raise Exception(f"Unknown model provider: {self.config.options.imagegen.model.provider}")
      logger.debug(f"Client initialized: {self.client}")
    except Exception as e:
      logger.error(f"Error initializing model: {e}")

  async def chat(self, *, prompt, reply_func=None):
    try:
      logger.debug(f"Received prompt: {prompt}")

      if not self.client:
        raise Exception("Client was not initialized")

      if self.config.options.imagegen.model.provider == 'openai':
        img = self.client.images.generate(
          prompt=prompt,
          model=self.config.options.imagegen.model.name,
          size=self.config.options.imagegen.size,
          n=self.config.options.imagegen.n,
          # quality=self.config.options.imagegen.quality, # only for dalle3
          # style=self.config.options.imagegen.style,     # only for dalle3
          user=f"user_{self.config.userId}",
          response_format="b64_json",
        )
        # logger.debug(f"img: {img}")

        content = ''
        for index, data_item in enumerate(img.data):
          image_base64 = data_item.b64_json
          # logger.debug(f"image_base64: {image_base64}")
          get_url = await self.upload_file(
            file_base64=image_base64,
            filename=f'image_{index}.png',
            content_type='image/png',
          )
          logger.debug(f"get_url: {get_url}")
          if reply_func:
            reply_func(get_url)
          content += f'![Generated Image]({get_url})'
        return content

      else:
        raise Exception(f"Unknown model provider: {self.config.options.imagegen.model.provider}")

      # image_bytes = base64.b64decode(img.data[0].b64_json)
      # logger.debug(f"image_bytes: {image_bytes}")
      # return image_bytes

      # if self.file_manager.is_shared_file_url(prompt):
      #   return self.file_manager.add_file_url(prompt)
      # files_info = self.file_manager.get_files_info()
      # if files_info:
      #   self.file_manager.clear()

    except Exception as e:
      logger.error(f"Imagegen error: {e}")
      return f"Error: {str(e)}"
