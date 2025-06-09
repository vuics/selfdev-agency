'''
TtsV1 Agent Archetype
'''
import logging

from openai import OpenAI

from xmpp_agent import XmppAgent

logger = logging.getLogger("TtsV1")


class TtsV1(XmppAgent):
  '''
  TtsV1 provides speech synthesis from text to audio
  '''
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    # self.file_manager = FileManager()

  async def start(self):
    await super().start()
    try:
      if self.config.options.tts.model.provider == 'openai':
        self.client = OpenAI(
          api_key=self.config.options.tts.model.apiKey or None,
        )
      else:
        raise Exception(f"Unknown model provider: {self.config.options.tts.model.provider}")
      logger.debug(f"Client initialized: {self.client}")
    except Exception as e:
      logger.error(f"Error initializing model: {e}")

  async def chat(self, *, prompt, reply_func=None):
    try:
      logger.debug(f"Received prompt: {prompt}")

      if not self.client:
        raise Exception("Client was not initialized")

      if self.config.options.tts.model.provider == 'openai':
        params = {
          "prompt": prompt,
          "model": self.config.options.tts.model.name,
          "size": self.config.options.tts.size,
          "n": self.config.options.tts.n,
          "user": f"user_{self.config.userId}",
          "response_format": "b64_json",
        }
        if self.config.options.tts.model.name == "dall-e-3":
          params["quality"] = self.config.options.tts.quality
          params["style"] = self.config.options.tts.style

        img = self.client.images.generate(**params)
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
          logger.info(f"get_url: {get_url}")
          if reply_func:
            reply_func(get_url)
          content += f'![Generated Image]({get_url})'
        return content

      else:
        raise Exception(f"Unknown model provider: {self.config.options.tts.model.provider}")

      # image_bytes = base64.b64decode(img.data[0].b64_json)
      # logger.debug(f"image_bytes: {image_bytes}")
      # return image_bytes

      # if self.file_manager.is_shared_file_url(prompt):
      #   return self.file_manager.add_file_url(prompt)
      # files_info = self.file_manager.get_files_info()
      # if files_info:
      #   self.file_manager.clear()

    except Exception as e:
      logger.error(f"Tts error: {e}")
      return f"Error: {str(e)}"
