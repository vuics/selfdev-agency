import asyncio

class AsyncShellExecutor:
  def __init__(self, shell='/bin/sh'):
    self.shell = shell
    self.process = None

  async def start_shell(self):
    """Starts an interactive shell session."""
    self.process = await asyncio.create_subprocess_exec(
      self.shell,
      stdin=asyncio.subprocess.PIPE,
      stdout=asyncio.subprocess.PIPE,
      stderr=asyncio.subprocess.PIPE
    )

  async def send_command(self, command):
    """Sends a command to the shell process."""
    if self.process and self.process.stdin:
      self.process.stdin.write(command.encode() + b'\n')
      await self.process.stdin.drain()
    else:
      print("Shell is not running.")

  async def read_output(self):
    """Reads output from the shell asynchronously."""
    if self.process:
      while True:
        line = await self.process.stdout.readline()
        if not line:
          break
        print(line.decode(), end='')
    else:
      print("Shell is not running.")

  async def read_errors(self):
    """Reads error output from the shell asynchronously."""
    if self.process:
      while True:
        line = await self.process.stderr.readline()
        if not line:
          break
        print(line.decode(), end='')
    else:
      print("Shell is not running.")

  async def run(self):
    """Runs the interactive shell and reads input/output."""
    await self.start_shell()
    asyncio.create_task(self.read_output())
    asyncio.create_task(self.read_errors())

    while True:
      command = await asyncio.get_event_loop().run_in_executor(None, input, "$ ")
      if command.lower() in ('exit', 'quit'):
        self.process.terminate()
        break
      await self.send_command(command)

if __name__ == "__main__":
  executor = AsyncShellExecutor()
  asyncio.run(executor.run())
