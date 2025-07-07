import os
from loguru import logger
from version import __version__
from config.config_handler import config
import sys

if os.environ.get("MODE", "dev") == "prod":
    log_dir = "/approot/data"
else:
    log_dir = "../../../Outputs/result"
os.makedirs(log_dir, exist_ok=True)

logger.remove()
logger.add(
    sys.stderr,
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message} | {extra}",
    level=config["CONSOLE_LOG_LEVEL"],
    backtrace=True,
    diagnose=True,
    colorize=True,
    serialize=False,
    enqueue=True,
)
logger.add(
    f"{log_dir}/engine.log",
    rotation="50MB",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message} | {extra}",
    level=config["FILE_LOG_LEVEL"],
    backtrace=True,
    diagnose=False,
    colorize=True,
    serialize=False,
    enqueue=True,
)

logger.info("Starting service...", version=__version__)

#from generators import TTSGenerator
import aio_pika
import asyncio
from core.queue_utils import process_message


async def main():
    #tts_generator = TTSGenerator()
    connection = await aio_pika.connect_robust(config["QUEUE_CONNECTION"])

    # Create separate channels for consuming and publishing
    task_channel = await connection.channel()
    result_channel = await connection.channel()

    # Configure task queue
    task_queue = await task_channel.declare_queue("task_queue", durable=True)

    # Start consuming tasks
    await task_queue.consume(
        lambda message: process_message(message, result_channel, tts_generator)
    )

    # Keep connection alive
    try:
        await asyncio.Future()
    finally:
        await connection.close()


if __name__ == "__main__":
    asyncio.run(main())
