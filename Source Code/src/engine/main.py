import asyncio
import aio_pika
import os
import sys
from loguru import logger
from config.config_handler import config
from core.queue_utils import process_message

# Setup logging
if os.environ.get("MODE", "dev") == "prod":
    log_dir = "/approot/data"
else:
    log_dir = "../../../Outputs/result"
os.makedirs(log_dir, exist_ok=True)

logger.remove()
logger.add(
    sys.stderr,
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message} | {extra}",
    level=config.get("CONSOLE_LOG_LEVEL", "INFO"),
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
    level=config.get("FILE_LOG_LEVEL", "INFO"),
    backtrace=True,
    diagnose=False,
    colorize=True,
    serialize=False,
    enqueue=True,
)

async def main():
    """Main worker function that runs in a continuous loop"""
    connection = None
    try:
        # Connect to RabbitMQ
        connection = await aio_pika.connect_robust(config["QUEUE_CONNECTION"])
        logger.info("Connected to RabbitMQ")

        # Create channel
        channel = await connection.channel()
        await channel.set_qos(prefetch_count=1)

        # Declare main queue
        queue = await channel.declare_queue(
            "videototxt_queue",
            durable=True
        )

        # Declare result queue
        result_channel = await connection.channel()
        await result_channel.declare_queue(
            "emotion_detection_result_queue",  # Changed to match your use case
            durable=True
        )

        logger.info("Worker started. Waiting for messages...")

        # Continuous message processing loop
        async with queue.iterator() as queue_iter:
            async for message in queue_iter:
                try:
                    await process_message(message, result_channel)
                except Exception as e:
                    logger.exception(f"Failed to process message: {e}")
                    # Optionally: reject/nack the message if processing fails
                    await message.nack()

    except (KeyboardInterrupt, asyncio.CancelledError):
        logger.info("Worker stopping due to interrupt...")
    except Exception as e:
        logger.exception(f"Worker failed: {e}")
    finally:
        try:
            if connection and not connection.is_closed:
                await connection.close()
                logger.info("RabbitMQ connection closed")
        except Exception as e:
            logger.error(f"Error closing connection: {e}")

if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    try:
        loop.run_until_complete(main())
    except KeyboardInterrupt:
        logger.info("Worker stopped gracefully")
    finally:
        loop.close()
