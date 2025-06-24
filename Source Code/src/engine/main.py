import asyncio
import aio_pika
import os
import sys
from loguru import logger
from config.config_handler import config
from core.queue_utils import process_message

# Set up logging directories
if os.environ.get("MODE", "dev") == "prod":
    log_dir = "/approot/data"
else:
    log_dir = "../../../../Outputs/result"
os.makedirs(log_dir, exist_ok=True)

# Configure logging
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
    """
    Main function for the video to text engine
    """
    try:
        # Connect to RabbitMQ
        connection = await aio_pika.connect_robust(config["QUEUE_CONNECTION"])
        logger.info(f"Connected to RabbitMQ: {connection}")

        # Create channels
        channel = await connection.channel()
        result_channel = await connection.channel()

        # Set QoS
        await channel.set_qos(prefetch_count=1)

        # Declare queues
        queue = await channel.declare_queue(
            "videototxt_queue",
            durable=True,
            arguments={"x-max-priority": 10}
        )

        # Declare result queue
        await result_channel.declare_queue(
            "videototxt_result_queue",
            durable=True
        )

        logger.info("Started worker. Waiting for messages...")

        # Process messages
        async with queue.iterator() as queue_iter:
            async for message in queue_iter:
                try:
                    await process_message(message, result_channel)
                except Exception as e:
                    logger.exception(f"Error processing message: {e}")

    except (KeyboardInterrupt, asyncio.CancelledError):
        logger.info("Worker stopping due to interrupt...")
    except Exception as e:
        logger.exception(f"Error in main: {e}")
    finally:
        try:
            if 'connection' in locals() and connection and not connection.is_closed:
                await connection.close()
                logger.info("RabbitMQ connection closed")
        except Exception as e:
            logger.error(f"Error closing connection: {e}")


if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    try:
        loop.run_until_complete(main())
    except KeyboardInterrupt:
        logger.info("Worker stopped")
    finally:
        loop.close()
