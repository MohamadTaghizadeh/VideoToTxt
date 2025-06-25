import asyncio
import aio_pika
import os
import sys
from loguru import logger
from config.config_handler import config
from core.queue_utils import process_message


import argparse
from emotic import Emotic
from yolo_inference import yolo_video



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



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0, help='gpu id')
    parser.add_argument('--mode', type=str, default='train_test', choices=['train', 'test', 'train_test', 'inference','yolo_inference','video'])
    parser.add_argument('--experiment_path', type=str, required=True, help='Path to save experiment files (results, models, logs)')
    parser.add_argument('--model_dir_name', type=str, default='Models', help='Name of the directory to save models')
    parser.add_argument('--result_dir_name', type=str, default='Outputs', help='Name of the directory to save results(predictions, labels mat files)')
    parser.add_argument('--inference_file', type=str, help='Text file containing image context paths and bounding box')
    # Generate args
    args = parser.parse_args()
    return args


def check_paths(args):    

    folders= [args.result_dir_name, args.model_dir_name]
    paths = list()
    for folder in folders:
        folder_path = os.path.join(args.experiment_path, folder)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        paths.append(folder_path)

    return paths


if __name__ == "__main__":
    # Parse arguments and setup paths
    args = parse_args()
    print('mode ', args.mode)
    result_path, model_path = check_paths(args)

    # Emotion categories setup
    cat = ['Affection', 'Anger', 'Annoyance', 'Anticipation', 'Aversion', 'Confidence', 'Disapproval', 'Disconnection', 
           'Disquietment', 'Doubt/Confusion', 'Embarrassment', 'Engagement', 'Esteem', 'Excitement', 'Fatigue', 'Fear', 'Happiness', 
           'Pain', 'Peace', 'Pleasure', 'Sadness', 'Sensitivity', 'Suffering', 'Surprise', 'Sympathy', 'Yearning']
    cat2ind = {}
    ind2cat = {}
    for idx, emotion in enumerate(cat):
        cat2ind[emotion] = idx
        ind2cat[idx] = emotion

    # Normalization parameters
    context_mean = [0.4690646, 0.4407227, 0.40508908]
    context_std = [0.2514227, 0.24312855, 0.24266963]
    body_mean = [0.43832874, 0.3964344, 0.3706214]
    body_std = [0.24784276, 0.23621225, 0.2323653]
    context_norm = [context_mean, context_std]
    body_norm = [body_mean, body_std]

    # Run the main async function with the event loop
    loop = asyncio.get_event_loop()
    try:
        if args.mode == 'video':
            if args.inference_file is None:
                raise ValueError('Inference file not provided. Please pass a valid inference file for inference')
            loop.run_until_complete(yolo_video(args.inference_file, result_path, model_path, context_norm, body_norm, ind2cat, args))
        else:
            raise ValueError('Unknown mode')
    except KeyboardInterrupt:
        logger.info("Worker stopped")
    finally:
        loop.close()

