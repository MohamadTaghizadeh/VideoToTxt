#from generators import TextProcessor
import aio_pika
import json
import base64
import numpy as np
import cv2
from loguru import logger
import os
from datetime import datetime

if os.environ.get("MODE", "dev") == "prod":
    output_dir = "/approot/data/result"
else:
    output_dir = "../../../Outputs/VideoToTxt/result"
os.makedirs(output_dir, exist_ok=True)


async def process_video(encoded_video):
    logger.info("Starting Video processing with skew correction")
    try:
        # Decode base64 video
        video_data = base64.b64decode(encoded_video)
        nparr = np.frombuffer(video_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is None:
            raise ValueError("Invalid video data")

        logger.info("Video successfully decoded")

        # Create an instance of TextProcessor
        #processor = TextProcessor()

        # Process video using the process_video method
        original_video, annotated_video, text_data = processor.process_video(frame)

        # Convert text_data to a single string
        text_list = []
        for text_entry in text_data['texts']:
            text_list.append(text_entry['text'])

        # Join all text with spaces
        video_results = ' '.join(text_list)

        if not video_results.strip():
            logger.warning("No text detected in the video")
            return ""

        logger.info(f"Successfully extracted text from video")
        logger.info(f"Skew correction applied: {video_data['skew_corrected']}")
        logger.info(f"Extracted text: {video_results}")

        # Optional: save processed video
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        request_id = "video_to_text"  # This will be overridden with the actual request_id in process_message
        output_path = f"{output_dir}/{request_id}_{timestamp}.jpg"
        # Uncomment to save the processed video
        # cv2.imwrite(output_path, annotated_video)

        return video_results

    except Exception as e:
        logger.exception(f"Error processing video: {e}")
        return ""


async def process_message(
        message: aio_pika.IncomingMessage,
        result_channel: aio_pika.Channel,
):
    async with message.process():
        try:
            # Parse the message body
            message_body = json.loads(message.body.decode())
            encoded_video = message_body["video"]
            request_id = message_body["request_id"]
            priority = message_body.get("priority", 1)

            logger.info(
                "Processing Video task", request_id=request_id, priority=priority
            )

            # Publish in-progress status
            await result_channel.default_exchange.publish(
                aio_pika.Message(
                    body=json.dumps({
                        "request_id": request_id,
                        "status": "in_progress"
                    }).encode(),
                    headers={"request_id": request_id}
                ),
                routing_key="video_result_queue",
            )

            # Process the video
            video_results = await process_video(encoded_video)

            # Optional: save processed video (output path will use request_id)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"{output_dir}/{request_id}_{timestamp}.jpg"
            # Uncomment if you want to save the processed video
            # cv2.imwrite(output_path, annotated_video)

            if video_results.strip():
                # Successful processing
                result = {
                    "request_id": request_id,
                    "status": "completed",
                    "results": {"video_data": video_results},
                }
            else:
                # No text detected or processing error
                result = {
                    "request_id": request_id,
                    "status": "failed",
                    "error": "No text detected or processing error"
                }

        except Exception as e:
            logger.exception(e)
            result = {
                "request_id": request_id if 'request_id' in locals() else "unknown",
                "status": "failed",
                "error": str(e)
            }

        # Publish result
        await result_channel.default_exchange.publish(
            aio_pika.Message(
                body=json.dumps(result).encode(),
                headers={"request_id": result["request_id"]}
            ),
            routing_key="video_result_queue",
        )
