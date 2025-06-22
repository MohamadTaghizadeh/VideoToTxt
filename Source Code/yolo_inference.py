import argparse 
import cv2
import numpy as np 
import os 

import torch 
from torchvision import transforms

from emotic import Emotic 
from inference import infer
from yolo_utils import prepare_yolo, rescale_boxes, non_max_suppression


import json
import matplotlib.pyplot as plt
from collections import defaultdict


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0, help='gpu id')
    parser.add_argument('--video_file', type=str, help='Test video file')
    # Generate args
    args = parser.parse_args()
    return args


def get_bbox(yolo_model, device, image_context, yolo_image_size=416, conf_thresh=0.8, nms_thresh=0.4):

  test_transform = transforms.Compose([transforms.ToPILImage(),transforms.ToTensor()])
  image_yolo = test_transform(cv2.resize(image_context, (416, 416))).unsqueeze(0).to(device)

  with torch.no_grad():
    detections = yolo_model(image_yolo)
    nms_det  = non_max_suppression(detections, conf_thresh, nms_thresh)[0]
    det = rescale_boxes(nms_det, yolo_image_size, (image_context.shape[:2]))
  
  bboxes = []
  for x1, y1, x2, y2, _, _, cls_pred in det:
    if cls_pred == 0:  # checking if predicted_class = persons. 
      x1 = int(min(image_context.shape[1], max(0, x1)))
      x2 = int(min(image_context.shape[1], max(x1, x2)))
      y1 = int(min(image_context.shape[0], max(15, y1)))
      y2 = int(min(image_context.shape[0], max(y1, y2)))
      bboxes.append([x1, y1, x2, y2])
  return np.array(bboxes)




def yolo_video(video_file, result_path, model_path, context_norm, body_norm, ind2cat, ind2vad, args, skip_frames=9):
 
    device = torch.device("cuda:%s" %(str(args.gpu)) if torch.cuda.is_available() else "cpu")
    yolo = prepare_yolo(model_path)
    yolo = yolo.to(device)
    yolo.eval()

    thresholds = torch.FloatTensor(np.load(os.path.join(model_path, 'val_thresholds.npy'))).to(device) 
    model_context = torch.load(os.path.join(model_path,'model_context1.pth'), weights_only=False).to(device)
    model_body = torch.load(os.path.join(model_path,'model_body1.pth'), weights_only=False).to(device)
    emotic_model = torch.load(os.path.join(model_path,'model_emotic1.pth'), weights_only=False).to(device)
    model_context.eval()
    model_body.eval()
    emotic_model.eval()
    models = [model_context, model_body, emotic_model]

    # Enhanced data storage with averages
    frame_data = {
        
        'frame_numbers': [],
        'categories': {cat: [] for cat in ind2cat.values()},
        'vad': {
            'valence': [],
            'arousal': [],
            'dominance': [],
            'valence_av': 0,  # Will store average
            'arousal_av': 0,
            'dominance_av': 0,
            'valence_percent': 0,  # Will store percentage
            'arousal_percent': 0,
            'dominance_percent': 0
        },
        'person_detected': [],
        'processed_frames': 0,
        'skip_frames': skip_frames
    }


    video_stream = cv2.VideoCapture(video_file)
    writer = None

    # Temporary AVI file path
    temp_avi_path = os.path.join(result_path, 'temp_result.avi')
    final_mp4_path = os.path.join(result_path, 'result_vid.mp4')

    print(f'Starting testing on video with frame skipping (every {skip_frames+1} frames)')
    frame_count = 0
    processed_count = 0

    while True:
        (grabbed, frame) = video_stream.read()
        if not grabbed:
            break

        frame_count += 1

        # Skip frames according to parameter
        if skip_frames > 0 and frame_count % (skip_frames + 1) != 1:
            continue

        processed_count += 1
        frame_data['frame_numbers'].append(frame_count)
        frame_data['processed_frames'] = processed_count
        image_context = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        person_detected = False

        try: 
            bbox_yolo = get_bbox(yolo, device, image_context)
            if len(bbox_yolo) > 0:
               person_detected = True
               for pred_bbox in bbox_yolo:
                    # Enable printing for debugging
                    pred_cat, pred_cont = infer(context_norm, body_norm, ind2cat, ind2vad,
                                              device, thresholds, models,
                                              image_context=image_context,
                                              bbox=pred_bbox,
                                              to_print=True)  # Changed to True for debugging
                    
                    print(f"Frame {frame_count} - VAD: {pred_cont} - Cats: {pred_cat}")  # Debug print
                    
                    # Store results
                    frame_data['vad']['valence'].append(float(pred_cont[0]))
                    frame_data['vad']['arousal'].append(float(pred_cont[1]))
                    frame_data['vad']['dominance'].append(float(pred_cont[2]))
                    
                    # Track all categories
                    for cat in ind2cat.values():
                        frame_data['categories'][cat].append(1 if cat in pred_cat else 0)

                    # Draw on frame
                    write_text_vad = 'vad ' + ' '.join([f'{v:.1f}' for v in pred_cont])
                    image_context = cv2.rectangle(image_context, 
                                                (pred_bbox[0], pred_bbox[1]),
                                                (pred_bbox[2], pred_bbox[3]), 
                                                (255, 0, 0), 3)
                    cv2.putText(image_context, write_text_vad, 
                               (pred_bbox[0], pred_bbox[1] - 5), 
                               cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)
                    for i, emotion in enumerate(pred_cat):
                        cv2.putText(image_context, emotion, 
                                   (pred_bbox[0], pred_bbox[1] + (i+1)*12), 
                                   cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)
                    

            # Handle frames without detections
            frame_data['person_detected'].append(person_detected)
            # Ensure we have data for all frames
            if not person_detected:
                frame_data['vad']['valence'].append(0.0)
                frame_data['vad']['arousal'].append(0.0)
                frame_data['vad']['dominance'].append(0.0)
                for cat in ind2cat.values():
                    frame_data['categories'][cat].append(0)
                    
        except Exception as e:
            print(f"Error processing frame {frame_count}: {str(e)}")
            frame_data['person_detected'].append(False)
            frame_data['vad']['valence'].append(0.0)
            frame_data['vad']['arousal'].append(0.0)
            frame_data['vad']['dominance'].append(0.0)
            for cat in ind2cat.values():
                frame_data['categories'][cat].append(0)

        # Write frame to video    
        if writer is None:
            # Use MJPG codec for AVI which is widely supported
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            writer = cv2.VideoWriter(temp_avi_path, fourcc, 30, 
                                    (image_context.shape[1], image_context.shape[0]), True)  
        
        writer.write(cv2.cvtColor(image_context, cv2.COLOR_RGB2BGR))
    


    # Release resources # Cleanup and save results
    writer.release()
    video_stream.release() 
    

    
    # Save results before exiting
    save_video_results(frame_data, result_path, ind2cat)
    # After processing:
    #generate_plots(frame_data, result_path, ind2cat)

    print(f'Completed processing {processed_count} frames (skipped every {skip_frames} frames)')



def save_video_results(frame_data, result_path, ind2cat):
    
    """Save results with proper validation and plotting."""
    # Create directory if it doesn't exist
    os.makedirs(result_path, exist_ok=True)
    
    # Calculate category percentages
    total_frames = len(frame_data['frame_numbers'])
    category_percent = {}

    for cat in ind2cat.values():
        if total_frames > 0:
            detection_count = sum(frame_data['categories'][cat])
            category_percent[cat] = (detection_count / total_frames) * 100
        else:
            category_percent[cat] = 0.0

    # Add to frame_data
    
    frame_data['category_percent'] = category_percent

    # Save raw data (JSON)
    with open(os.path.join(result_path, 'video_results.json'), 'w') as f:
        json.dump(frame_data, f, indent=2)
    
    # Generate plots
    #generate_plots(frame_data, result_path, ind2cat)

    # Calculate detection rate
    detection_rate = sum(frame_data['person_detected'])/len(frame_data['person_detected'])
    print(f"Person detection rate: {detection_rate*100:.1f}%")
    

    # Only plot if we have detections
    if sum(frame_data['person_detected']) > 0:
        # VAD Plot
        plt.figure(figsize=(12, 6))
        plt.plot(frame_data['frame_numbers'], frame_data['vad']['valence'], label='Valence')
        plt.plot(frame_data['frame_numbers'], frame_data['vad']['arousal'], label='Arousal')
        plt.plot(frame_data['frame_numbers'], frame_data['vad']['dominance'], label='Dominance')
        plt.xlabel('Frame Number')
        plt.ylabel('Value')
        plt.title('VAD Values Over Frame')
        plt.legend()
        #plt.savefig(os.path.join(result_path, 'vad_values.png'))
        plt.close()

        # Category Plot (only show detected categories)
        detected_cats = [cat for cat in ind2cat.values() 
                        if sum(frame_data['categories'][cat]) > 0]
        if detected_cats:
            plt.figure(figsize=(15, 6))
            for cat in detected_cats:
                plt.plot(frame_data['frame_numbers'], 
                        frame_data['categories'][cat], 
                        label=cat)
            plt.xlabel('Frame Number')
            plt.ylabel('Detection (1=Present)')
            plt.title('Category Detection')
            plt.legend(bbox_to__anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            #plt.savefig(os.path.join(result_path, 'category_plot.png'), dpi=300)
            plt.close()
    else:
        print("No persons detected - skipping plots")

