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
    parser.add_argument('--experiment_path', type=str, required=True, help='Path of experiment files (results, models, logs)')
    parser.add_argument('--model_dir', type=str, default='models', help='Folder to access the models')
    parser.add_argument('--result_dir', type=str, default='results', help='Path to save the results')
    parser.add_argument('--inference_file', type=str, help='Text file containing image context paths and bounding box')
    parser.add_argument('--video_file', type=str, help='Test video file')
    # Generate args
    args = parser.parse_args()
    return args


def get_bbox(yolo_model, device, image_context, yolo_image_size=416, conf_thresh=0.8, nms_thresh=0.4):
  ''' Use yolo to obtain bounding box of every person in context image. 
  :param yolo_model: Yolo model to obtain bounding box of every person in context image. 
  :param device: Torch device. Used to send tensors to GPU (if available) for faster processing. 
  :yolo_image_size: Input image size for yolo model. 
  :conf_thresh: Confidence threshold for yolo model. Predictions with object confidence > conf_thresh are returned. 
  :nms_thresh: Non-maximal suppression threshold for yolo model. Predictions with IoU > nms_thresh are returned. 
  :return: Numpy array of bounding boxes. Array shape = (no_of_persons, 4). 
  '''
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


def yolo_infer(images_list, result_path, model_path, context_norm, body_norm, ind2cat, ind2vad, args):
  ''' Infer on a list of images defined in images_list text file to obtain bounding boxes of persons in the images using yolo model.
  :param images_list: Text file specifying the images to conduct inference. A row in the file is Path_of_image. 
  :param result_path: Directory path to save the results (images with the predicted emotion categories and continuous emotion dimesnions).
  :param model_path: Directory path to load models and val_thresholds to perform inference.
  :param context_norm: List containing mean and std values for context images. 
  :param body_norm: List containing mean and std values for body images. 
  :param ind2cat: Dictionary converting integer index to categorical emotion. 
  :param ind2vad: Dictionary converting integer index to continuous emotion dimension (Valence, Arousal and Dominance).
  :param args: Runtime arguments.
  '''
  device = torch.device("cuda:%s" %(str(args.gpu)) if torch.cuda.is_available() else "cpu")
  yolo = prepare_yolo(model_path)
  yolo = yolo.to(device)
  yolo.eval()

  thresholds = torch.FloatTensor(np.load(os.path.join(result_path, '/home/robin/M/Body-Language-and-Emotion-Recognition/deploy/api/proj/debug_exp/results/val_thresholds.npy'))).to(device) 
  model_context = torch.load(os.path.join(model_path,'/home/robin/M/Body-Language-and-Emotion-Recognition/models/model_context1.pth'), weights_only=False).to(device)
  model_body = torch.load(os.path.join(model_path,'/home/robin/M/Body-Language-and-Emotion-Recognition/models/model_body1.pth'), weights_only=False).to(device)
  emotic_model = torch.load(os.path.join(model_path,'/home/robin/M/Body-Language-and-Emotion-Recognition/models/model_emotic1.pth'), weights_only=False).to(device)
  models = [model_context, model_body, emotic_model]

  with open(images_list, 'r') as f:
    lines = f.readlines()
  
  for idx, line in enumerate(lines):
    image_context_path = line.split('\n')[0].split(' ')[0]
    image_context = cv2.cvtColor(cv2.imread(image_context_path), cv2.COLOR_BGR2RGB)
    try:
      bbox_yolo = get_bbox(yolo, device, image_context)
      for pred_bbox in bbox_yolo:
        pred_cat, pred_cont = infer(context_norm, body_norm, ind2cat, ind2vad, device, thresholds, models, image_context=image_context, bbox=pred_bbox, to_print=False)
        write_text_vad = list()
        for continuous in pred_cont:
          write_text_vad.append(str('%.1f' %(continuous)))
        write_text_vad = 'vad ' + ' '.join(write_text_vad) 
        image_context = cv2.rectangle(image_context, (pred_bbox[0], pred_bbox[1]),(pred_bbox[2] , pred_bbox[3]), (255, 0, 0), 3)
        cv2.putText(image_context, write_text_vad, (pred_bbox[0], pred_bbox[1] - 5), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)
        for i, emotion in enumerate(pred_cat):
          cv2.putText(image_context, emotion, (pred_bbox[0], pred_bbox[1] + (i+1)*12), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)
    except Exception as e:
      print ('Exception for image ',image_context_path)
      print (e)
    cv2.imwrite(os.path.join(result_path, 'img_%r.jpg' %(idx)), cv2.cvtColor(image_context, cv2.COLOR_RGB2BGR))
    print ('completed inference for image %d'  %(idx))


def yolo_video(video_file, result_path, model_path, context_norm, body_norm, ind2cat, ind2vad, args, skip_frames=9):
    ''' Perform inference on a video. First yolo model is used to obtain bounding boxes of persons in every frame.
    After that the emotic model is used to obtain categoraical and continuous emotion predictions. 
    :param video_file: Path of video file. 
    :param result_path: Directory path to save the results (output video).
    :param model_path: Directory path to load models and val_thresholds to perform inference.
    :param context_norm: List containing mean and std values for context images. 
    :param body_norm: List containing mean and std values for body images. 
    :param ind2cat: Dictionary converting integer index to categorical emotion. 
    :param ind2vad: Dictionary converting integer index to continuous emotion dimension (Valence, Arousal and Dominance).
    :param args: Runtime arguments.
    '''  
    device = torch.device("cuda:%s" %(str(args.gpu)) if torch.cuda.is_available() else "cpu")
    yolo = prepare_yolo(model_path)
    yolo = yolo.to(device)
    yolo.eval()

    thresholds = torch.FloatTensor(np.load(os.path.join(result_path, '/home/robin/M/Body-Language-and-Emotion-Recognition/deploy/api/proj/debug_exp/results/val_thresholds.npy'))).to(device) 
    model_context = torch.load(os.path.join(model_path,'/home/robin/M/Body-Language-and-Emotion-Recognition/models/model_context1.pth'), weights_only=False).to(device)
    model_body = torch.load(os.path.join(model_path,'/home/robin/M/Body-Language-and-Emotion-Recognition/models/model_body1.pth'), weights_only=False).to(device)
    emotic_model = torch.load(os.path.join(model_path,'/home/robin/M/Body-Language-and-Emotion-Recognition/models/model_emotic1.pth'), weights_only=False).to(device)
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
    
# Calculate VAD averages and percentages
    if len(frame_data['vad']['valence']) > 0:
        frame_data['vad']['valence_av'] = np.mean(frame_data['vad']['valence'])
        frame_data['vad']['arousal_av'] = np.mean(frame_data['vad']['arousal'])
        frame_data['vad']['dominance_av'] = np.mean(frame_data['vad']['dominance'])
        
        # Convert to percentage (since VAD is 0-10 scale)
        frame_data['vad']['valence_percent'] = frame_data['vad']['valence_av'] * 10
        frame_data['vad']['arousal_percent'] = frame_data['vad']['arousal_av'] * 10
        frame_data['vad']['dominance_percent'] = frame_data['vad']['dominance_av'] * 10



    # Release resources # Cleanup and save results
    writer.release()
    video_stream.release() 
    
    # Convert AVI to MP4 using FFmpeg
    try:
        import subprocess
        print("Converting AVI to MP4...")
        subprocess.run([
            'ffmpeg',
            '-i', temp_avi_path,
            '-c:v', 'libx264',
            '-preset', 'fast',
            '-crf', '23',
            '-c:a', 'aac',
            '-strict', 'experimental',
            final_mp4_path
        ], check=True)
        
        # Remove temporary AVI file
        os.remove(temp_avi_path)
        print(f"Successfully converted to MP4: {final_mp4_path}")
    except Exception as e:
        print(f"Error converting to MP4: {str(e)}")
        print(f"Original AVI file saved at: {temp_avi_path}")

    
    # After processing:
    generate_plots(frame_data, result_path)
    # Save results before exiting
    save_video_results(frame_data, result_path, ind2cat)

    print(f'Completed processing {processed_count} frames (skipped every {skip_frames} frames)')

    


def yolo_webcam(result_path, model_path, context_norm, body_norm, ind2cat, ind2vad, args):
    ''' Perform inference on webcam feed.
    :param result_path: Directory path to save the results.
    :param model_path: Directory path to load models and val_thresholds.
    :param context_norm: List containing mean and std values for context images. 
    :param body_norm: List containing mean and std values for body images. 
    :param ind2cat: Dictionary converting integer index to categorical emotion. 
    :param ind2vad: Dictionary converting integer index to continuous emotion dimension.
    :param args: Runtime arguments.
    '''
    device = torch.device("cuda:%s" %(str(args.gpu)) if torch.cuda.is_available() else "cpu")
    
    # Load models
    yolo = prepare_yolo(model_path).to(device).eval()
    thresholds = torch.FloatTensor(np.load(os.path.join(result_path, '/home/robin/M/Body-Language-and-Emotion-Recognition/deploy/api/proj/debug_exp/results/val_thresholds.npy'))).to(device) 
    model_context = torch.load(os.path.join(model_path,'/home/robin/M/Body-Language-and-Emotion-Recognition/models/model_context1.pth'), weights_only=False).to(device)
    model_body = torch.load(os.path.join(model_path,'/home/robin/M/Body-Language-and-Emotion-Recognition/models/model_body1.pth'), weights_only=False).to(device)
    emotic_model = torch.load(os.path.join(model_path,'/home/robin/M/Body-Language-and-Emotion-Recognition/models/model_emotic1.pth'), weights_only=False).to(device)

    models = [model_context, model_body, emotic_model]

    # Data storage for plots
    frame_data = {
        'categories': defaultdict(list),
        'vad': {'valence': [], 'arousal': [], 'dominance': []},
        'frame_count': 0
    }
    

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise IOError("Cannot open webcam")

    print('Starting webcam processing...')
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_data['frame_count'] += 1
            image_context = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            try:
                bbox_yolo = get_bbox(yolo, device, image_context)
                for pred_bbox in bbox_yolo:
                    pred_cat, pred_cont = infer(context_norm, body_norm, ind2cat, ind2vad, 
                                              device, thresholds, models, 
                                              image_context=image_context, bbox=pred_bbox, 
                                              to_print=False)
                    
                    # Store data for plotting
                    for cat in pred_cat:
                        frame_data['categories'][cat].append(1)
                    for cat in ind2cat.values():
                        if cat not in pred_cat:
                            frame_data['categories'][cat].append(0)
                    
                    frame_data['vad']['valence'].append(pred_cont[0])
                    frame_data['vad']['arousal'].append(pred_cont[1])
                    frame_data['vad']['dominance'].append(pred_cont[2])

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

            except Exception as e:
                print(f'Error processing frame: {e}')

            cv2.imshow('Webcam Emotion Detection', cv2.cvtColor(image_context, cv2.COLOR_RGB2BGR))
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        
        # Generate and save plots
        generate_plots(frame_data, result_path)
        
        # Save data to files
        save_data(frame_data, result_path)


def generate_plots(frame_data, result_path):
    """Generate plots for categories and VAD values."""
    # Plot categories
    plt.figure(figsize=(15, 8))
    for cat, values in frame_data['categories'].items():
        plt.plot(values, label=cat)
    plt.title('Category Detection Over Frames')
    plt.xlabel('Frame Number')
    plt.ylabel('Detection (1=Present, 0=Absent)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(result_path, 'category_detection.png'))
    plt.close()

def save_video_results(frame_data, result_path, ind2cat):
    
    """Save results with proper validation and plotting."""
    # Create directory if it doesn't exist
    os.makedirs(result_path, exist_ok=True)
    
    # Save raw data
    with open(os.path.join(result_path, 'video_results.json'), 'w') as f:
        json.dump(frame_data, f, indent=2)
    
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
        plt.savefig(os.path.join(result_path, 'vad_values.png'))
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
            plt.savefig(os.path.join(result_path, 'category_plot.png'), dpi=300)
            plt.close()
    else:
        print("No persons detected - skipping plots")




def check_paths(args):
  ''' Check (create if they don't exist) experiment directories.
  :param args: Runtime arguments as passed by the user.
  :return: result_dir_path, model_dir_path.
  '''
  if args.inference_file is not None: 
    if not os.path.exists(args.inference_file):
      raise ValueError('inference file does not exist. Please pass a valid inference file')
  if args.video_file is not None: 
    if not os.path.exists(args.video_file):
      raise ValueError('video file does not exist. Please pass a valid video file')
  if args.inference_file is None and args.video_file is None: 
    raise ValueError(' both inference file and video file can\'t be none. Please specify one and run again')
  model_path = os.path.join(args.experiment_path, args.model_dir)
  if not os.path.exists(model_path):
    raise ValueError('model path %s does not exist. Please pass a valid model_path' %(model_path))
  result_path = os.path.join(args.experiment_path, args.result_dir)
  if not os.path.exists(result_path):
    os.makedirs(result_path)
  return result_path, model_path

if __name__=='__main__':
  args = parse_args()

  result_path, model_path = check_paths(args)

  cat = ['Affection', 'Anger', 'Annoyance', 'Anticipation', 'Aversion', 'Confidence', 'Disapproval', 'Disconnection', \
          'Disquietment', 'Doubt/Confusion', 'Embarrassment', 'Engagement', 'Esteem', 'Excitement', 'Fatigue', 'Fear','Happiness', \
          'Pain', 'Peace', 'Pleasure', 'Sadness', 'Sensitivity', 'Suffering', 'Surprise', 'Sympathy', 'Yearning']
  cat2ind = {}
  ind2cat = {}
  for idx, emotion in enumerate(cat):
      cat2ind[emotion] = idx
      ind2cat[idx] = emotion
  
  vad = ['Valence', 'Arousal', 'Dominance']
  ind2vad = {}
  for idx, continuous in enumerate(vad):
      ind2vad[idx] = continuous
  
  context_mean = [0.4690646, 0.4407227, 0.40508908]
  context_std = [0.2514227, 0.24312855, 0.24266963]
  body_mean = [0.43832874, 0.3964344, 0.3706214]
  body_std = [0.24784276, 0.23621225, 0.2323653]
  context_norm = [context_mean, context_std]
  body_norm = [body_mean, body_std]

  if args.inference_file is not None: 
    print ('inference over inference file images')
    yolo_infer(args.inference_file, result_path, model_path, context_norm, body_norm, ind2cat, ind2vad, args)
  if args.video_file is not None:
    print ('inference over test video')
    yolo_video(args.video_file, result_path, model_path, context_norm, body_norm, ind2cat, ind2vad, args)
