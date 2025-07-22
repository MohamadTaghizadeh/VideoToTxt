import os
import torch
import cv2
import numpy as np
from torchvision import transforms
import torch.nn as nn 

# Emotion categories
CATEGORIES = [
    'Affection', 'Anger', 'Annoyance', 'Anticipation', 'Aversion', 'Confidence',
    'Disapproval', 'Disconnection', 'Disquietment', 'Doubt/Confusion', 'Embarrassment',
    'Engagement', 'Esteem', 'Excitement', 'Fatigue', 'Fear', 'Happiness', 'Pain',
    'Peace', 'Pleasure', 'Sadness', 'Sensitivity', 'Suffering', 'Surprise',
    'Sympathy', 'Yearning'
]

class Emotic(nn.Module):
  ''' Model'''
  def __init__(self, num_context_features, num_body_features):
    super(Emotic,self).__init__()
    self.num_context_features = num_context_features
    self.num_body_features = num_body_features
    self.fc1 = nn.Linear((self.num_context_features + num_body_features), 256)
    self.bn1 = nn.BatchNorm1d(256)
    self.d1 = nn.Dropout(p=0.5)
    self.fc_cat = nn.Linear(256, 26)
    self.fc_cont = nn.Linear(256, 3)
    self.relu = nn.ReLU()

    
  def forward(self, x_context, x_body):
    context_features = x_context.view(-1, self.num_context_features)
    body_features = x_body.view(-1, self.num_body_features)
    fuse_features = torch.cat((context_features, body_features), 1)
    fuse_out = self.fc1(fuse_features)
    fuse_out = self.bn1(fuse_out)
    fuse_out = self.relu(fuse_out)
    fuse_out = self.d1(fuse_out)    
    cat_out = self.fc_cat(fuse_out)
    cont_out = self.fc_cont(fuse_out)
    return cat_out, cont_out


class EmotionDetector:
    def __init__(self, gpu=0):
        self.device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
        
        # Define paths
        if os.environ.get("MODE", "dev") == "prod":
            model_dir = "/approot/models"
        else:
            model_dir = os.path.normpath("../../../Models")
        
        # Load thresholds
        self.thresholds = torch.FloatTensor(np.load(
            os.path.join(model_dir, 'val_thresholds.npy')
        )).to(self.device)
        
        # Load models using torch.load with weights_only=False
        # We'll load the state_dict directly instead of the full model
        self.model_context = self._load_model(os.path.join(model_dir, 'model_context1.pth'))
        self.model_body = self._load_model(os.path.join(model_dir, 'model_body1.pth'))
        self.emotic_model = self._load_model(os.path.join(model_dir, 'model_emotic1.pth'))

        # Normalization parameters
        self.context_mean = [0.4690646, 0.4407227, 0.40508908]
        self.context_std = [0.2514227, 0.24312855, 0.24266963]
        self.body_mean = [0.43832874, 0.3964344, 0.3706214]
        self.body_std = [0.24784276, 0.23621225, 0.2323653]

    def _load_model(self, model_path):
        """Helper function to load models without requiring original class definitions"""
        try:
            # First, try loading with weights_only=True to avoid class issues
            model_data = torch.load(model_path, map_location=self.device, weights_only=True)
            
            # Create a simple container and load the state dict
            model = torch.nn.Sequential()
            model.load_state_dict(model_data)
            return model.eval().to(self.device)
            
        except Exception as e1:
            try:
                # If that fails, try loading the full model but handle missing classes
                model_data = torch.load(model_path, map_location=self.device, weights_only=False)
                
                # If it's a dict with state_dict
                if isinstance(model_data, dict) and 'state_dict' in model_data:
                    model = torch.nn.Sequential()
                    model.load_state_dict(model_data['state_dict'])
                elif isinstance(model_data, dict):
                    model = torch.nn.Sequential()
                    model.load_state_dict(model_data)
                else:
                    # Try to use the model directly
                    model = model_data
                
                return model.eval().to(self.device)
                
            except Exception as e2:
                logger.error(f"Failed to load model {model_path}. Error 1: {e1}, Error 2: {e2}")
                # Return a dummy model as fallback
                return torch.nn.Sequential(torch.nn.Linear(1, 1)).eval().to(self.device)

    def get_bbox(self, yolo_model, device, image_context, yolo_image_size=416, conf_thresh=0.8, nms_thresh=0.4):
        """Your original YOLO bounding box detection code"""
        test_transform = transforms.Compose([transforms.ToPILImage(), transforms.ToTensor()])
        image_yolo = test_transform(cv2.resize(image_context, (416, 416))).unsqueeze(0).to(device)

        with torch.no_grad():
            detections = yolo_model(image_yolo)
            nms_det = non_max_suppression(detections, conf_thresh, nms_thresh)[0]
            det = rescale_boxes(nms_det, yolo_image_size, (image_context.shape[:2]))
        
        bboxes = []
        for x1, y1, x2, y2, _, _, cls_pred in det:
            if cls_pred == 0:  # checking if predicted_class = persons
                x1 = int(min(image_context.shape[1], max(0, x1)))
                x2 = int(min(image_context.shape[1], max(x1, x2)))
                y1 = int(min(image_context.shape[0], max(15, y1)))
                y2 = int(min(image_context.shape[0], max(y1, y2)))
                bboxes.append([x1, y1, x2, y2])
        return np.array(bboxes)

    def detect_persons(self, image):
        """Your original YOLO-based person detection"""
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return self.get_bbox(self.yolo, self.device, image_rgb)

    def preprocess_images(self, context_img, body_img):
        """Your original image preprocessing"""
        context_img = cv2.resize(context_img, (224, 224))
        body_img = cv2.resize(body_img, (128, 128))
        
        test_transform = transforms.Compose([transforms.ToPILImage(), transforms.ToTensor()])
        
        context_norm = transforms.Normalize(self.context_mean, self.context_std)  
        body_norm = transforms.Normalize(self.body_mean, self.body_std)

        image_context = context_norm(test_transform(context_img)).unsqueeze(0).to(self.device)
        image_body = body_norm(test_transform(body_img)).unsqueeze(0).to(self.device)

        return image_context, image_body

    def predict_emotion(self, context_img, body_img):
        """Your original emotion prediction"""
        image_context, image_body = self.preprocess_images(context_img, body_img)
        
        with torch.no_grad():
            pred_context = self.model_context(image_context)
            pred_body = self.model_body(image_body)
            pred_cat, _ = self.emotic_model(pred_context, pred_body)
            pred_cat = pred_cat.squeeze(0)
            bool_cat_pred = torch.gt(pred_cat, self.thresholds)
        
        return [CATEGORIES[i] for i in range(len(bool_cat_pred)) if bool_cat_pred[i]]

    def process_frame(self, frame):
        """Process single frame with your original logic"""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        detections = []
        
        bboxes = self.detect_persons(frame_rgb)
        if len(bboxes) > 0:
            for pred_bbox in bboxes:
                emotions = self.predict_emotion(frame_rgb, frame_rgb[pred_bbox[1]:pred_bbox[3], pred_bbox[0]:pred_bbox[2]])
                detections.append({
                    "bounding_box": pred_bbox,
                    "emotions": emotions
                })
        
        return detections

    def process_video(self, video_path, skip_frames=9):
        """Your original video processing logic"""
        cap = cv2.VideoCapture(video_path)
        frame_count = 0
        emotion_counts = {cat: 0 for cat in CATEGORIES}
        total_detections = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            if skip_frames > 0 and frame_count % (skip_frames + 1) != 1:
                continue

            detections = self.process_frame(frame)
            for detection in detections:
                total_detections += 1
                for emotion in detection["emotions"]:
                    emotion_counts[emotion] += 1

        cap.release()

        # Calculate percentages
        results = {
            "total_frames": frame_count,
            "frames_processed": frame_count // (skip_frames + 1),
            "emotion_percentages": {
                cat: (count / total_detections * 100) if total_detections > 0 else 0
                for cat, count in emotion_counts.items()
            }
        }
        
        return results

# Add your YOLO utility functions directly in this file
def xywh2xyxy(x):
    y = x.new(x.shape)
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y

def non_max_suppression(prediction, conf_thres=0.5, nms_thres=0.4):
    prediction[..., :4] = xywh2xyxy(prediction[..., :4])
    output = [None for _ in range(len(prediction))]
    
    for image_i, image_pred in enumerate(prediction):
        image_pred = image_pred[image_pred[:, 4] >= conf_thres]
        if not image_pred.size(0):
            continue
            
        score = image_pred[:, 4] * image_pred[:, 5:].max(1)[0]
        image_pred = image_pred[(-score).argsort()]
        class_confs, class_preds = image_pred[:, 5:].max(1, keepdim=True)
        detections = torch.cat((image_pred[:, :5], class_confs.float(), class_preds.float()), 1)
        
        keep_boxes = []
        while detections.size(0):
            large_overlap = bbox_iou(detections[0, :4].unsqueeze(0), detections[:, :4]) > nms_thres
            label_match = detections[0, -1] == detections[:, -1]
            invalid = large_overlap & label_match
            weights = detections[invalid, 4:5]
            detections[0, :4] = (weights * detections[invalid, :4]).sum(0) / weights.sum()
            keep_boxes += [detections[0]]
            detections = detections[~invalid]
            
        if keep_boxes:
            output[image_i] = torch.stack(keep_boxes)
    
    return output

def bbox_iou(box1, box2, x1y1x2y2=True):
    if not x1y1x2y2:
        b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
        b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
        b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
        b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2
    else:
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * torch.clamp(
        inter_rect_y2 - inter_rect_y1 + 1, min=0)
    
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)
    
    return inter_area / (b1_area + b2_area - inter_area + 1e-16)

def rescale_boxes(boxes, current_dim, original_shape):
    orig_h, orig_w = original_shape
    pad_x = max(orig_h - orig_w, 0) * (current_dim / max(original_shape))
    pad_y = max(orig_w - orig_h, 0) * (current_dim / max(original_shape))
    unpad_h = current_dim - pad_y
    unpad_w = current_dim - pad_x
    
    boxes[:, 0] = ((boxes[:, 0] - pad_x // 2) / unpad_w) * orig_w
    boxes[:, 1] = ((boxes[:, 1] - pad_y // 2) / unpad_h) * orig_h
    boxes[:, 2] = ((boxes[:, 2] - pad_x // 2) / unpad_w) * orig_w
    boxes[:, 3] = ((boxes[:, 3] - pad_y // 2) / unpad_h) * orig_h
    
    return boxes
