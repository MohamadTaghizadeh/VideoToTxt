import argparse
import os

from emotic import Emotic
from yolo_inference import yolo_video


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


if __name__ == '__main__':
    args = parse_args()
    print ('mode ', args.mode)

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

    if args.mode == 'video':
        if args.inference_file is None:
            raise ValueError('Inference file not provided. Please pass a valid inference file for inference')
        yolo_video(args.inference_file, result_path, model_path, context_norm, body_norm, ind2cat, ind2vad, args)

    else:
        raise ValueError('Unknown mode')

