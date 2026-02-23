import torch
import os
import random
import logging
import numpy as np

from torch.nn import functional as F
from scipy.optimize import linear_sum_assignment as linear_assignment
from datetime import datetime
from tensorboardX import SummaryWriter


def select_confident_samples(model, dataloader, num_samples, from_image=True):

    logits_list = []
    image_id_list = []

    for images, _, image_id, base_token, tag_token, attr_token, _ in dataloader:
        image_id_list.append(image_id)
        images = images.cuda(non_blocking=True)

        with torch.no_grad():
            if from_image:
                logits, _, _, _ = model(images, base_token, tag_token, attr_token, is_eval=True)
            else:
                _, logits, _, _ = model(images, base_token, tag_token, attr_token, is_eval=True)
        
        logits_list.append(logits)

    # Apply softmax with temperature scaling to logits to highlight the dominant class.
    logits_all = torch.cat(logits_list, dim=0)
    logits_all = F.softmax(logits_all / 0.01, dim=-1)

    # Select the indices of the top 'num_samples' most confident samples for each class based on their logits.
    top_k_per_cls = [logits_all[:, i].argsort(descending=True)[:num_samples] for i in range(logits_all.shape[1])]

    # Initialize an empty dictionary to store confident samples.
    confident_samples_map = {}
    # Flatten the list of image IDs from nested lists to a single list.
    image_id_flattened = [img_id for sublist in image_id_list for img_id in sublist]
    for idx, image_indices in enumerate(top_k_per_cls):
        for image_index in image_indices:
            image_id = image_id_flattened[image_index]
            # Map image ID of the confident sample to its corresponding class index.
            confident_samples_map[image_id] = idx

    return confident_samples_map


def evaluate_accuracy(preds, targets, mask):
    mask = mask.astype(bool)
    targets = targets.astype(int)
    preds = preds.astype(int)

    old_classes_gt = set(targets[mask])
    new_classes_gt = set(targets[~mask])

    assert preds.size == targets.size
    D = max(preds.max(), targets.max()) + 1
    w = np.zeros((D, D), dtype=int)
    for i in range(preds.size):
        w[preds[i], targets[i]] += 1

    ind = linear_assignment(w.max() - w)
    ind = np.vstack(ind).T

    ind_map = {j: i for i, j in ind}
    total_acc = sum([w[i, j] for i, j in ind])
    total_instances = preds.size

    total_acc /= total_instances

    old_acc = 0
    total_old_instances = 0
    for i in old_classes_gt:
        old_acc += w[ind_map[i], i]
        total_old_instances += sum(w[:, i])
    old_acc /= total_old_instances

    new_acc = 0
    total_new_instances = 0
    for i in new_classes_gt:
        new_acc += w[ind_map[i], i]
        total_new_instances += sum(w[:, i])
    new_acc /= total_new_instances

    return total_acc, old_acc, new_acc


# Set random seeds for reproducibility
def set_random_seeds(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


# Set up directories for the experiment
def set_experiment_directories(experiment_name, output_dir):
    current_time = datetime.now().strftime('%Y-%m-%d-%H-%M')
    experiment_dir = os.path.join(output_dir, f"{experiment_name}-{current_time}")
    os.makedirs(experiment_dir, exist_ok=True)
    
    subdirs = ['logs', 'models', 'tensorboard_logs']
    for subdir in subdirs:
        os.makedirs(os.path.join(experiment_dir, subdir), exist_ok=True)
    return experiment_dir


# Set up logging configuration
def set_logging(log_path):
    logging.basicConfig(filename=log_path, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger()
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger


# Set computation device (GPU or CPU)
def set_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


# Set up TensorBoard for logging metrics
def set_tensorboard(tensorboard_log_dir, args):
    writer = SummaryWriter(tensorboard_log_dir)
    args_str = '\n '.join(f'{k}={v}' for k, v in vars(args).items())
    writer.add_text('Arguments', args_str)
    return writer


# Log initial experiment settings
def log_experiment_settings(logger, args):
    action = "Training" if not args.evaluate else "Evaluating"
    logger.info(f"{action} {args.experiment_name} with the following settings:")
    args_str = '\n '.join(f'{k}={v}' for k, v in vars(args).items())
    logger.info(f'Command-line arguments: {args_str}')


# validate evaluation path
def validate_evaluation_path(evaluate_path):
    essential_paths = [
        evaluate_path,
        os.path.join(evaluate_path, 'logs', 'log.txt'),
        os.path.join(evaluate_path, 'models', 'model.pth')
    ]
    for path in essential_paths:
        if not os.path.exists(path):
            raise ValueError(f"Invalid path: {path}. Please provide a valid path for evaluation.")


# Initialize the experiment environment
def init_experiment(args):
    set_random_seeds(args.seed)
    
    if not args.evaluate:
        if hasattr(args, 'interrupted_path') and os.path.exists(args.interrupted_path):
            experiment_dir = args.interrupted_path
        else:
            experiment_dir = set_experiment_directories(args.experiment_name, args.output_dir)
        
        args.log_path = os.path.join(experiment_dir, 'logs', 'log.txt')
        args.model_path = os.path.join(experiment_dir, 'models', 'model.pth')
        tensorboard_log_dir = os.path.join(experiment_dir, 'tensorboard_logs')
    else:
        validate_evaluation_path(args.evaluate_path)
        args.log_path = os.path.join(args.evaluate_path, 'logs', 'log.txt')
        args.model_path = os.path.join(args.evaluate_path, 'models', 'model.pth')
    
    logger = set_logging(args.log_path)
    args.device = set_device()
    log_experiment_settings(logger, args)
    
    if not args.evaluate:
        writer = set_tensorboard(tensorboard_log_dir, args)
        return args, logger, writer

    return args, logger


def process_file(retrieved_text_path):

    loaded_data = np.load(retrieved_text_path, allow_pickle=True)

    data_dict = {}
    for item in loaded_data:
        key = item['image_id']
        tags = [tag.replace('a photo of a ', '').strip() for tag in item['tags']]
        attributes = [attribute.strip() for attribute in item['attributes']]
        sub_dict = {
            'tags': tags,
            'attributes': attributes 
        }
        data_dict[key] = sub_dict
    return data_dict