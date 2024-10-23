import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from dataset import MRNetDataset
from feature_extractor import FeatureExtractor

def save_features(features, labels, idx, output_dir):
    """Save extracted features and labels to disk as .npy files."""
    feature_path = os.path.join(output_dir, f"features_{idx:04d}.npy")
    label_path = os.path.join(output_dir, f"labels_{idx:04d}.npy")
    np.save(feature_path, features.numpy())
    np.save(label_path, labels.numpy())

def main():
    #root_dir = "/Users/chrissychen/Documents/PhD_2nd_year/miccai2025/MRNet-v1.0"  
    root_dir = "/home/yaxi/MRNet-v1.0_gpu" 
    output_dir = "/home/yaxi/Carti_tex_morph/extracted_features"
    os.makedirs(output_dir, exist_ok=True)
    
    labels_files = {
        'abnormal': os.path.join(root_dir, 'train-abnormal.csv'),
        'acl': os.path.join(root_dir, 'train-acl.csv'),
        'meniscus': os.path.join(root_dir, 'train-meniscus.csv')
    }

    # Load the dataset
    dataset = MRNetDataset(root_dir=root_dir, phase='train', view='coronal', labels_files=labels_files)

    # Set up DataLoader
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False)

    # Initialize the feature extractor
    feature_extractor = FeatureExtractor()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    feature_extractor.to(device)

    # Extract features and save them
    for idx, (image, labels) in enumerate(data_loader):
        image, labels = image.to(device), labels.to(device)
        features = feature_extractor(image.squeeze(0)) 

        # Save features and labels
        save_features(features, labels.squeeze(0), idx, output_dir)

        if idx % 100 == 0:
            print(f"Processed {idx} images")

if __name__ == "__main__":
    main()
