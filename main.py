import os
import torch
from torch.utils.data import DataLoader
from dataset import MRNetDataset

from nnunet_model import train_nnunet_model, infer_nnunet_model
from feature_extractor import extract_features_from_ROI
from utils import save_model_weights

# Define train_logistic_regression_model and output_feature_importance directly here
import torch
import torch.nn as nn
import torch.optim as optim

def train_model1_and_extract_features(nnunet, dataloader, device):
    nnunet.eval()  # Set nnU-Net to evaluation mode
    extracted_features = []
    labels = []
    
    for batch_idx, batch in enumerate(dataloader):
        images, batch_labels = batch
        images = images.to(device)
        batch_labels = batch_labels.to(device)

        # Model1 (nnU-Net) to generate mask
        with torch.no_grad():
            mask = nnunet(images)
        print(f"Forward pass input shape: {images.shape}")
        print(f"Model output shape (mask): {mask.shape}")

        # Extract features from the masked image (ROI)
        for i in range(images.size(0)):
            cropped_image = images[i] * mask[i]  # Crop the image using the mask
            print(f"  Processing image {i + 1} in batch {batch_idx + 1}:")
            print(f"    Cropped image shape: {cropped_image.shape}")
            features = extract_features_from_ROI(cropped_image)  # Assume this function now handles torch tensors
            print(f"    Extracted features shape: {features.shape}")
            
            features = features.view(-1)
            extracted_features.append(features)
            labels.append(batch_labels[i])

    labels = torch.stack(labels).float()

    return torch.stack(extracted_features), labels


class LogisticRegressionModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(LogisticRegressionModel, self).__init__()
        # Dynamically set the input size based on extracted features
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, x):
        outputs = self.linear(x)
        return outputs

def train_logistic_regression_model(extracted_features, labels, num_epochs=10, learning_rate=0.01):
    input_size = extracted_features.shape[1]
    output_size = labels.shape[1]  # Number of classes
    print(f"Input size: {input_size}, Output size: {output_size}")
    print(f"Extracted features shape: {extracted_features.shape}")
    print(f"Labels shape: {labels.shape}")

    model = LogisticRegressionModel(input_size, output_size)
    criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(num_epochs):
        model.train()

        # Forward pass
        outputs = model(extracted_features)
        print(f"Outputs shape: {outputs.shape}")
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch+1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

    return model

def output_feature_importance(model):
    # Access the learned weights as the feature importance
    importance = model.linear.weight.detach().cpu()  # Keep the weights in torch.Tensor format
    print("Feature Importance (Tensor):", importance)
    return importance


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    root_dir = "/Users/chrissychen/Documents/PhD_2nd_year/miccai2025/MRNet-v1.0"  # Update to your dataset path
    labels_files = {
        'abnormal': os.path.join(root_dir, 'train-abnormal.csv'),
        'acl': os.path.join(root_dir, 'train-acl.csv'),
        'meniscus': os.path.join(root_dir, 'train-meniscus.csv')
    }

    # Dataset and DataLoader
    train_dataset = MRNetDataset(root_dir=root_dir, phase='train', view='coronal', labels_files=labels_files, target_size=(32, 256, 256))
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)

    # Load and train nnU-Net model
    print("Training nnU-Net model...")
    nnunet = train_nnunet_model(train_loader, device, epochs=2)  # You can adjust the number of epochs

    # Step 1: Train nnU-Net and extract features
    print("Extracting features from ROIs...")
    extracted_features, labels = train_model1_and_extract_features(nnunet, train_loader, device)

    # Step 2: Train logistic regression model
    print("Training logistic regression model...")
    logistic_regression_model = train_logistic_regression_model(extracted_features, labels, num_epochs=10, learning_rate=0.01)

    # Step 3: Save the feature weights to CSV
    #feature_names = [f'feature_{i}' for i in range(extracted_features.shape[1])]
    #save_model_weights(logistic_regression_model, feature_names, "feature_weights.csv")

    # Output feature importance
    importance = output_feature_importance(logistic_regression_model)
    print("Feature importance:", importance)


if __name__ == "__main__":
    main()


