import os
import torch
from torch.utils.data import DataLoader
from dataset import MRNetDataset
from unet_model import UNet  # Import your custom UNet
from feature_extractor import extract_features_from_ROI
from outcome_predictor import LogisticRegressionModel
from utils import save_model_weights
import torch.optim as optim
import torch.nn as nn



def train_combined_model(unet, dataloader, device, num_epochs=10, lr=0.001):
    
    optimizer_unet = optim.Adam(unet.parameters(), lr=lr)
    
    regression_loss_fn = nn.BCELoss()  

    regression_model = None  

    unet.train()  
    print(f"UNet is using device: {next(unet.parameters()).device}")

    for epoch in range(num_epochs):
        #print(f"\nEpoch [{epoch+1}/{num_epochs}]")  
        total_loss = 0
        for batch_idx, batch in enumerate(dataloader):
            images, reg_labels = batch  
            images = images.to(device)
            reg_labels = reg_labels.to(device)

            print(f"Batch {batch_idx+1}:")  
            #print(f"  Input images shape: {images.shape}")
            #print(f"  Regression labels shape: {reg_labels.shape}")

           
            masks = unet(images)
            #print(f"  UNet output (masks) shape: {masks.shape}")

            
            features = []
            for i in range(images.size(0)):
                cropped_image = images[i] * masks[i]  # Cropping the image with the mask
                print(f"    Processing image {i+1}/{images.size(0)}, cropped image shape: {cropped_image.shape}")
                features.append(extract_features_from_ROI(cropped_image))
                print(f"    Image {i+1} processed, extracted feature shape: {features[-1].shape}")  # Debug: Feature extraction

            print("  Combining features...")
            features = torch.stack(features).to(device)  # Combine all extracted features
            print(f"  Combined features shape: {features.shape}")

            
            if regression_model is None:
                input_dim = features.shape[1]  
                regression_model = LogisticRegressionModel(input_dim=input_dim, output_dim=1).to(device)
                optimizer_regression = optim.Adam(regression_model.parameters(), lr=lr)  
                print(f"  Regression model initialized with input_dim: {input_dim}")

            
            print("  Performing forward pass through regression model...")
            reg_outputs = regression_model(features)
            print(f"  Regression model output shape: {reg_outputs.shape}")

            
            reg_loss = regression_loss_fn(reg_outputs, reg_labels)
            print(f"  Regression loss: {reg_loss.item()}")  

            # Backpropagation
            optimizer_unet.zero_grad()
            optimizer_regression.zero_grad()
            total_loss = reg_loss
            total_loss.backward()
            optimizer_unet.step()
            optimizer_regression.step()

        print(f"Epoch [{epoch+1}/{num_epochs}] completed, Total Loss: {total_loss.item():.4f}")  
    return unet, regression_model


def main():
   
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

   
    #root_dir = "/Users/chrissychen/Documents/PhD_2nd_year/miccai2025/MRNet-v1.0"  
    root_dir = "/home/yaxi/MRNet-v1.0_gpu" 
    labels_files = {
        'abnormal': os.path.join(root_dir, 'train-abnormal.csv'),
        'acl': os.path.join(root_dir, 'train-acl.csv'),
        'meniscus': os.path.join(root_dir, 'train-meniscus.csv')
    }


    train_dataset = MRNetDataset(
        root_dir=root_dir,
        phase='train',
        view='coronal',
        labels_files=labels_files,
        target_size=(32, 256, 256)
    )
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)

    print("Initializing UNet model...")
    unet = UNet(in_channels=1, out_channels=3).to(device)
    print("UNet model initialized.")

    print("Starting training process...")
    trained_unet, trained_regression_model = train_combined_model(unet, train_loader, device)
    print("Training process completed.")

    save_model_weights(trained_unet, 'nnunet_model.pth')
    save_model_weights(trained_regression_model, 'regression_model.pth')


if __name__ == "__main__":
    main()
