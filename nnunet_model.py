
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F

class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super(UNet, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv3d(in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv3d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(2)
        )
        self.decoder = nn.Sequential(
            nn.Conv3d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv3d(64, out_channels, kernel_size=3, padding=1),
            nn.Sigmoid()  
        )

    def forward(self, x):
        print(f"Forward pass input shape: {x.shape}")  # Debug line
        x = self.encoder(x)
        x = F.interpolate(x, scale_factor=(2, 2, 2), mode='trilinear', align_corners=False)
        print(f"After encoder and interpolation, shape: {x.shape}")  # Debug line
        x = self.decoder(x)
        print(f"Final output shape: {x.shape}")  # Debug line
        return x


def train_nnunet_model(train_loader, device, epochs=2):
    model = UNet(in_channels=1, out_channels=3).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.BCELoss() 

    model.train()
    for epoch in range(epochs):
        for images, labels in train_loader:
            images = images.to(device)
            print(f"Input images shape: {images.shape}")

            # Forward pass
            optimizer.zero_grad()
            outputs = model(images)
            print(f"Model output shape: {outputs.shape}")  # Debug line

            masks = labels.to(device)  # assuming labels are mask-like
            print(f"Initial masks shape: {masks.shape}") 

            if masks.dim() == 2:  # Assuming these are classification labels
                N, num_classes = masks.shape
                mask_shape = (N, num_classes, images.shape[2], images.shape[3], images.shape[4])  # Match MRI image shape
                pseudo_masks = torch.zeros(mask_shape).to(device)
                
                for i in range(N):
                    for j in range(num_classes):
                        label_value = masks[i, j].unsqueeze(0).unsqueeze(1).unsqueeze(2).unsqueeze(3)  # Shape: [1, 1, 1, 1]
                        pseudo_masks[i, j] = label_value.repeat(1, images.shape[2], images.shape[3], images.shape[4])
                
                masks = pseudo_masks
                print(f"Generated pseudo-masks shape: {masks.shape}")
                

            masks = F.interpolate(masks, size=outputs.shape[2:], mode='trilinear', align_corners=False)
            print(f"Masks shape after interpolation: {masks.shape}")
            # Calculate loss and backpropagate
            loss = criterion(outputs, masks)
            print(f"Loss: {loss.item()}")
            loss.backward()
            optimizer.step()
    return model


def infer_nnunet_model(model, mri_image):
    model.eval()
    with torch.no_grad():
        mri_image = mri_image.unsqueeze(0).to(next(model.parameters()).device) 
        predicted_mask = model(mri_image)
        return predicted_mask.squeeze(0)
