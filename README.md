
### MRI Image Disease Classification System

**Input for the entire system:**

- **MRI Images**: 3D .nii files
- **Classification Labels**: Labels for training (e.g., healthy, diseased)

### **Step 1: Model 1 - nnU-Net (Segmentation Model)**

- **Purpose**: Train a segmentation model (nnU-Net) to predict regions of interest (ROIs) without supervision by segmentation masks but using classification labels in subsequent stages to refine the process.
- **Input for Model 1**:
    - MRI images in 3D .nii format.
- **Output for Model 1**:
    - Predicted masks representing regions of interest (ROIs) in the MRI images.
- **Inference using Model 1**: Once trained, use the model to generate segmentation masks (ROIs) for new MRI images. These masks are then passed to the next step for feature extraction.

### **Step 2: Feature Extraction**

- **Input**:
    - 3D MRI images (cropped using the ROI masks generated by Model 1).
- **Output**:
    - **Texture features**: Capturing intensity-based characteristics (e.g., GLCM, GLRLM).
    - **Morphological features**: Describing shape, size, and geometry of the segmented regions (e.g., volume, surface area).

### **Step 3: Model 2 - Classification Model (Logistic Regression)**

- **Purpose**: Classify the MRI images based on extracted features (texture and morphological).
- **Input for Model 2**:
    - Extracted texture and morphological features from the ROIs.
- **Output for Model 2**:
    - Classification labels (e.g., healthy or diseased). 0 or 1
- **Post-Training**: After training the logistic regression model, extract the weights of the features to assess their importance in making predictions.

### **Prediction Output for the Entire System**:

- **Final Output**: Predicted classification labels for new MRI images (e.g., disease classification).