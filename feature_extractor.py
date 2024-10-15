import torch
import numpy as np
import torch.nn.functional as F

def extract_features_from_ROI(masked_image):
    """
    Extract texture and shape features from a masked region of interest (ROI).
    
    Args:
    masked_image (torch.Tensor): A 5D tensor of shape (Batch, Channels, Depth, Height, Width).

    Returns:
    torch.Tensor: A tensor of shape (Batch, FeatureLength) containing both texture and shape features.
    """
    num_classes = masked_image.shape[0]
    combined_features = []

    for c in range(num_classes):
        image = masked_image[c]  # Shape: (Depth, Height, Width)
        print(f"Processing class {c+1}/{num_classes}, image shape: {image.shape}")

        if image.dim() != 3:
            raise ValueError(f"Expected 3D image, got {image.dim()}D image.")

        texture_features = first_order_stats(image) 
        print("First-order stats extracted")

        shape_features = extract_shape_features(image)
        print("Shape features extracted")

        glcm_features = extract_glcm_features(image)

        gldm_features = extract_gldm_features(image)
        glrlm_features = extract_glrlm_features(image)
        
        #need to be fixed
        glszm_features = extract_glszm_features(image)
        #need to be fixed
        ngtdm_features = extract_ngtdm_features(image)

        combined_feature_vector = torch.cat([
            texture_features,
            shape_features,
            glcm_features,
            gldm_features,
            glrlm_features,
            glszm_features,
            ngtdm_features
        ], dim=0)

        combined_features.append(combined_feature_vector)
        print(f"Combined feature vector shape: {combined_feature_vector.shape}")

    return torch.stack(combined_features, dim=0)


def extract_shape_features(image):
    """
    Extract shape features for a single 3D image.
    
    Args:
    image (torch.Tensor): A 4D tensor of shape (Channels, Depth, Height, Width).
    
    Returns:
    torch.Tensor: A tensor containing shape-related features.
    """
    volume, surface_area = calculate_geometric_features(image)

    eigenvalues = calculate_eigenvalues(image)

    shape_features = shape_descriptors(volume, surface_area, eigenvalues)

    return shape_features


def calculate_geometric_features(image):
    """
    Calculate basic geometric features such as volume and surface area.
    
    Args:
    image (torch.Tensor): A 4D tensor of shape (Channels, Depth, Height, Width).
    
    Returns:
    tuple: (volume, surface_area)
    """
    volume = torch.sum(image > 0).item()

    surface_area = torch.sum((image > 0).float()) 

    return volume, surface_area


def calculate_eigenvalues(image):
    """
    Calculate eigenvalues using Principal Component Analysis (PCA) for shape description.

    Args:
    image (torch.Tensor): A 3D tensor of shape (Depth, Height, Width).

    Returns:
    torch.Tensor: A tensor of sorted eigenvalues (smallest to largest).
    """
    coordinates = torch.nonzero(image > 0).float() 
    num_points = coordinates.shape[0]
    if num_points < 3:
        # Not enough points to calculate a full 3D covariance matrix
        # Return zeros for missing eigenvalues
        eigenvalues = torch.zeros(3, device=image.device)
        if num_points == 2:
            # Calculate covariance matrix for 2D
            centered_coords = coordinates - coordinates.mean(dim=0)
            covariance_matrix = torch.matmul(centered_coords.T, centered_coords) / (num_points - 1)
            eigenvals = torch.linalg.eigvalsh(covariance_matrix)
            eigenvalues[:2] = torch.sort(eigenvals)[0]
        elif num_points == 1:
            # Only one point, eigenvalues are zeros
            pass
        return eigenvalues
    centered_coords = coordinates - coordinates.mean(dim=0)

    # Calculate covariance matrix
    covariance_matrix = torch.matmul(centered_coords.T, centered_coords) / (num_points - 1)

    # Eigenvalue decomposition
    eigenvalues = torch.linalg.eigvalsh(covariance_matrix)

    # Ensure eigenvalues are sorted in ascending order and have size 3
    eigenvalues = torch.sort(eigenvalues)[0]
    if eigenvalues.numel() < 3:
        eigenvalues = torch.cat([eigenvalues, torch.zeros(3 - eigenvalues.numel(), device=image.device)])

    return eigenvalues



def shape_descriptors(volume, surface_area, eigenvalues):
    """
    Derive advanced shape descriptors like sphericity, compactness, and flatness.
    
    Args:
    volume (float): Volume of the 3D region.
    surface_area (float): Surface area of the 3D region.
    eigenvalues (torch.Tensor): Eigenvalues of the shape.

    Returns:
    torch.Tensor: A tensor containing derived shape descriptors.
    """
    sphericity = calculate_sphericity(volume, surface_area)
    compactness = calculate_compactness(volume, surface_area)
    flatness = calculate_flatness(eigenvalues)

    # Combine all shape descriptors into a single tensor
    shape_features = torch.tensor([sphericity, compactness, flatness], dtype=torch.float32)

    return shape_features


def first_order_stats(x):
    """
    Extract first-order statistical features from a 2D or 3D tensor.
    
    Args:
    x (torch.Tensor): Input tensor.
    
    Returns:
    torch.Tensor: A tensor containing the extracted first-order statistical features.
    """
    x = x + 1  # Shift value can be adjusted as needed
    Np = torch.numel(x)  # Total number of voxels
    X = x.view(-1)  # Flatten the tensor

    # Calculate the histogram
    Ng = 256  # Number of gray levels
    hist = torch.histc(X, bins=Ng, min=float(X.min()), max=float(X.max()))
    p_i = hist / Np  # Normalized histogram

    # Calculate first-order statistics
    energy = torch.sum(X ** 2)
    voxel_volume = 1.0  # Adjust based on actual voxel volume
    total_energy = voxel_volume * torch.sum(X ** 2)
    entropy = -torch.sum(p_i * torch.log2(p_i + 1e-9))
    minimum = torch.min(X)
    tenth_percentile = torch.quantile(X, 0.1)
    ninetieth_percentile = torch.quantile(X, 0.9)
    maximum = torch.max(X)
    mean = torch.mean(X)
    median = torch.median(X)
    iqr = torch.quantile(X, 0.75) - torch.quantile(X, 0.25)
    range_ = maximum - minimum
    mad = torch.mean(torch.abs(X - mean))
    X_10_90 = X[(X >= tenth_percentile) & (X <= ninetieth_percentile)]
    mean_10_90 = torch.mean(X_10_90)
    rmad = torch.mean(torch.abs(X_10_90 - mean_10_90))
    rms = torch.sqrt(torch.mean(X ** 2))
    std_dev = torch.std(X)
    skewness = torch.mean(((X - mean) / std_dev) ** 3)
    kurtosis = torch.mean(((X - mean) / std_dev) ** 4) - 3
    variance = torch.var(X)
    uniformity = torch.sum(p_i ** 2)

    # Return all features as a tensor
    features = torch.tensor([energy, total_energy, entropy, minimum, tenth_percentile, ninetieth_percentile,
                             maximum, mean, median, iqr, range_, mad, rmad, rms, std_dev, skewness,
                             kurtosis, variance, uniformity], dtype=torch.float32)
    return features


def calculate_sphericity(volume, surface_area):
    """
    Calculate sphericity based on volume and surface area.
    
    Args:
    volume (float): Volume of the shape.
    surface_area (float): Surface area of the shape.
    
    Returns:
    float: Sphericity value.
    """
    return (36 * np.pi * volume ** 2) ** (1.0 / 3.0) / surface_area


def calculate_compactness(volume, surface_area):
    """
    Calculate compactness based on volume and surface area.
    
    Args:
    volume (float): Volume of the shape.
    surface_area (float): Surface area of the shape.
    
    Returns:
    float: Compactness value.
    """
    return (36.0 * np.pi) * (volume ** 2.0) / (surface_area ** 3.0)


def calculate_flatness(eigenvalues):
    """
    Calculate flatness based on eigenvalues.
    
    Args:
    eigenvalues (torch.Tensor): A tensor of eigenvalues (sorted from smallest to largest).
    
    Returns:
    float: Flatness value.
    """
    if eigenvalues.numel() < 2:
        return 0.0

    smallest = eigenvalues[0]
    largest = eigenvalues[-1]

    if largest == 0:
        return 0.0
    return torch.sqrt(smallest / (largest + 1e-9)).item()  # Least axis over major axis



def discretize_image(image, Ng):
    """
    Discretize image into Ng gray levels.
    
    Args:
    image (torch.Tensor): 3D tensor (Depth, Height, Width).
    Ng (int): Number of gray levels.

    Returns:
    torch.Tensor: Discretized image with values from 0 to Ng-1.
    """
    min_val = image.min()
    max_val = image.max()

    # Avoid division by zero
    if max_val == min_val:
        return torch.zeros_like(image, dtype=torch.long)

    # Scale image to [0, Ng-1]
    image = (image - min_val) / (max_val - min_val) * (Ng - 1)
    image = image.long()

    return image

def shift_image(image, dx, dy, dz):
    """
    Shift the 3D image tensor by dx, dy, dz.
    If the shift moves beyond the boundary, zeros are filled in.
    
    Args:
    image (torch.Tensor): 3D tensor to shift.
    dx (int): Shift along Depth axis.
    dy (int): Shift along Height axis.
    dz (int): Shift along Width axis.
    
    Returns:
    torch.Tensor: Shifted image tensor.
    """
    D, H, W = image.shape
    shifted_image = torch.zeros_like(image)

    src_x_start = max(-dx, 0)
    src_x_end = min(D - dx, D)
    dst_x_start = max(dx, 0)
    dst_x_end = min(D + dx, D)

    src_y_start = max(-dy, 0)
    src_y_end = min(H - dy, H)
    dst_y_start = max(dy, 0)
    dst_y_end = min(H + dy, H)

    src_z_start = max(-dz, 0)
    src_z_end = min(W - dz, W)
    dst_z_start = max(dz, 0)
    dst_z_end = min(W + dz, W)

    shifted_image[dst_x_start:dst_x_end, dst_y_start:dst_y_end, dst_z_start:dst_z_end] = \
        image[src_x_start:src_x_end, src_y_start:src_y_end, src_z_start:src_z_end]

    return shifted_image


def compute_glcm(image, Ng, offsets, symmetric=True):
    """
    Compute the GLCM matrix for a 3D image.

    Args:
    image (torch.Tensor): 3D tensor (D, H, W) with integer values from 0 to Ng-1.
    Ng (int): Number of gray levels.
    offsets (list): List of offsets (dx, dy, dz).
    symmetric (bool): Whether to make the GLCM matrix symmetric.

    Returns:
    torch.Tensor: GLCM matrix of shape (Ng, Ng).
    """
    if image.dim() != 3:
        raise ValueError(f"Expected 3D image, got {image.dim()}D image.")
    
    D, H, W = image.shape
    glcm_sum = torch.zeros((Ng, Ng), dtype=torch.float32, device=image.device)

    for offset in offsets:
        dx, dy, dz = offset

        # Depending on the sign of dx, dy, dz, we need to adjust the slices
        if dx >= 0:
            x1 = slice(0, D - dx)
            x2 = slice(dx, D)
        else:
            x1 = slice(-dx, D)
            x2 = slice(0, D + dx)

        if dy >= 0:
            y1 = slice(0, H - dy)
            y2 = slice(dy, H)
        else:
            y1 = slice(-dy, H)
            y2 = slice(0, H + dy)

        if dz >= 0:
            z1 = slice(0, W - dz)
            z2 = slice(dz, W)
        else:
            z1 = slice(-dz, W)
            z2 = slice(0, W + dz)

        image1 = image[x1, y1, z1]
        image2 = image[x2, y2, z2]

        # Now image1 and image2 are the same shape
        i_vals = image1.flatten()
        j_vals = image2.flatten()

        # Create a 2D histogram using torch.bincount
        indices = i_vals * Ng + j_vals  # Combined indices
        glcm = torch.bincount(indices, minlength=Ng*Ng).float()
        glcm = glcm.reshape(Ng, Ng)

        if symmetric:
            glcm = glcm + glcm.T

        glcm_sum += glcm

    return glcm_sum


def extract_glcm_features(image, Ng=8, offsets=[(0, 1, 0), (1, 0, 0), (0, 0, 1)], symmetric=True):
    """
    Extract GLCM features from a 3D image.

    Args:
    image (torch.Tensor): A 3D tensor (Depth, Height, Width).
    Ng (int): Number of gray levels.
    offsets (list): List of offsets (dx, dy, dz).
    symmetric (bool): Whether to make the GLCM matrix symmetric.

    Returns:
    torch.Tensor: A tensor containing GLCM features.
    """
    # Discretize the image into Ng levels
    image = discretize_image(image, Ng)

    # Compute the GLCM matrix
    glcm = compute_glcm(image, Ng, offsets, symmetric=symmetric)

    # Normalize the GLCM matrix
    glcm = glcm / (glcm.sum() + 1e-9)

    eps = 1e-9

    # Compute probabilities
    p_ij = glcm  # Ng x Ng

    p_i = p_ij.sum(dim=1)  # Ng
    p_j = p_ij.sum(dim=0)  # Ng

    i_values = torch.arange(Ng, dtype=torch.float32, device=image.device)
    j_values = torch.arange(Ng, dtype=torch.float32, device=image.device)

    # Compute means
    mu_i = torch.sum(p_i * i_values)
    mu_j = torch.sum(p_j * j_values)

    # Compute standard deviations
    sigma_i = torch.sqrt(torch.sum(p_i * (i_values - mu_i) ** 2))
    sigma_j = torch.sqrt(torch.sum(p_j * (j_values - mu_j) ** 2))

    # Prepare matrices for computation
    i_matrix = i_values.unsqueeze(1).expand(Ng, Ng)
    j_matrix = j_values.unsqueeze(0).expand(Ng, Ng)

    # Compute features
    features = []

    # 1. Energy
    energy = torch.sum(p_ij ** 2)
    features.append(energy)

    # 2. Entropy
    entropy = -torch.sum(p_ij * torch.log2(p_ij + eps))
    features.append(entropy)

    # 3. Contrast
    contrast = torch.sum((i_matrix - j_matrix) ** 2 * p_ij)
    features.append(contrast)

    # 4. Homogeneity
    homogeneity = torch.sum(p_ij / (1.0 + torch.abs(i_matrix - j_matrix)))
    features.append(homogeneity)

    # 5. Correlation
    if sigma_i * sigma_j == 0:
        correlation = 0.0  # Avoid division by zero
    else:
        correlation = torch.sum(((i_matrix - mu_i) * (j_matrix - mu_j) * p_ij)) / (sigma_i * sigma_j)
    features.append(correlation)

    # 6. Sum of Squares (Variance)
    sum_squares = torch.sum((i_matrix - mu_i) ** 2 * p_ij)
    features.append(sum_squares)

    # Return all features as a tensor
    return torch.tensor(features, dtype=torch.float32, device=image.device)

def compute_gldm(image, Ng, alpha=0, delta=1):
    """
    Compute the Gray Level Dependence Matrix (GLDM) for a 3D image.
    """
    D, H, W = image.shape

    # Generate neighbor offsets
    offsets = [(dx, dy, dz)
               for dx in range(-delta, delta+1)
               for dy in range(-delta, delta+1)
               for dz in range(-delta, delta+1)
               if not (dx == 0 and dy == 0 and dz == 0)]

    # Initialize total_dependents
    total_dependents = torch.zeros_like(image, dtype=torch.int32)

    # Compute dependent neighbors
    for offset in offsets:
        dx, dy, dz = offset
        shifted_image = shift_image(image, dx, dy, dz)
        abs_diff = torch.abs(image - shifted_image)
        dependent_mask = (abs_diff <= alpha).int()
        total_dependents += dependent_mask

    # Flatten and compute GLDM matrix
    i_values = image.view(-1)
    j_values = total_dependents.view(-1)
    MaxNumDependents = total_dependents.max().item()
    Nd = MaxNumDependents + 1
    indices = (i_values.long() - 1) * Nd + j_values.long()
    counts = torch.bincount(indices, minlength=Ng * Nd)
    P_gldm = counts.view(Ng, Nd)

    return P_gldm


def extract_gldm_features(image, Ng=8, alpha=0, delta=1):
    """
    Extract GLDM features from a 3D image.
    """
    # Discretize the image
    image = discretize_image(image, Ng) + 1

    # Compute the GLDM matrix
    P_gldm = compute_gldm(image, Ng, alpha=alpha, delta=delta)

    # Compute GLDM features
    Nz = P_gldm.sum()
    eps = 1e-9
    if Nz == 0:
        Nz = eps
    p_gldm = P_gldm / Nz
    p_g = p_gldm.sum(dim=1)
    p_d = p_gldm.sum(dim=0)
    i_vector = torch.arange(1, Ng+1, dtype=torch.float32, device=image.device)
    j_vector = torch.arange(0, P_gldm.shape[1], dtype=torch.float32, device=image.device)

    features = []

    # Small Dependence Emphasis (SDE)
    SDE = torch.sum(p_d[1:] / (j_vector[1:] ** 2))
    features.append(SDE)

    # Large Dependence Emphasis (LDE)
    LDE = torch.sum(p_d[1:] * (j_vector[1:] ** 2))
    features.append(LDE)

    # Gray Level Non-Uniformity (GLN)
    GLN = torch.sum(p_g ** 2) / Nz
    features.append(GLN)

    # Dependence Non-Uniformity (DN)
    DN = torch.sum(p_d ** 2) / Nz
    features.append(DN)

    # Dependence Non-Uniformity Normalized (DNN)
    DNN = torch.sum(p_d ** 2) / (Nz ** 2)
    features.append(DNN)

    # Gray Level Variance (GLV)
    mu_i = torch.sum(p_g * i_vector)
    GLV = torch.sum(p_g * ((i_vector - mu_i) ** 2))
    features.append(GLV)

    # Dependence Variance (DV)
    mu_j = torch.sum(p_d * j_vector)
    DV = torch.sum(p_d * ((j_vector - mu_j) ** 2))
    features.append(DV)

    # Dependence Entropy (DE)
    DE = -torch.sum(p_gldm * torch.log2(p_gldm + eps))
    features.append(DE)

    # Low Gray Level Emphasis (LGLE)
    LGLE = torch.sum(p_g / (i_vector ** 2))
    features.append(LGLE)

    # High Gray Level Emphasis (HGLE)
    HGLE = torch.sum(p_g * (i_vector ** 2))
    features.append(HGLE)

    # Small Dependence Low Gray Level Emphasis (SDLGLE)
    SDLGLE = torch.sum(p_gldm[1:, 1:] / ((i_vector[1:, None] ** 2) * (j_vector[None, 1:] ** 2)))
    features.append(SDLGLE)

    # Small Dependence High Gray Level Emphasis (SDHGLE)
    SDHGLE = torch.sum(p_gldm[1:, 1:] * (i_vector[1:, None] ** 2) / (j_vector[None, 1:] ** 2))
    features.append(SDHGLE)

    # Large Dependence Low Gray Level Emphasis (LDLGLE)
    LDLGLE = torch.sum(p_gldm[1:, 1:] * (j_vector[None, 1:] ** 2) / (i_vector[1:, None] ** 2))
    features.append(LDLGLE)

    # Large Dependence High Gray Level Emphasis (LDHGLE)
    LDHGLE = torch.sum(p_gldm[1:, 1:] * (i_vector[1:, None] ** 2) * (j_vector[None, 1:] ** 2))
    features.append(LDHGLE)

    return torch.tensor(features, dtype=torch.float32, device=image.device)


def compute_runs(line):
    """
    Compute runs in a 1D tensor.

    Args:
        line (torch.Tensor): 1D tensor

    Returns:
        List of (gray_level, run_length)
    """
    # Get the positions where the value changes
    change_positions = torch.where(line[1:] != line[:-1])[0] + 1
    # Append the start and end positions
    positions = torch.cat((torch.tensor([0], device=line.device), change_positions, torch.tensor([len(line)], device=line.device)))
    run_lengths = positions[1:] - positions[:-1]
    gray_levels = line[positions[:-1]]
    runs = list(zip(gray_levels.tolist(), run_lengths.tolist()))
    return runs


def compute_glrlm_along_axis(image, Ng, axis):
    """
    Compute the GLRLM along the specified axis.

    Args:
        image (torch.Tensor): 3D tensor of shape (D, H, W)
        Ng (int): Number of gray levels
        axis (int): Axis along which to compute GLRLM (0, 1, 2)

    Returns:
        torch.Tensor: GLRLM matrix of shape (Ng, max_run_length)
    """
    D, H, W = image.shape
    max_run_length = max(D, H, W)
    P_glrlm = torch.zeros((Ng, max_run_length), dtype=torch.float32, device=image.device)

    if axis == 0:
        for h in range(H):
            for w in range(W):
                line = image[:, h, w]
                runs = compute_runs(line)
                for gray_level, run_length in runs:
                    P_glrlm[gray_level, run_length - 1] += 1
    elif axis == 1:
        for d in range(D):
            for w in range(W):
                line = image[d, :, w]
                runs = compute_runs(line)
                for gray_level, run_length in runs:
                    P_glrlm[gray_level, run_length - 1] += 1
    elif axis == 2:
        for d in range(D):
            for h in range(H):
                line = image[d, h, :]
                runs = compute_runs(line)
                for gray_level, run_length in runs:
                    P_glrlm[gray_level, run_length - 1] += 1
    else:
        raise ValueError(f"Invalid axis {axis}, expected 0, 1, or 2")

    return P_glrlm


def compute_glrlm(image, Ng):
    """
    Compute the GLRLM for the 3D image.

    Args:
        image (torch.Tensor): 3D tensor of shape (D, H, W)
        Ng (int): Number of gray levels

    Returns:
        torch.Tensor: GLRLM matrix of shape (Ng, max_run_length)
    """
    P_glrlm_0 = compute_glrlm_along_axis(image, Ng, axis=0)
    P_glrlm_1 = compute_glrlm_along_axis(image, Ng, axis=1)
    P_glrlm_2 = compute_glrlm_along_axis(image, Ng, axis=2)
    P_glrlm = P_glrlm_0 + P_glrlm_1 + P_glrlm_2
    return P_glrlm


def extract_glrlm_features(image, Ng=8):
    """
    Extract GLRLM features from a 3D image.

    Args:
        image (torch.Tensor): A 3D tensor (Depth, Height, Width).
        Ng (int): Number of gray levels.

    Returns:
        torch.Tensor: A tensor containing GLRLM features.
    """
    # Discretize the image into Ng levels
    image = discretize_image(image, Ng)
    D, H, W = image.shape

    # Compute the GLRLM matrix
    P_glrlm = compute_glrlm(image, Ng)

    # Total number of runs
    Nr = torch.sum(P_glrlm) + 1e-9  # Avoid division by zero

    # Total number of voxels
    Np = D * H * W

    # Compute pr and pg
    pr = torch.sum(P_glrlm, dim=0)  # Sum over gray levels, result is length Nr
    pg = torch.sum(P_glrlm, dim=1)  # Sum over run lengths, result is length Ng

    # Vectors of indices
    ivector = torch.arange(Ng, dtype=torch.float32, device=image.device) + 1  # Gray levels from 1 to Ng
    jvector = torch.arange(P_glrlm.shape[1], dtype=torch.float32, device=image.device) + 1  # Run lengths from 1 to Nr

    eps = 1e-9

    features = []

    # 1. Short Run Emphasis (SRE)
    SRE = torch.sum(pr / (jvector ** 2)) / Nr
    features.append(SRE)

    # 2. Long Run Emphasis (LRE)
    LRE = torch.sum(pr * (jvector ** 2)) / Nr
    features.append(LRE)

    # 3. Gray Level Non-Uniformity (GLN)
    GLN = torch.sum(pg ** 2) / Nr
    features.append(GLN)

    # 4. Gray Level Non-Uniformity Normalized (GLNN)
    GLNN = torch.sum(pg ** 2) / (Nr ** 2)
    features.append(GLNN)

    # 5. Run Length Non-Uniformity (RLN)
    RLN = torch.sum(pr ** 2) / Nr
    features.append(RLN)

    # 6. Run Length Non-Uniformity Normalized (RLNN)
    RLNN = torch.sum(pr ** 2) / (Nr ** 2)
    features.append(RLNN)

    # 7. Run Percentage (RP)
    RP = Nr / Np
    features.append(RP)

    # For the next features, we need to compute p_glrlm = P_glrlm / Nr
    p_glrlm = P_glrlm / Nr

    # Gray Level Variance (GLV)
    mu_i = torch.sum(p_glrlm * ivector[:, None])
    GLV = torch.sum(p_glrlm * (ivector[:, None] - mu_i) ** 2)
    features.append(GLV)

    # Run Variance (RV)
    mu_j = torch.sum(p_glrlm * jvector[None, :])
    RV = torch.sum(p_glrlm * (jvector[None, :] - mu_j) ** 2)
    features.append(RV)

    # Run Entropy (RE)
    RE = -torch.sum(p_glrlm * torch.log2(p_glrlm + eps))
    features.append(RE)

    # Low Gray Level Run Emphasis (LGLRE)
    LGLRE = torch.sum(pg / (ivector ** 2)) / Nr
    features.append(LGLRE)

    # High Gray Level Run Emphasis (HGLRE)
    HGLRE = torch.sum(pg * (ivector ** 2)) / Nr
    features.append(HGLRE)

    # Short Run Low Gray Level Emphasis (SRLGLE)
    numerator = P_glrlm / ((ivector[:, None] ** 2) * (jvector[None, :] ** 2))
    SRLGLE = torch.sum(numerator) / Nr
    features.append(SRLGLE)

    # Short Run High Gray Level Emphasis (SRHGLE)
    numerator = P_glrlm * (ivector[:, None] ** 2) / (jvector[None, :] ** 2)
    SRHGLE = torch.sum(numerator) / Nr
    features.append(SRHGLE)

    # Long Run Low Gray Level Emphasis (LRLGLE)
    numerator = P_glrlm * (jvector[None, :] ** 2) / (ivector[:, None] ** 2)
    LRLGLE = torch.sum(numerator) / Nr
    features.append(LRLGLE)

    # Long Run High Gray Level Emphasis (LRHGLE)
    numerator = P_glrlm * (ivector[:, None] ** 2) * (jvector[None, :] ** 2)
    LRHGLE = torch.sum(numerator) / Nr
    features.append(LRHGLE)

    # Return features as a tensor
    return torch.tensor(features, dtype=torch.float32, device=image.device)

def extract_glszm_features(image, Ng=8):
    """
    Extract GLSZM features from a 3D image.

    Args:
        image (torch.Tensor): A 3D tensor (Depth, Height, Width).
        Ng (int): Number of gray levels.

    Returns:
        torch.Tensor: A tensor containing GLSZM features.
    """
    # Discretize the image into Ng levels
    image = discretize_image(image, Ng)

    D, H, W = image.shape
    Np = D * H * W

    # Initialize an empty dict to store the zones
    zone_counts = {}

    for gray_level in range(Ng):
        binary_image = (image == gray_level).int()

        # If there are no voxels at this gray level, skip
        if binary_image.sum() == 0:
            continue

        # Perform connected component labeling
        label_tensor = connected_components_labeling(binary_image)

        # Get unique labels and their counts (excluding background label 0)
        labels, counts = torch.unique(label_tensor[label_tensor > 0], return_counts=True)

        # For each zone size, update P_glszm[gray_level, zone_size]
        for size in counts:
            size = size.item()
            key = (gray_level, size)
            if key in zone_counts:
                zone_counts[key] += 1
            else:
                zone_counts[key] = 1

    # Now construct P_glszm
    if zone_counts:
        max_zone_size = max(size for (_, size) in zone_counts.keys())
    else:
        max_zone_size = 1

    P_glszm = torch.zeros((Ng, max_zone_size), dtype=torch.float32, device=image.device)

    for (gray_level, size), count in zone_counts.items():
        size_index = size - 1  # Adjust index since zone sizes start from 1
        P_glszm[gray_level, size_index] = count

    # Now compute the features
    Nz = P_glszm.sum() + 1e-9  # To avoid division by zero

    ps = P_glszm.sum(dim=0)  # Sum over gray levels, counts per zone size
    pg = P_glszm.sum(dim=1)  # Sum over zone sizes, counts per gray level

    ivector = torch.arange(Ng, dtype=torch.float32, device=image.device)  # Gray levels from 0 to Ng-1
    jvector = torch.arange(1, P_glszm.shape[1] + 1, dtype=torch.float32, device=image.device)  # Zone sizes from 1 to Ns

    # 1. Small Area Emphasis (SAE)
    SAE = torch.sum(ps / (jvector ** 2)) / Nz

    # 2. Large Area Emphasis (LAE)
    LAE = torch.sum(ps * (jvector ** 2)) / Nz

    # 3. Gray Level Non-Uniformity (GLN)
    GLN = torch.sum(pg ** 2) / Nz

    # 4. Gray Level Non-Uniformity Normalized (GLNN)
    GLNN = torch.sum(pg ** 2) / (Nz ** 2)

    # 5. Size Zone Non-Uniformity (SZN)
    SZN = torch.sum(ps ** 2) / Nz

    # 6. Size Zone Non-Uniformity Normalized (SZNN)
    SZNN = torch.sum(ps ** 2) / (Nz ** 2)

    # 7. Zone Percentage (ZP)
    ZP = Nz / Np

    # 8. Gray Level Variance (GLV)
    p_i = pg / Nz
    mu_i = torch.sum(p_i * ivector)
    GLV = torch.sum(p_i * (ivector - mu_i) ** 2)

    # 9. Zone Variance (ZV)
    p_j = ps / Nz
    mu_j = torch.sum(p_j * jvector)
    ZV = torch.sum(p_j * (jvector - mu_j) ** 2)

    # 10. Zone Entropy (ZE)
    p_ij = P_glszm / Nz
    eps = 1e-9
    ZE = -torch.sum(p_ij * torch.log2(p_ij + eps))

    # 11. Low Gray Level Zone Emphasis (LGLZE)
    LGLZE = torch.sum(pg / (ivector ** 2)) / Nz

    # 12. High Gray Level Zone Emphasis (HGLZE)
    HGLZE = torch.sum(pg * (ivector ** 2)) / Nz

    # 13. Small Area Low Gray Level Emphasis (SALGLE)
    numerator = P_glszm / ((ivector[:, None] ** 2) * (jvector[None, :] ** 2))
    SALGLE = torch.sum(numerator) / Nz

    # 14. Small Area High Gray Level Emphasis (SAHGLE)
    numerator = P_glszm * (ivector[:, None] ** 2) / (jvector[None, :] ** 2)
    SAHGLE = torch.sum(numerator) / Nz

    # 15. Large Area Low Gray Level Emphasis (LALGLE)
    numerator = P_glszm * (jvector[None, :] ** 2) / (ivector[:, None] ** 2)
    LALGLE = torch.sum(numerator) / Nz

    # 16. Large Area High Gray Level Emphasis (LAHGLE)
    numerator = P_glszm * (ivector[:, None] ** 2) * (jvector[None, :] ** 2)
    LAHGLE = torch.sum(numerator) / Nz

    features = torch.tensor([
        SAE, LAE, GLN, GLNN, SZN, SZNN, ZP, GLV, ZV,
        ZE, LGLZE, HGLZE, SALGLE, SAHGLE, LALGLE, LAHGLE
    ], dtype=torch.float32, device=image.device)

    return features


def connected_components_labeling(binary_image):
    """
    Perform connected component labeling on a binary 3D image using PyTorch operations.

    Args:
        binary_image (torch.Tensor): 3D tensor with values 0 or 1

    Returns:
        label_tensor (torch.Tensor): 3D tensor with integer labels
    """

    # Initialize labels
    label_tensor = torch.zeros_like(binary_image, dtype=torch.int32)
    current_label = 1

    # Define 3D neighborhood kernel for 26-connectivity
    kernel = torch.ones((3, 3, 3), dtype=torch.float32, device=binary_image.device)
    kernel[1, 1, 1] = 0  # Exclude the center voxel

    # Get the foreground voxels
    to_process = binary_image.clone().bool()

    while to_process.any():
        # Find the first voxel to start
        idx = (to_process > 0).nonzero(as_tuple=False)[0]
        seed = torch.zeros_like(binary_image, dtype=torch.float32)
        seed[idx[0], idx[1], idx[2]] = 1.0

        # Initialize the component mask
        component = seed.clone()

        # Iteratively dilate the seed within the binary image
        prev_component = torch.zeros_like(component)
        while not torch.equal(component, prev_component):
            prev_component = component.clone()
            dilation = F.conv3d(
                component.unsqueeze(0).unsqueeze(0),
                kernel.unsqueeze(0).unsqueeze(0),
                padding=1
            ).squeeze()
            component = torch.min((dilation > 0).float(), binary_image.float())

        # Assign labels to the connected component
        label_tensor[component.bool()] = current_label
        current_label += 1

        # Remove the labeled component from the to_process mask
        to_process[component.bool()] = 0

    return label_tensor

def extract_ngtdm_features(image, Ng=8, delta=1):
    """
    Extract NGTDM features from a 3D image.

    Args:
    image (torch.Tensor): A 3D tensor (Depth, Height, Width).
    Ng (int): Number of gray levels.
    delta (int): Neighborhood distance.

    Returns:
    torch.Tensor: A tensor containing NGTDM features.
    """
    # Discretize the image into Ng levels
    image = discretize_image(image, Ng)
    mask = torch.ones_like(image, dtype=torch.float32, device=image.device)  # Assuming the mask is all ones

    # Compute n_i and s_i
    n_i, s_i = compute_ngtdm(image, mask, Ng, delta=delta)

    Nvp = n_i.sum()
    p_i = n_i / Nvp  # Shape (Ng,)

    Ngp = (n_i > 0).sum()  # Number of gray levels where n_i != 0

    # Now compute the features

    # 1. Coarseness
    # Coarseness = 1 / sum_{i=1}^{Ng} (p_i * s_i)
    denominator = (p_i * s_i).sum()
    if denominator != 0:
        coarseness = 1.0 / denominator
    else:
        coarseness = 1e6  # As per the note in the code

    # 2. Contrast
    # Contrast = [sum_{i=1}^{Ng} sum_{j=1}^{Ng} p_i * p_j * (i - j)^2 / [Ngp * (Ngp - 1)]] * [sum_{i=1}^{Ng} s_i / Nvp]
    # Only consider gray levels where p_i != 0
    valid_indices = n_i > 0
    p_i_valid = p_i[valid_indices]
    i_values = torch.arange(Ng, device=image.device)[valid_indices].float()

    Ngp = p_i_valid.numel()
    denominator_contrast = Ngp * (Ngp - 1)
    if denominator_contrast == 0:
        contrast = 0
    else:
        i_matrix = i_values.unsqueeze(0)
        j_matrix = i_values.unsqueeze(1)
        contrast = ((p_i_valid.unsqueeze(0) * p_i_valid.unsqueeze(1) * (i_matrix - j_matrix) ** 2).sum()) / denominator_contrast
        contrast *= s_i.sum() / Nvp

    # 3. Busyness
    # Busyness = sum_{i=1}^{Ng} (p_i * s_i) / sum_{i=1}^{Ng} sum_{j=1}^{Ng} |i*p_i - j*p_j|
    numerator_busyness = (p_i * s_i).sum()
    # Compute |i*p_i - j*p_j|
    i_values_full = torch.arange(Ng, device=image.device).float()
    i_p_i = i_values_full * p_i
    abs_diff_busyness = torch.abs(i_p_i.unsqueeze(0) - i_p_i.unsqueeze(1))
    denominator_busyness = abs_diff_busyness.sum()
    if denominator_busyness != 0:
        busyness = numerator_busyness / denominator_busyness
    else:
        busyness = 0

    # 4. Complexity
    # Complexity = [sum_{i=1}^{Ng} sum_{j=1}^{Ng} |i - j| * [p_i * s_i + p_j * s_j] / (p_i + p_j)] / Nvp
    # Avoid division by zero when p_i + p_j == 0
    p_i_matrix = p_i.unsqueeze(0) + p_i.unsqueeze(1)
    numerator_complexity_matrix = torch.abs(i_values_full.unsqueeze(0) - i_values_full.unsqueeze(1)) * (p_i.unsqueeze(0) * s_i.unsqueeze(0) + p_i.unsqueeze(1) * s_i.unsqueeze(1))
    denominator_complexity_matrix = p_i_matrix
    denominator_complexity_matrix[denominator_complexity_matrix == 0] = 1  # Avoid division by zero
    complexity_matrix = numerator_complexity_matrix / denominator_complexity_matrix
    complexity = complexity_matrix.sum() / Nvp

    # 5. Strength
    # Strength = sum_{i=1}^{Ng} sum_{j=1}^{Ng} (p_i + p_j)*(i - j)^2 / sum_{i=1}^{Ng} s_i
    numerator_strength = ((p_i.unsqueeze(0) + p_i.unsqueeze(1)) * (i_values_full.unsqueeze(0) - i_values_full.unsqueeze(1)) ** 2).sum()
    denominator_strength = s_i.sum()
    if denominator_strength != 0:
        strength = numerator_strength / denominator_strength
    else:
        strength = 0

    features = torch.tensor([coarseness, contrast, busyness, complexity.item(), strength], dtype=torch.float32, device=image.device)

    return features

def compute_ngtdm(image, mask, Ng, delta=1):
    """
    Compute the NGTDM matrix for a 3D image.

    Args:
    image (torch.Tensor): Discretized image, 3D tensor of shape (D, H, W) with integer values from 0 to Ng-1.
    mask (torch.Tensor): Binary mask, same shape as image.
    Ng (int): Number of gray levels.
    delta (int): Neighborhood distance.

    Returns:
    n_i (torch.Tensor): Number of pixels with gray level i that have valid neighborhoods, shape (Ng,)
    s_i (torch.Tensor): Sum of absolute differences for gray level i, shape (Ng,)
    """
    # Define the 3D convolution kernel
    kernel_size = 2 * delta + 1
    center = delta
    kernel = torch.ones((kernel_size, kernel_size, kernel_size), dtype=torch.float32, device=image.device)
    kernel[center, center, center] = 0  # Exclude center pixel

    # Prepare the image and mask
    image = image.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
    mask = mask.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions

    padding = delta  # To keep the output size same as input size

    # Compute sum of neighboring gray levels
    neighbor_sum = F.conv3d(image.float() * mask.float(), kernel.unsqueeze(0).unsqueeze(0), padding=padding)

    # Compute count of neighboring pixels
    neighbor_count = F.conv3d(mask.float(), kernel.unsqueeze(0).unsqueeze(0), padding=padding)

    # Avoid division by zero
    neighbor_count[neighbor_count == 0] = 1

    # Compute average neighbor gray level
    average_neighbor_gray = neighbor_sum / neighbor_count

    # Valid pixels are those where the mask is 1 and neighbor count > 0
    valid_pixels = (mask > 0) & (neighbor_count > 0)

    # Compute absolute difference between pixel value and average neighbor gray level
    abs_diff = torch.abs(image.float() - average_neighbor_gray) * valid_pixels.float()

    # Flatten image, mask, abs_diff
    image_flat = image[valid_pixels].long().view(-1)  # Shape [N]
    abs_diff_flat = abs_diff[valid_pixels].view(-1)  # Shape [N]

    # For each gray level i, accumulate n_i and s_i
    n_i = torch.zeros(Ng, device=image.device)
    s_i = torch.zeros(Ng, device=image.device)

    for i in range(Ng):
        indices = (image_flat == i)
        n_i[i] = indices.sum()
        s_i[i] = abs_diff_flat[indices].sum()

    return n_i, s_i