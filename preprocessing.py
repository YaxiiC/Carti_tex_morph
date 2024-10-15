import numpy as np
import nibabel as nib
import os

def convert_npy_to_nii_gz(npy_folder, output_folder, prefix='CORO_000'):
    """
    Converts all .npy files in the given folder to .nii.gz format and renames them 
    according to the format 'BRATS_000_0001.nii.gz', 'BRATS_000_0002.nii.gz', etc.
    
    Parameters:
    - npy_folder: Path to the folder containing .npy files.
    - output_folder: Path to the folder where .nii.gz files will be saved.
    - prefix: Naming prefix for the output files (default: 'BRATS_000').
    """
    # Ensure output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Get all .npy files in the folder
    npy_files = sorted([f for f in os.listdir(npy_folder) if f.endswith('.npy')])

    # Convert each .npy file to .nii.gz
    for idx, npy_file in enumerate(npy_files):
        npy_path = os.path.join(npy_folder, npy_file)
        
        # Load the numpy array
        npy_array = np.load(npy_path)
        
        # Convert to NIfTI format
        nii_image = nib.Nifti1Image(npy_array, affine=np.eye(4))  # Affine is identity for simplicity
        
        # Create the new file name following the 'BRATS_000_XXXX.nii.gz' pattern
        new_file_name = f'{prefix}_{str(idx).zfill(4)}.nii.gz'#
        output_path = os.path.join(output_folder, new_file_name)
        
        # Save the NIfTI image as .nii.gz
        nib.save(nii_image, output_path)
        
        print(f"Converted {npy_file} to {new_file_name}")

# Example usage
npy_folder = "/Users/chrissychen/Documents/PhD_2nd_year/miccai2025/MRNet-v1.0/train/coronal_orig"  # Folder where your .npy files are located
output_folder = "/Users/chrissychen/Documents/PhD_2nd_year/miccai2025/MRNet-v1.0/train/coronal-nii"  # Folder where the .nii.gz files will be saved

convert_npy_to_nii_gz(npy_folder, output_folder)
