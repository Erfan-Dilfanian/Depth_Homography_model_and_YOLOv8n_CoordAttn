"""
description:

This code recieves manually-found correpsondant points as an excel file

Also recieve the two images from which points are

It calucaltes the homoraphy matrix from the excel file of manually-found correspondant points

Then, it fuse theimages with this matrix

also, a test function is defined

using this function, the homography matrix will test registration on ither set of images as test images


running guide:

1- install python3 on SSD portable:

visit https://www.python.org/downloads/windows/

python 3.11.9 could work

in the installer, put to path to somewhere in SSD, like: Z:\Python311
Dont forget to check 'Add python to environemtn variables' in the installer

2- create a venv (recommended)

python -m venv venv

venv\Scripts\activate

if didnt allow, open powershell and type:

Get-ExecutionPolicy

if it yielded: 
Restricted

then type: 
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned

now you should be able to activate venv

3 - install dependencies:

pip install pandas numpy matplotlib opencv-python openpyxl

4 - run:

python .\video_to_frames.py


"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2



def fuse_images(visible_path, thermal_path, transformation_matrix, alpha, output_path):
    """
    Loads visible and thermal images, applies perspective transformation to the thermal image,
    fuses them using weighted blending, and saves the result.
    
    :param visible_path: Path to the visible spectrum image.
    :param thermal_path: Path to the thermal image.
    :param transformation_matrix: Perspective transformation matrix (homography matrix M).
    :param alpha: Weight for blending (0.0 = only thermal, 1.0 = only visible).
    :param output_path: Path to save the fused image.
    """
    new_visible_image = cv2.imread(visible_path)
    new_thermal_image = cv2.imread(thermal_path)
    
    if new_visible_image is None or new_thermal_image is None:
        print(f"Error loading images: {visible_path} or {thermal_path}")
        return
    
    # Warp thermal image to align with visible image
    warped_new_thermal = cv2.warpPerspective(new_thermal_image, transformation_matrix, 
                                             (new_visible_image.shape[1], new_visible_image.shape[0]))
    
    # Perform weighted fusion
    fusion_new_image = cv2.addWeighted(new_visible_image, alpha, warped_new_thermal, 1 - alpha, 0)
    
    # Display the fused image
    cv2.imshow('Fused Image', fusion_new_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # Save the fused image
    cv2.imwrite(output_path, fusion_new_image)
    print(f"Fused image saved to {output_path}")



# Read data from Excel
excel_data = pd.read_excel('manualPoints.xlsx')
visible_points = excel_data[['Visible_X', 'Visible_Y']].values
thermal_points = excel_data[['Thermal_X', 'Thermal_Y']].values

print("Thermal Points Shape:", thermal_points.shape)
print("Visible Points Shape:", visible_points.shape)
# Read visible and thermal images
visible_image = cv2.imread('vs.png')
thermal_image = cv2.imread('ir.png')

# Plotting the manual points in a separate figure
plt.figure(figsize=(12, 6))

# Plotting points on visible image
plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(visible_image, cv2.COLOR_BGR2RGB))
plt.scatter(visible_points[:, 0], visible_points[:, 1], color='red', label='Manual Points')
plt.title('Visible Image with Manual Points')
plt.legend()

# Plotting points on thermal image
plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(thermal_image, cv2.COLOR_BGR2RGB))
plt.scatter(thermal_points[:, 0], thermal_points[:, 1], color='blue', label='Manual Points')
plt.title('Thermal Image with Manual Points')
plt.legend()

plt.tight_layout()
plt.show()


# Calculate transformation matrix using all points
M, _ = cv2.findHomography(np.float32(thermal_points), np.float32(visible_points))

# Invert the transformation matrix
M_inv = cv2.invert(M)[1]

# Warp thermal image based on transformation matrix
warped_thermal = cv2.warpPerspective(thermal_image, M, (visible_image.shape[1], visible_image.shape[0]))


# Fusion and alignment
alpha = 0.7  # Adjust alpha based on desired blending effect (0.0 for only thermal, 1.0 for only visible)
fusion_image = cv2.addWeighted(visible_image, alpha, warped_thermal, 1 - alpha, 0)

print('warped thermal',warped_thermal.shape)
print('infrared image shape:', thermal_image.shape)
print('rgb image shape:', visible_image.shape)
print('fusion image shape:', fusion_image.shape)

# Display or save the fused image
cv2.imshow('Fused Image', fusion_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Save the fused image
cv2.imwrite('fused_image.jpg', fusion_image)

# Display the homography matrix
print("Homography Matrix:")
print(M)

# Replace <depth_value> with whatever depth this homography corresponds to
depth = 6 

# Save both depth and homography in one .npz file
np.savez(f'homography_depth_{depth}.npz', depth=depth, homography=M)

print(f"Saved homography + depth → homography_depth_{depth}.npz")
