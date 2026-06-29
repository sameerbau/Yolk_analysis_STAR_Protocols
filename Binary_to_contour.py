# -*- coding: utf-8 -*-
"""
Created on Mon Jun 29 16:52:25 2026

@author:Sameer Thukral with Claude code
"""
# -*- coding: utf-8 -*-
"""

"""
"""
Created on Fri Dec 13 17:42:50 2024
This code analyzes kymograph images to extract cell membrane contours.

    It uses vertical line scanning to identify the membrane's position.
    It converts pixel coordinates to physical units (microns and seconds) based on user-defined conversion factors.
    It cleans the contours by removing outliers and filling gaps using interpolation.
    It generates two visualizations for each kymograph:
        Raw detected contour
        Cleaned and processed contour
    It outputs the results as CSV files and saves the visualizations as PNG images.
    It can process multiple kymograph files within a specified folder.

"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import skimage.io as io
from sklearn.neighbors import LocalOutlierFactor
import os

# Conversion factors (can be modified as needed)
PIXEL_TO_MICRON = 0.20  # Example: 1 pixel = 0.5 microns
PIXEL_TO_SECOND = 40  # Example: 1 pixel = 0.1 seconds

# Plot configuration parameters
PLOT_WIDTH_INCHES = 12  # Width of the plot in inches
PLOT_HEIGHT_INCHES = 8   # Height of the plot in inches
PLOT_DPI = 300          # Resolution of the plot
MAX_DISPLAY_WIDTH = 1500  # Maximum width for display scaling (pixels)
MAX_DISPLAY_HEIGHT = 1000 # Maximum height for display scaling (pixels)

# Analysis parameters
SMOOTHING_WINDOW = 2     # Window size for rolling average smoothing
OUTLIER_WINDOW = 10      # Window size for outlier detection
GAP_THRESHOLD = 1       # Threshold for gap detection (in seconds)

def get_folder_path():
    return input("Enter the folder path containing TIF files: ")

def find_vertical_contour(binary_image):
    """Extract contour using vertical line scanning with correct orientation."""
    height, width = binary_image.shape
    contour_points = []
    
    # Scan each vertical line
    for x in range(width):
        column = binary_image[:, x]
        # Find first non-zero point from top (upper membrane)
        nonzero_indices = np.nonzero(column)[0]
        if len(nonzero_indices) > 0:
            # Take the first intersection from top and invert y coordinate
            y = height - nonzero_indices[0]  # Invert y coordinate
            contour_points.append([y, x])
    
    if not contour_points:
        return pd.DataFrame(columns=['Distance_microns', 'Time_seconds'])
    
    # Convert to DataFrame with units
    contour_df = pd.DataFrame(contour_points, columns=['Y', 'X'])
    
    # Convert pixels to physical units
    contour_df['Distance_microns'] = contour_df['Y'] * PIXEL_TO_MICRON
    contour_df['Time_seconds'] = contour_df['X'] * PIXEL_TO_SECOND
    
    # Optional: Smooth the contour using rolling average
    contour_df['Distance_microns'] = contour_df['Distance_microns'].rolling(
        window=SMOOTHING_WINDOW, center=True).mean().bfill().ffill()
    
    return contour_df[['Distance_microns', 'Time_seconds']]

def clean_contour(contour_df):
    """Clean the contour by filling gaps and removing upper outliers."""
    # First, identify and remove points that are significantly above the local mean
    rolling_mean = contour_df['Distance_microns'].rolling(window=OUTLIER_WINDOW, center=True).mean()
    rolling_std = contour_df['Distance_microns'].rolling(window=OUTLIER_WINDOW, center=True).std()
    
    # Remove points that are above the rolling mean by more than 2 standard deviations
    valid_points = abs(contour_df['Distance_microns'] - rolling_mean) <= 2 * rolling_std
    cleaned_df = contour_df[valid_points].copy()
    
    # Sort by time to ensure proper interpolation
    cleaned_df = cleaned_df.sort_values('Time_seconds').reset_index(drop=True)
    
    # Find gaps in time series
    time_diff = cleaned_df['Time_seconds'].diff()
    gap_threshold = GAP_THRESHOLD * PIXEL_TO_SECOND
    
    # Fill gaps using linear interpolation
    if time_diff.max() > gap_threshold:
        # Create a continuous time series
        full_time_range = pd.Series(
            np.arange(
                cleaned_df['Time_seconds'].min(),
                cleaned_df['Time_seconds'].max() + PIXEL_TO_SECOND,
                PIXEL_TO_SECOND
            )
        )
        
        # Reindex and interpolate
        cleaned_df = cleaned_df.set_index('Time_seconds')
        cleaned_df = cleaned_df.reindex(full_time_range)
        cleaned_df['Distance_microns'] = cleaned_df['Distance_microns'].interpolate(
            method='cubic',
            limit_direction='both'
        )
        
        # Reset index to get Time_seconds as a column again
        cleaned_df = cleaned_df.reset_index()
        cleaned_df = cleaned_df.rename(columns={'index': 'Time_seconds'})
    
    # Optional: Apply light smoothing to reduce noise while preserving shape
    cleaned_df['Distance_microns'] = cleaned_df['Distance_microns'].rolling(
        window=SMOOTHING_WINDOW, center=True, min_periods=1
    ).mean()
    
    return cleaned_df

def calculate_display_dimensions(image_shape):
    """Calculate appropriate display dimensions while maintaining aspect ratio."""
    height, width = image_shape
    aspect_ratio = width / height
    
    if width > MAX_DISPLAY_WIDTH:
        new_width = MAX_DISPLAY_WIDTH
        new_height = int(new_width / aspect_ratio)
    elif height > MAX_DISPLAY_HEIGHT:
        new_height = MAX_DISPLAY_HEIGHT
        new_width = int(new_height * aspect_ratio)
    else:
        new_width = width
        new_height = height
    
    return new_width / 100, new_height / 100  # Convert to inches

def plot_and_save_contours(image, raw_contour, cleaned_contour, output_path_raw, output_path_cleaned):
    """Plot and save both raw and cleaned contours with proper units."""
    height, width = image.shape
    extent = [0, width * PIXEL_TO_SECOND, 0, height * PIXEL_TO_MICRON]
    
    # Calculate display dimensions
    plot_width, plot_height = calculate_display_dimensions(image.shape)
    
    # Plot raw contour
    plt.figure(figsize=(PLOT_WIDTH_INCHES, PLOT_HEIGHT_INCHES))
    plt.imshow(image, origin='upper', cmap='gray', extent=extent, aspect='auto')
    plt.plot(raw_contour['Time_seconds'], raw_contour['Distance_microns'], 
            'g.', color='red', markersize=12, label='Raw Contour')
    plt.title('Raw Contour Detection')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Distance (μm)')
    plt.legend()
    # Remove grid (this removes the white grid lines)
    plt.grid(False)
    plt.tight_layout()
    plt.savefig(output_path_raw, dpi=PLOT_DPI, bbox_inches='tight')
    plt.close()
    
    # Plot cleaned contour
    plt.figure(figsize=(PLOT_WIDTH_INCHES, PLOT_HEIGHT_INCHES))
    plt.imshow(image, origin='upper', cmap='gray', extent=extent, aspect='auto')
    plt.plot(cleaned_contour['Time_seconds'], cleaned_contour['Distance_microns'], 
            'r.', label='Cleaned Contour')
    plt.title('Cleaned Contour Detection')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Distance (μm)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path_cleaned, dpi=PLOT_DPI, bbox_inches='tight')
    plt.close()

def process_kymograph(file_path):
    """Process a single kymograph file."""
    try:
        # Load image
        image = io.imread(file_path)
        
        # Get raw contour
        raw_contour = find_vertical_contour(image)
        if raw_contour.empty:
            print(f"No contour found in {file_path}")
            return
              
        # Clean contour
        cleaned_contour = clean_contour(raw_contour)
        
        # Generate output paths
        base_name = os.path.splitext(file_path)[0]
        output_csv = f"{base_name}_results.csv"
        output_plot_raw = f"{base_name}_raw_contour.png"
        output_plot_cleaned = f"{base_name}_cleaned_contour.png"
        
        # Save results
        raw_contour.to_csv(f"{base_name}_raw_results.csv", index=False)     
        cleaned_contour.to_csv(output_csv, index=False)
        plot_and_save_contours(image, raw_contour, cleaned_contour, 
                             output_plot_raw, output_plot_cleaned)
        
        print(f"Processed {os.path.basename(file_path)}")
        print(f"Saved: {os.path.basename(output_csv)}")
        print(f"Saved: {os.path.basename(output_plot_raw)}")
        print(f"Saved: {os.path.basename(output_plot_cleaned)}")
        
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")

def main():
    """Main function to process all kymographs in a folder."""
    folder_path = get_folder_path()
    
    print("\nStarting kymograph processing...")
    for filename in os.listdir(folder_path):
        if filename.endswith((".tif", ".tiff")):
            file_path = os.path.join(folder_path, filename)
            print(f"\nProcessing: {filename}")
            process_kymograph(file_path)
    
    print("\nProcessing complete!")

if __name__ == "__main__":
    main()