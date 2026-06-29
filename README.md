This code analyzes kymograph images to extract kymograph contours.

    It uses vertical line scanning to identify the membrane's position.
    It converts pixel coordinates to physical units (microns and seconds) based on user-defined conversion factors.
    It cleans the contours by removing outliers and filling gaps using interpolation.
    It generates two visualizations for each kymograph:
        Raw detected contour
        Cleaned and processed contour
    It outputs the results as CSV files and saves the visualizations as PNG images.
    It can process multiple kymograph files within a specified folder.

    Input: Binarized Kymographs geneted from Fiji. Original image: SampleA, Binarized image: BinarizedA 
