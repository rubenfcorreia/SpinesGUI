import numpy as np
import tifffile as tiff
import os
# Constants
width, height = 512, 512
fps = 30
duration = 120  # seconds
frames = fps * duration  # Total frames = 3600
circle_radius = 40

# Define positions
center = (width // 2, height // 2)
corners = [(50, 50), (462, 50), (50, 462), (462, 462)]  # 4 corners
pair = [(200, 256), (282, 256)]  # Close circles

# Function to create oscillating brightness pattern
def oscillate(frequency, frame, max_brightness=255):
    """Create an oscillating brightness pattern using a sine wave."""
    return int((np.sin(2 * np.pi * frame / frequency) + 1) / 2 * max_brightness)

# Function to draw a circle
def draw_circle(frame, position, brightness):
    """Draw a filled circle on the given frame at the specified position with the given brightness."""
    y, x = np.ogrid[:height, :width]
    mask = (x - position[0])**2 + (y - position[1])**2 <= circle_radius**2
    frame[mask] = brightness

# Define file path
file_path = "simulated_data.tiff"

# Open TIFF file for writing in an incremental manner
with tiff.TiffWriter(file_path, bigtiff=True) as tif:
    for f in range(frames):
        # Initialize single frame with 3 planes
        frame = np.zeros((3, height, width), dtype=np.uint8)
        
        # Plane 1: Central circle oscillates every 10s
        brightness1 = oscillate(10 * fps, f)
        draw_circle(frame[0], center, brightness1)
        
        # Plane 2: Corner circles brightness
        brightness2_left = oscillate(30 * fps, f)
        brightness2_right = oscillate(5 * fps, f)
        draw_circle(frame[1], corners[0], brightness2_left)
        draw_circle(frame[1], corners[2], brightness2_left)
        draw_circle(frame[1], corners[1], brightness2_right)
        draw_circle(frame[1], corners[3], brightness2_right)
        
        # Plane 3: Pair of circles brightness oscillates every 50s
        brightness3 = oscillate(50 * fps, f)
        draw_circle(frame[2], pair[0], brightness3)
        draw_circle(frame[2], pair[1], brightness3)
        
        # Save frame incrementally
        tif.write(frame, photometric='minisblack')

print(f"TIFF file saved as {file_path} in {os.os.getcwd()}")
