import numpy as np
import tifffile as tiff

# Constants
width, height = 512, 512
fps = 30
duration = 120  # seconds
frames = fps * duration  # Total frames = 3600
circle_radius = 40
initial_brightness = 50  # Start with some brightness

# Define new positions based on your request
center = (width // 2, height // 2)
corners = [(50, 50), (462, 50), (50, 462), (462, 462)]  # 4 corners
pair = [(180, 256), (332, 256)]  # Two circles farther apart horizontally

# Function for fast brightness rise and fall (triangular wave)
def pulse_brightness(frequency, frame, max_brightness=255, min_brightness=50):
    """Creates a fast-rise, fast-fall brightness pulse effect using a triangular wave."""
    cycle_pos = (frame % frequency) / frequency  # Cycle position (0 to 1)
    if cycle_pos < 0.5:
        brightness = min_brightness + (max_brightness - min_brightness) * (cycle_pos * 2)
    else:
        brightness = max_brightness - (max_brightness - min_brightness) * ((cycle_pos - 0.5) * 2)
    return int(brightness)

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
        
        # Plane 1: Central circle - Brightens every 10s
        brightness1 = pulse_brightness(10 * fps, f)
        draw_circle(frame[0], center, brightness1)
        
        # Plane 2: Corner circles brightness with new cycles
        brightness2_top_left = pulse_brightness(60 * fps, f)  # 60s cycle
        brightness2_bottom_left = pulse_brightness(30 * fps, f)  # 30s cycle
        brightness2_top_right = pulse_brightness(5 * fps, f)  # 5s cycle
        brightness2_bottom_right = pulse_brightness(20 * fps, f)  # 20s cycle
        draw_circle(frame[1], corners[0], brightness2_top_left)
        draw_circle(frame[1], corners[2], brightness2_bottom_left)
        draw_circle(frame[1], corners[1], brightness2_top_right)
        draw_circle(frame[1], corners[3], brightness2_bottom_right)
        
        # Plane 3: Two circles - different cycles
        brightness3_left = pulse_brightness(15 * fps, f)  # Left circle: 15s cycle
        brightness3_right = pulse_brightness(50 * fps, f)  # Right circle: 50s cycle
        draw_circle(frame[2], pair[0], brightness3_left)
        draw_circle(frame[2], pair[1], brightness3_right)
        
        # Save frame incrementally
        tif.write(frame, photometric='minisblack')

print(f"TIFF file saved as {file_path}")
