import numpy as np
from PIL import Image


def create_local_map(global_map, x, y, theta, map_size, scale, s_global, map_center):
    '''
    Creates a local occupancy map from the robot's perspective.

    Parameters:
    - global_map: 2D numpy array representing the global occupancy grid map.
    - x, y: Robot's position in global coordinates (meters).
    - theta: Robot's heading (radians), where 0 means facing the positive x-direction.
    - map_size: Tuple (N, N), desired size of the local map in pixels.
    - scale: Resolution of the local map (meters per pixel).
    - s_global: Resolution of the global map (meters per pixel).

    Returns:
    - local_map: 2D numpy array representing the local occupancy map.
    '''

    # if x is scalar, convert to numpy array
    if isinstance(x, (int, float, np.generic)):
        x = np.array([x])
        y = np.array([y])
        theta = np.array([theta])

    K = len(x)  # Number of robots (batch size)

    # Determine the size of the local map
    if isinstance(map_size, (int, float)):
        N = int(map_size)
    else:
        N = int(map_size[0])  # Size of the local map (N x N)
    L = N * scale  # Physical length of the local map (meters)

    # Create local grid coordinates centered around the robot
    xs = np.linspace(-L / 2 + scale / 2, L / 2 - scale / 2, N)
    ys = np.linspace(-L / 2 + scale / 2, L / 2 - scale / 2, N)
    x_local, y_local = np.meshgrid(xs, ys)

    # Flatten the coordinate grids for vectorized computation
    x_local_flat = x_local.flatten()
    y_local_flat = y_local.flatten()

    # Expand local coordinates to batch size
    x_local_flat_expanded = np.tile(x_local_flat, (K, 1))  # Shape (K, N*N)
    y_local_flat_expanded = np.tile(y_local_flat, (K, 1))  # Shape (K, N*N)

    # Expand robot positions and orientations to match local coordinates
    x_expanded = x[:, np.newaxis]  # Shape (K, 1)
    y_expanded = y[:, np.newaxis]  # Shape (K, 1)
    cos_theta = np.cos(theta)[:, np.newaxis]  # Shape (K, 1)
    sin_theta = np.sin(theta)[:, np.newaxis]  # Shape (K, 1)

    # Compute global coordinates for all robots in the batch (vectorized)
    x_global_flat = cos_theta * x_local_flat_expanded - sin_theta * y_local_flat_expanded + x_expanded  # Shape (K, N*N)
    y_global_flat = sin_theta * x_local_flat_expanded + cos_theta * y_local_flat_expanded + y_expanded  # Shape (K, N*N)

    # using original env method to get indices
    y_indices = np.floor((map_center[1] - y_global_flat) / s_global).astype(int)
    x_indices = np.floor((x_global_flat + map_center[0]) / s_global).astype(int)

    # Handle boundary conditions to prevent index out of bounds
    x_indices = np.clip(x_indices, 0, global_map.shape[1] - 1)  # Assuming shape of global_maps is (K, H, W)
    y_indices = np.clip(y_indices, 0, global_map.shape[0] - 1)

    # Extract occupancy values from the global maps for each robot (vectorized)
    occupancy_values = global_map[y_indices, x_indices]  # Shape (K, N*N)

    # Reshape the occupancy values to form the local maps for each robot
    local_maps = occupancy_values.reshape(K, N, N)  # Shape (K, N, N)

    return local_maps


def png_to_binary_csv(image_path, csv_path, downsample_factor=1):
    """
    Converts a PNG image to a binary CSV file.
    White pixels (RGB 255,255,255) are converted to 0, all others to 1.
    Allows downsampling by a given factor before saving.

    :param image_path: Path to the input PNG image.
    :param csv_path: Path to save the output CSV file.
    :param downsample_factor: Factor by which to downsample the image (default is 1, no downsampling).
    """
    # Open the image and convert it to RGB
    img = Image.open(image_path).convert('RGB')

    # Downsample the image if required
    if downsample_factor > 1:
        new_size = (img.width // downsample_factor, img.height // downsample_factor)
        img = img.resize(new_size)

    img_array = np.array(img)

    # Create a binary matrix: 0 for white pixels, 1 for all others
    binary_matrix = np.where(np.all(img_array == [255, 255, 255], axis=-1), 0, 1)

    # Save to CSV using NumPy
    np.savetxt(csv_path, binary_matrix, fmt='%d', delimiter=',')

    print(f"Binary CSV saved to {csv_path}")
