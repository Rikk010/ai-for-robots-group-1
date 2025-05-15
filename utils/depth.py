import cv2
import numpy as np

from PIL import Image

def get_depth(pipe, frame, box, normalize=False):
    """
    Get depth of the object in the box
    Higher means closer
    Lower means further
    Args:
        pipe: depth estimation pipeline
        frame: image frame
        box: bounding box (x1, y1, x2, y2)
        normalize: if True, normalize depth to [0, 1]
    """
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    depth = pipe(Image.fromarray(rgb))["depth"]

    # Get mean depth in box
    x1, y1, x2, y2 = box
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    depth = np.array(depth)
    if normalize:
        depth = (depth - np.min(depth)) / (np.max(depth) - np.min(depth))

    # Crop depth to box
    depth = depth[y1:y2, x1:x2]
    depth = np.mean(depth)

    return depth