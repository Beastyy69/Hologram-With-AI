import cv2
import numpy as np

class HologramOverlay:
    def generate_projection(self, frame, size=400):
        canvas = np.zeros((size*3, size*3, 3), dtype=np.uint8)
        square = cv2.resize(frame, (size, size))
        # Orientation logic confirmed by test_feature.py
        canvas[0:size, size:size*2] = cv2.rotate(square, cv2.ROTATE_180)
        canvas[size*2:size*3, size:size*2] = square
        canvas[size:size*2, 0:size] = cv2.rotate(square, cv2.ROTATE_90_CLOCKWISE)
        canvas[size:size*2, size*2:size*3] = cv2.rotate(square, cv2.ROTATE_90_COUNTERCLOCKWISE)
        return canvas