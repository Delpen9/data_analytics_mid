# standard libraries
import os
import numpy as np
import math

# image manipulation
import cv2

if __name__ == '__main__':
    current_path = os.path.abspath(__file__)
    image_file_path = os.path.abspath(os.path.join(current_path, '..', '..', 'data', 'lotus-1.jpg'))

    image = cv2.imread(image_file_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    new_image_shape = (700, 800)

    output = np.zeros(new_image_shape, dtype = np.uint8)

    for y in range(new_image_shape[0]):
        for x in range(new_image_shape[1]):
            x_input_pos = int(x * gray.shape[1] / new_image_shape[1])
            y_input_pos = int(y * gray.shape[0] / new_image_shape[0])
            
            # =============================
            # Handle corner cases
            # =============================
            corner_condition_1 = (x_input_pos == 0) and (y_input_pos == 0)
            corner_condition_2 = (x_input_pos == gray.shape[1] - 1) and (y_input_pos == 0)
            corner_condition_3 = (x_input_pos == 0) and (y_input_pos == gray.shape[0] - 1)
            corner_condition_4 = (x_input_pos == gray.shape[1] - 1) and (y_input_pos == gray.shape[0] - 1)

            combined_corner_condition = corner_condition_1 or corner_condition_2 or corner_condition_3 or corner_condition_4

            if combined_corner_condition:
                output[y, x] = gray[y_input_pos, x_input_pos]
            # =============================

            # =============================
            # Handle edge cases
            # =============================
            x_edge_condition_1 = (x_input_pos == 0)
            x_edge_condition_2 = (x_input_pos == gray.shape[1] - 1)

            y_edge_condition_1 = (y_input_pos == 0)
            y_edge_condition_2 = (y_input_pos == gray.shape[0] - 1)

            combined_x_edge_condition = x_edge_condition_1 or x_edge_condition_2
            combined_y_edge_condition = y_edge_condition_1 or y_edge_condition_2
            
            if combined_x_edge_condition and not combined_corner_condition:
                output[y, x] = 0.5 * gray[y_input_pos + 1, x_input_pos] +\
                            0.5 * gray[y_input_pos - 1, x_input_pos]
            if combined_y_edge_condition and not combined_corner_condition:
                output[y, x] = 0.5 * gray[y_input_pos, x_input_pos + 1] +\
                            0.5 * gray[y_input_pos, x_input_pos - 1]
            # =============================

            # =============================
            # Handle normal cases
            # =============================
            normal_condition = not combined_x_edge_condition and not combined_y_edge_condition and not combined_corner_condition
            if normal_condition:
                _R1 = 0.5 * gray[y_input_pos - 1, x_input_pos - 1] +\
                    0.5 * gray[y_input_pos - 1, x_input_pos + 1]

                _R2 = 0.5 * gray[y_input_pos + 1, x_input_pos - 1] +\
                    0.5 * gray[y_input_pos + 1, x_input_pos + 1]

                output[y, x] = 0.5 * _R1 + 0.5 * _R2
            # =============================
            
    image_file_path = os.path.abspath(os.path.join(current_path, '..', '..', 'output', 'lotus_resized.png'))
    cv2.imwrite(image_file_path, output)
