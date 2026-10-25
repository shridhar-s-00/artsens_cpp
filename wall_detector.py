# wall_detector.py

import numpy as np
import math
from scipy.signal import filtfilt, find_peaks, butter

class WallDetector:

    

    def __init__(self):
        self.b, self.a = butter(
            4,
            800000,
            fs=80e6,
            btype='low'
        )

    def wall_detection(
        self,
        frame_data,
        bounding_box_coordinates,
        artery_threshold=100 // 2
    ):
        scaling_const = 4000 / 1024

        x_1 = bounding_box_coordinates[0]
        x_2 = bounding_box_coordinates[2]

        x_1 = math.floor(x_1 * scaling_const)
        x_2 = math.ceil(x_2 * scaling_const)

        data = np.array(frame_data)

        envelope = filtfilt(self.b, self.a, data)

        data_ROI = data[x_1:x_2]
        envelope_ROI = envelope[x_1:x_2]

        roi_length = len(envelope_ROI)
        mid_point = roi_length // 2

        left_half = envelope_ROI[:mid_point]
        right_half = envelope_ROI[mid_point:]

        left_peaks, _ = find_peaks(left_half, distance=10)

        if len(left_peaks) == 0:
            return [], 0

        left_peak_heights = left_half[left_peaks]

        sorted_indices = np.argsort(left_peak_heights)[::-1]
        top2_indices = sorted_indices[:2]

        top2_peaks = left_peaks[top2_indices]
        top2_heights = left_half[top2_peaks]

        sorted_order = np.argsort(top2_heights)[::-1]

        top2_peaks = top2_peaks[sorted_order]
        top2_heights = top2_heights[sorted_order]

        if len(top2_heights) > 1:
            if top2_heights[1] < 0.5 * top2_heights[0]:
                top2_peaks = top2_peaks[:1]

        best_left_peak_local = np.max(top2_peaks)
        best_left_peak = best_left_peak_local + x_1

        right_peaks, _ = find_peaks(right_half, distance=300)

        if len(right_peaks) == 0:
            return [int(best_left_peak)], 1

        right_peaks_roi = right_peaks + mid_point

        valid_right_peaks = []

        for rp in right_peaks_roi:
            global_rp = rp + x_1

            if abs(global_rp - best_left_peak) >= 300:
                valid_right_peaks.append(rp)

        if len(valid_right_peaks) == 0:
            return [int(best_left_peak)], 1

        valid_right_peaks = np.array(valid_right_peaks)

        right_peak_heights = envelope_ROI[valid_right_peaks]

        best_right_peak_roi = valid_right_peaks[
            np.argmax(right_peak_heights)
        ]

        best_right_peak = best_right_peak_roi + x_1

        return [
            int(best_left_peak),
            int(best_right_peak)
        ], 2
