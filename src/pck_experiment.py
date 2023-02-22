import os
import pickle
import time

import cv2
import numpy as np
from matplotlib import pyplot as plt

import pose_utils as pu
from rescaled_oks_experiment import compute_predictions

if __name__ == "__main__":
    annotation_data = pu.load_annotations('../CarnivoreID-1/metadata_cleaned.json')
    annotations = annotation_data.get('annotations')
    annotations = pu.filter_annotations(annotations)

    predictions_file = os.path.join("..", "results", "predictions.obj")
    filename_start = os.path.join("..", "results", "oks_result_")

    if not os.path.exists(predictions_file):
        model_name = 'animal'
        pose_model = pu.load_pose_model(model_name)
        predictions = compute_predictions(annotations, pose_model)
        with open(predictions_file, "wb") as f:
            pickle.dump(predictions, f)
    else:
        with open(predictions_file, 'rb') as f:
            predictions = pickle.load(f)

    scale = 1
    thresholds = [1.3**i for i in range(-5, 32)]

    final_pcks = {}
    relative_correct = []

    print("Running PCK computations for thresholds")
    for threshold in thresholds:
        start = time.time()
        # Init at ones to avoid zero division
        detected_counts = np.ones((20,))
        keypoint_counts = np.ones((20,))
        number_of_correct = 0
        for annotation in annotations:
            ground_truth_keypoints, predicted_keypoints, keypoint_visibilities = pu.get_oks_keypoints(
                annotation.get('keypoints'), predictions.get(annotation.get('image_path')))

            keypoint_counts = keypoint_counts + keypoint_visibilities
            annotation_pck = pu.pck(ground_truth_keypoints, predicted_keypoints, keypoint_visibilities, scale, threshold)
            detected_counts = detected_counts + annotation_pck
            number_of_correct += pu.kps_within_threshold(ground_truth_keypoints, predicted_keypoints, keypoint_visibilities, scale, threshold)

        relative_correct.append(number_of_correct / len(annotations))
        final_pcks[threshold] = np.divide(detected_counts, keypoint_counts)
        end = time.time()
        print(f'Computation with threshold {threshold} finished running in {end - start} seconds')
    with open(os.path.join("..", "results", "pcks.obj"), 'wb') as f:
        pickle.dump(final_pcks, f)

    plt.plot(thresholds, relative_correct, label="relative correct")
    plt.legend()
    plt.grid(True)
    plt.show()
