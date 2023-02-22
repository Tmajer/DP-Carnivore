import json

import torch, torchvision
import mmpose
import cv2
import numpy as np
from mmpose.apis import (inference_top_down_pose_model, init_pose_model,
                         vis_pose_result, process_mmdet_results)
from mmdet.apis import inference_detector, init_detector
import matplotlib.pyplot as plt
import random

import src.pose_utils as pu

if __name__ == "__main__":
    annotation_data = pu.load_annotations('../CarnivoreID-1/metadata_tiny_cleaned.json')

    annotations = annotation_data.get('annotations')

    annotations = pu.filter_annotations(annotations)

    random.seed(64)
    #random.seed(288)
    random.shuffle(annotations)
    model_name = 'animal'
    pose_model = pu.load_pose_model(model_name)

    for index, annotation in enumerate(annotations):
        bbox = [{'bbox': annotation.get('bbox')}]
        image = cv2.imread(annotation.get('image_path'))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        pose_results, returned_outputs = inference_top_down_pose_model(
            pose_model,
            annotation.get('image_path'),
            bbox,
            format='xywh')

        keypoints, _ = pu.label_animalpose_keypoints(pose_results)

        ground_truth_keypoints, predicted_keypoints, keypoint_visibilities = pu.get_oks_keypoints(
            annotation.get('keypoints'), keypoints)

        for vis_index, visibility in enumerate(keypoint_visibilities):
            if 0 < visibility:
                plt.plot(predicted_keypoints[vis_index][0], predicted_keypoints[vis_index][1], 'ro')
                plt.plot(ground_truth_keypoints[vis_index][0], ground_truth_keypoints[vis_index][1], 'gx')

        for label, kp in keypoints.items():
            plt.text(kp[0], kp[1], label, fontsize=12,
                     ha='center', va='bottom', color='red')

        plt.imshow(image)
        plt.show()

    model_name = 'my'
    pose_model = pu.load_pose_model(model_name)

    for index, annotation in enumerate(annotations):
        bbox = [{'bbox': annotation.get('bbox')}]
        image = cv2.imread(annotation.get('image_path'))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        pose_results, returned_outputs = inference_top_down_pose_model(
            pose_model,
            annotation.get('image_path'),
            bbox,
            format='xywh')

        keypoints, _ = pu.label_animalpose_keypoints(pose_results)

        ground_truth_keypoints, predicted_keypoints, keypoint_visibilities = pu.get_oks_keypoints(
            annotation.get('keypoints'), keypoints)

        for vis_index, visibility in enumerate(keypoint_visibilities):
            if 0 < visibility:
                plt.plot(predicted_keypoints[vis_index][0], predicted_keypoints[vis_index][1], 'ro')
                plt.plot(ground_truth_keypoints[vis_index][0], ground_truth_keypoints[vis_index][1], 'gx')

        for label, kp in keypoints.items():
            plt.text(kp[0], kp[1], label, fontsize=12,
                     ha='center', va='bottom', color='red')

        plt.imshow(image)
        plt.show()

    """
            for label, kp in keypoints.items():
                plt.plot(kp[0], kp[1], 'ro')
                plt.text(kp[0], kp[1], label, fontsize=12, ha='center', va='bottom', color='red')
    """
