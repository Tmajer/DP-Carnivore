import math
import os
import pickle
import statistics
import time
from enum import Enum

import cv2
import numpy as np
from matplotlib import pyplot as plt
from mmpose.apis import (inference_top_down_pose_model)

import src.pose_utils as pu


def oks_scalesqrt(ground_truth_keypoints, predicted_keypoints, scale, visibilities, keypoint_weights, correction_factor=1):
    similarity_num = 0
    similarity_denum = 0
    for index, visibility in enumerate(visibilities):
        if 0 < visibility:
            dist = np.linalg.norm(predicted_keypoints[index, :] - ground_truth_keypoints[index, :])
            similarity_num += math.exp(-dist**2 / (2 * scale * correction_factor * keypoint_weights[index]**2))
            similarity_denum += 1
    if similarity_denum == 0:
        # what to return here
        return 0
    else:
        return similarity_num / similarity_denum


def oks(ground_truth_keypoints, predicted_keypoints, scale, visibilities, keypoint_weights, correction_factor=1):
    similarity_num = 0
    similarity_denum = 0
    for index, visibility in enumerate(visibilities):
        if 0 < visibility:
            dist = np.linalg.norm(predicted_keypoints[index, :] - ground_truth_keypoints[index, :])
            similarity_num += math.exp(-dist**2 / (2 * scale**2 * correction_factor * keypoint_weights[index]**2))
            similarity_denum += 1
    if similarity_denum == 0:
        # what to return here
        return 0
    else:
        return similarity_num / similarity_denum


def oks_standard(ground_truth_keypoints, predicted_keypoints, scale, visibilities, keypoint_weights, image_ratio, correction_factor=1):
    similarity_num = 0
    similarity_denum = 0
    for index, visibility in enumerate(visibilities):
        if 0 < visibility:
            dist = np.linalg.norm(predicted_keypoints[index, :] - ground_truth_keypoints[index, :])
            similarity_num += math.exp(-dist ** 2 / (2 * correction_factor * image_ratio * scale**2 * keypoint_weights[index] ** 2))
            similarity_denum += 1
    if similarity_denum == 0:
        # what to return here
        return 0
    else:
        return similarity_num / similarity_denum


class OksType(Enum):
    OKSSTAND = 2
    OKSSQRT = 1
    OKSNORM = 0


class OksExperimentResult:
    def __init__(self):
        self.per_image_oks = {}
        self.zero_oks_files = []
        self.absolute_nonzero_count = 0
        self.relative_nonzero_score = 0
        self.total_oks = 0
        self.scaling_constant = 0
        self.average_oks = 0

    def compute_relative_nonzero_score(self, total_count):
        self.relative_nonzero_score = self.absolute_nonzero_count / total_count

    def compute_average_oks(self, total_count):
        self.average_oks = self.total_oks / total_count


class OksExperiment:
    def __init__(self, oks_type: OksType):
        self.oks_experiment_results: list[OksExperimentResult] = []
        self.oks_type = oks_type
        self.number_of_images = 0

    def run_experiment(self, annotations, predictions, image_sizes, constants, sigmas):
        self.number_of_images = len(annotations)
        for scaling_constant in constants:
            print(f"Running for scaling constant {scaling_constant}")
            oks_experiment_result = OksExperimentResult()
            oks_experiment_result.scaling_constant = scaling_constant

            for annotation in annotations:
                self.oks_scale_experiment_iter(oks_experiment_result, annotation, predictions, image_sizes, scaling_constant, sigmas)

            oks_experiment_result.compute_relative_nonzero_score(self.number_of_images)
            oks_experiment_result.compute_average_oks(self.number_of_images)
            self.oks_experiment_results.append(oks_experiment_result)

    def oks_scale_experiment_iter(self, oks_experiment_result: OksExperimentResult, annotation, predictions, image_sizes, scaling_constant, sigmas):
        ground_truth_keypoints, predicted_keypoints, keypoint_visibilities = pu.get_oks_keypoints(
            annotation.get('keypoints'), predictions.get(annotation.get('image_path')))

        if self.oks_type.value == 2:
            im_size = image_sizes.get(annotation.get('image_path'))
            median_size = image_sizes.get('median')
            image_ratio = median_size / im_size
            box_size = annotation.get('bbox')[2] * annotation.get('bbox')[3]
            object_scale = math.sqrt(box_size)
            oks_score = oks_standard(ground_truth_keypoints, predicted_keypoints, object_scale, keypoint_visibilities,
                                     sigmas, image_ratio, scaling_constant)
        else:
            im_size = image_sizes.get(annotation.get('image_path'))
            box_size = annotation.get('bbox')[2] * annotation.get('bbox')[3]
            object_scale = box_size / im_size

            if self.oks_type.value:
                oks_score = oks_scalesqrt(ground_truth_keypoints, predicted_keypoints, object_scale, keypoint_visibilities,
                            sigmas, scaling_constant)
            else:
                oks_score = oks(ground_truth_keypoints, predicted_keypoints, object_scale, keypoint_visibilities,
                            sigmas, scaling_constant)

        oks_experiment_result.per_image_oks[annotation.get('image_path')] = oks_score

        if 0 < oks_score:
            oks_experiment_result.absolute_nonzero_count += 1
            oks_experiment_result.total_oks += oks_score
        else:
            oks_experiment_result.zero_oks_files.append(annotation)


def compute_predictions(annotations, pose_model):
    predictions = {}
    for annotation in annotations:
        bbox = [{'bbox': annotation.get('bbox')}]

        pose_results, returned_outputs = inference_top_down_pose_model(
            pose_model,
            annotation.get('image_path'),
            bbox,
            format='xywh')

        keypoints, _ = pu.label_animalpose_keypoints(pose_results)
        predictions[annotation.get('image_path')] = keypoints
    return predictions


if __name__ == "__main__":
    annotation_data = pu.load_annotations('../CarnivoreID-1/metadata_cleaned.json')
    annotations = annotation_data.get('annotations')
    annotations = pu.filter_annotations(annotations)

    predictions_file = os.path.join("..", "results", "predictions.obj")
    image_sizes_file = os.path.join("..", "results", "image_sizes.obj")
    filename_start = os.path.join("..", "results", "oks_result_")

    if not os.path.exists(image_sizes_file):
        image_sizes = {}
        for annotation in annotations:
            image = cv2.imread(annotation.get('image_path'))
            im_size = image.shape[0] * image.shape[1]
            image_sizes[annotation.get('image_path')] = im_size
        with open(image_sizes_file, "wb") as f:
            pickle.dump(image_sizes, f)
    else:
        with open(image_sizes_file, 'rb') as f:
            image_sizes = pickle.load(f)

    median_image_size = statistics.median(image_sizes.values())
    image_sizes['median'] = median_image_size

    if not os.path.exists(predictions_file):
        model_name = 'animal'
        pose_model = pu.load_pose_model(model_name)
        predictions = compute_predictions(annotations, pose_model)
        with open(predictions_file, "wb") as f:
            pickle.dump(predictions, f)
    else:
        with open(predictions_file, 'rb') as f:
            predictions = pickle.load(f)

    sigmas_ones = np.ones((20,))
    sigmas_ap10k = np.array([0.025, 0.025, 0.026, 0.035, 0.035, 0.10, 0.10, 0.10, 0.107, 0.107, 0.107, 0.107, 0.087,
                             0.087, 0.087, 0.087, 0.089, 0.089, 0.089, 0.089], dtype=np.float64)

    if not os.path.exists(f"{filename_start}1.obj"):
        print("Experiment 1 running")
        start = time.time()
        multipliers = [1, 3, 5, 7, 9]
        multiplicands = [1, 10, 100, 1000, 10_000, 100_000]
        constants_exp = [i * j for i in multipliers for j in multiplicands]
        oks_experiment_1 = OksExperiment(OksType.OKSSQRT)
        oks_experiment_1.run_experiment(annotations, predictions, image_sizes, constants_exp, sigmas_ones)
        with open(f"{filename_start}1.obj", "wb") as f:
            pickle.dump(oks_experiment_1, f)
        end = time.time()
        print(f'Experiment 1 finished running in {end - start} seconds')

    if not os.path.exists(f"{filename_start}2.obj"):
        print("Experiment 2 running")
        start = time.time()
        multipliers = [1, 3, 5, 7, 9]
        multiplicands = [1, 10, 100, 1000, 10_000, 100_000, 1_000_000]
        constants_exp = [i * j for i in multipliers for j in multiplicands]
        oks_experiment_2 = OksExperiment(OksType.OKSSQRT)
        oks_experiment_2.run_experiment(annotations, predictions, image_sizes, constants_exp, sigmas_ap10k)
        with open(f"{filename_start}2.obj", "wb") as f:
            pickle.dump(oks_experiment_2, f)
        end = time.time()
        print(f'Experiment 2 finished running in {end - start} seconds')

    if not os.path.exists(f"{filename_start}3.obj"):
        print("Experiment 3 running")
        start = time.time()
        multipliers = [1, 3, 5, 7, 9]
        multiplicands = [1, 10, 100, 1000, 10_000, 100_000, 1_000_000, 10_000_000]
        constants_exp = [i * j for i in multipliers for j in multiplicands]
        oks_experiment_3 = OksExperiment(OksType.OKSNORM)
        oks_experiment_3.run_experiment(annotations, predictions, image_sizes, constants_exp, sigmas_ones)
        with open(f"{filename_start}3.obj", "wb") as f:
            pickle.dump(oks_experiment_3, f)
        end = time.time()
        print(f'Experiment 3 finished running in {end - start} seconds')

    if not os.path.exists(f"{filename_start}4.obj"):
        print("Experiment 4 running")
        start = time.time()
        multipliers = [1, 3, 5, 7, 9]
        multiplicands = [1, 10, 100, 1000, 10_000, 100_000, 1_000_000, 10_000_000, 100_000_000, 1_000_000_000]
        constants_exp = [i * j for i in multipliers for j in multiplicands]
        oks_experiment_4 = OksExperiment(OksType.OKSNORM)
        oks_experiment_4.run_experiment(annotations, predictions, image_sizes, constants_exp, sigmas_ap10k)
        with open(f"{filename_start}4.obj", "wb") as f:
            pickle.dump(oks_experiment_4, f)
        end = time.time()
        print(f'Experiment 4 finished running in {end - start} seconds')

    if not os.path.exists(f"{filename_start}5.obj"):
        print("Experiment 5 running")
        start = time.time()
        multipliers = [1, 3, 5, 7, 9]
        multiplicands = [0.001, 0.01, 0.1, 1, 10, 100, 1000, 10_000]
        constants_exp = [i * j for i in multipliers for j in multiplicands]
        oks_experiment_5 = OksExperiment(OksType.OKSSTAND)
        oks_experiment_5.run_experiment(annotations, predictions, image_sizes, constants_exp, sigmas_ones)
        with open(f"{filename_start}5.obj", "wb") as f:
            pickle.dump(oks_experiment_5, f)
        end = time.time()
        print(f'Experiment 5 finished running in {end - start} seconds')

    if not os.path.exists(f"{filename_start}6.obj"):
        print("Experiment 6 running")
        start = time.time()
        multipliers = [1, 3, 5, 7, 9]
        multiplicands = [0.001, 0.01, 0.1, 1, 10, 100, 1000, 10_000, 100_000]
        constants_exp = [i * j for i in multipliers for j in multiplicands]
        oks_experiment_6 = OksExperiment(OksType.OKSSTAND)
        oks_experiment_6.run_experiment(annotations, predictions, image_sizes, constants_exp, sigmas_ap10k)
        with open(f"{filename_start}6.obj", "wb") as f:
            pickle.dump(oks_experiment_6, f)
        end = time.time()
        print(f'Experiment 6 finished running in {end - start} seconds')

    with open(f"{filename_start}1.obj", 'rb') as f:
        open_exp1: OksExperiment = pickle.load(f)
    with open(f"{filename_start}2.obj", 'rb') as f:
        open_exp2: OksExperiment = pickle.load(f)
    with open(f"{filename_start}3.obj", 'rb') as f:
        open_exp3: OksExperiment = pickle.load(f)
    with open(f"{filename_start}4.obj", 'rb') as f:
        open_exp4: OksExperiment = pickle.load(f)
    with open(f"{filename_start}5.obj", 'rb') as f:
        open_exp5: OksExperiment = pickle.load(f)
    with open(f"{filename_start}6.obj", 'rb') as f:
        open_exp6: OksExperiment = pickle.load(f)

    print(f"Average OKS for standard OKS computation with ones {open_exp5.oks_experiment_results[0].average_oks}")
    print(f"Nonzero score for standard OKS computation with ones {open_exp5.oks_experiment_results[0].relative_nonzero_score}")
    print(f"Average OKS for standard OKS computation with ap10k sigmas {open_exp6.oks_experiment_results[0].average_oks}")
    print(f"Nonzero score for standard OKS computation with ap10k sigmas {open_exp6.oks_experiment_results[0].relative_nonzero_score}")

    experiment_results = [open_exp1, open_exp2, open_exp3, open_exp4, open_exp5, open_exp6]
    for experiment_result in experiment_results:
        experiment_result.oks_experiment_results.sort(key=lambda x: x.scaling_constant)

    for index, experiment_result in enumerate(experiment_results):
        scales = []
        nonzero_scores = []
        average_oks = []

        results = experiment_result.oks_experiment_results

        for result in results:
            scales.append(result.scaling_constant)
            nonzero_scores.append(result.relative_nonzero_score)
            average_oks.append(result.average_oks)

        plt.subplot(3, 2, index + 1)
        plt.title(experiment_result.oks_type)
        plt.plot(scales, nonzero_scores, label="nonzero_scores")
        plt.plot(scales, average_oks, label="average oks")
        plt.legend()
        plt.xscale("log")
        plt.grid(True)
        #plt.xlabel("Correction factor")

    plt.show()

    weird_image_annot = open_exp6.oks_experiment_results[15].zero_oks_files[0]

    bbox = weird_image_annot.get('bbox')
    image = cv2.imread(weird_image_annot.get('image_path'))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    box_size = bbox[2] * bbox[3]
    object_scale = math.sqrt(box_size)

    predict_for_weirdim = predictions[weird_image_annot.get('image_path')]

    ground_truth_keypoints, predicted_keypoints, keypoint_visibilities = pu.get_oks_keypoints(
        weird_image_annot.get('keypoints'), predict_for_weirdim, keep_predictions=True)

    oks_for_weird = oks_standard(ground_truth_keypoints, predicted_keypoints, object_scale, keypoint_visibilities, sigmas_ap10k, 1, 1)
    plt.imshow(image)
    for vis_index, visibility in enumerate(keypoint_visibilities):
        plt.plot(predicted_keypoints[vis_index][0], predicted_keypoints[vis_index][1], 'ro')
        #plt.text(kp[0], kp[1], label, fontsize=12, ha='center', va='bottom', color='red')
        if 0 < visibility:
            plt.plot(ground_truth_keypoints[vis_index][0], ground_truth_keypoints[vis_index][1], 'gx')


    plt.show()

    a = 5
