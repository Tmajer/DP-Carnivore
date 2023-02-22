import numpy as np
import math, os
import json
import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from mmpose.apis import init_pose_model


def oks(ground_truth_keypoints, predicted_keypoints, scale, visibilities, keypoint_weights, scaling_const=1):
    similarity_num = 0
    similarity_denum = 0
    for index, visibility in enumerate(visibilities):
        if 0 < visibility:
            dist = np.linalg.norm(predicted_keypoints[index, :] - ground_truth_keypoints[index, :])
            similarity_num += math.exp(-dist**2 / (2 * scale * scaling_const * keypoint_weights[index]**2))
            similarity_denum += 1
    if similarity_denum == 0:
        # what to return here
        return 0
    else:
        return similarity_num / similarity_denum


def pck(ground_truth_keypoints, predicted_keypoints, visibilities, scale, threshold):
    correct_keypoints = np.zeros((len(visibilities)))
    for index, visibility in enumerate(visibilities):
        distance = np.linalg.norm(predicted_keypoints[index, :] - ground_truth_keypoints[index, :])
        if distance / scale < threshold and visibility:
            correct_keypoints[index] = 1
        else:
            correct_keypoints[index] = 0
    return correct_keypoints


def kps_within_threshold(ground_truth_keypoints, predicted_keypoints, visibilities, scale, threshold):
    for index, visibility in enumerate(visibilities):
        distance = np.linalg.norm(predicted_keypoints[index, :] - ground_truth_keypoints[index, :])
        if (distance / scale < threshold):
            continue
        elif visibility:
            return 0
    return 1


def load_annotations(filename):
    with open(filename) as json_file:
        data = json.load(json_file)
    return data


def filter_annotations(annotations):
    # Throws out annotation with NOT_FOR_POSE tag and changes filepath
    filtered_annotations = [d for d in annotations if d.get('tag') != 'NOT_FOR_POSE' and d.get('keypoints')]

    for annotation in filtered_annotations:
        annotation['image_path'] = f'../CarnivoreID-1/{annotation["image_id"]}.jpg'

    return filtered_annotations


def label_animalpose_keypoints(pose_results):
    # Annotates keypoints with their names
    keypoints = pose_results[0].get('keypoints')
    labeled_keypoints = {'left-eye': (keypoints[0][0], keypoints[0][1]),
                 'right-eye': (keypoints[1][0], keypoints[1][1]),
                 'left-earbase': (keypoints[2][0], keypoints[2][1]),
                 'right-earbase': (keypoints[3][0], keypoints[3][1]),
                 'nose': (keypoints[4][0], keypoints[4][1]),
                 'throat': (keypoints[5][0], keypoints[5][1]),
                 'tailbase': (keypoints[6][0], keypoints[6][1]),
                 'withers': (keypoints[7][0], keypoints[7][1]),
                 'L-F-elbow': (keypoints[8][0], keypoints[8][1]),
                 'R-F-elbow': (keypoints[9][0], keypoints[9][1]),
                 'L-B-knee': (keypoints[10][0], keypoints[10][1]),
                 'R-B-knee': (keypoints[11][0], keypoints[11][1]),
                 'L-F-wrist': (keypoints[12][0], keypoints[12][1]),
                 'R-F-wrist': (keypoints[13][0], keypoints[13][1]),
                 'L-B-ankle': (keypoints[14][0], keypoints[14][1]),
                 'R-B-ankle': (keypoints[15][0], keypoints[15][1]),
                 'L-F-paw': (keypoints[16][0], keypoints[16][1]),
                 'R-F-paw': (keypoints[17][0], keypoints[17][1]),
                 'L-B-paw': (keypoints[18][0], keypoints[18][1]),
                 'R-B-paw': (keypoints[19][0], keypoints[19][1])
                 }
    keypoint_credibility = {'left-eye': keypoints[0][2], 'right-eye': keypoints[1][2],
                            'left-earbase': keypoints[2][2], 'right-earbase': keypoints[3][2],
                            'nose': keypoints[4][2], 'throat': keypoints[5][2],
                            'tailbase': keypoints[6][2], 'withers': keypoints[7][2],
                            'L-F-elbow': keypoints[8][2], 'R-F-elbow': keypoints[9][2],
                            'L-B-knee': keypoints[10][2], 'R-B-knee': keypoints[11][2],
                            'L-F-wrist': keypoints[12][2], 'R-F-wrist': keypoints[13][2],
                            'L-B-ankle': keypoints[14][2], 'R-B-ankle': keypoints[15][2],
                            'L-F-paw': keypoints[16][2], 'R-F-paw': keypoints[17][2],
                            'L-B-paw': keypoints[18][2], 'R-B-paw': keypoints[19][2]
                            }
    return labeled_keypoints, keypoint_credibility


def load_pose_model(model_name):
    if model_name == 'atrw':
        config_file = '../res50_atrw_256x256.py'
        checkpoint_file = '../res50_atrw_256x256-546c4594_20210414.pth'
    elif model_name == 'animal':
        config_file = '../hrnet_w32_animalpose_256x256.py'
        checkpoint_file = '../hrnet_w32_animalpose_256x256-1aa7f075_20210426.pth'
    elif model_name == "my":
        config_file = './hrnet_w32_CARNIVORE_256x256.py'
        checkpoint_file = './work_dir/epoch_10.pth'
    return init_pose_model(config_file, checkpoint_file, device='cuda:0')  # or device='cuda:0'


def show_image_with_keypoints(annotation, bb=False):
    image = cv2.imread(annotation.get('image_path'))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    for (x, y) in annotation.get('keypoints').values():
        plt.plot(x, y, 'ro')

    plt.imshow(image)
    if bb:
        bbox = annotation.get('bbox')
        bbox = Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], linewidth=2, edgecolor='r', facecolor='none')
        plt.gca().add_patch(bbox)
    plt.show()


def bbox_convert(bbox):
    # Converts bounding box from x_min, y_min, width, height to x_min, y_min, x_max, y_max
    return np.ndarray([bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]], dtype=np.float32)


def get_oks_keypoints(ground_truth_dict: dict, pred_dict, keep_predictions=False):
    index_map = {'left-eye': 0, 'right-eye': 1, 'left-earbase': 2, 'right-earbase': 3, 'nose': 4, 'throat': 5,
                 'tailbase': 6, 'withers': 7, 'L-F-elbow': 8, 'R-F-elbow': 9, 'L-B-knee': 10, 'R-B-knee': 11,
                 'L-F-wrist': 12, 'R-F-wrist': 13, 'L-B-ankle': 14, 'R-B-ankle': 15, 'L-F-paw': 16, 'R-F-paw': 17,
                 'L-B-paw': 18, 'R-B-paw': 19
                 }
    ground_truth_keypoints = np.zeros((20, 2))
    predicted_keypoints = np.zeros((20, 2))
    keypoint_visibilities = np.zeros((20,))

    if keep_predictions:
        for key, value in pred_dict.items():
            ground_truth_keypoints[index_map.get(key)] = ground_truth_dict.get(key, 0)
            predicted_keypoints[index_map.get(key)] = value
            if ground_truth_dict.get(key, 0):
                keypoint_visibilities[index_map.get(key)] = 1

        return ground_truth_keypoints, predicted_keypoints, keypoint_visibilities

    for key, value in ground_truth_dict.items():
        ground_truth_keypoints[index_map.get(key)] = value
        predicted_keypoints[index_map.get(key)] = pred_dict.get(key)
        keypoint_visibilities[index_map.get(key)] = 1

    return ground_truth_keypoints, predicted_keypoints, keypoint_visibilities


if __name__ == "__main__":
    test_gt_kps = np.random.random((20, 2)) * 250
    test_pred_kps = np.random.random((20, 2)) * 250
    det_scale = 1.0
    kps_visibilities = np.ones((20, ))
    kps_weights = np.ones((20, ))

    gt_sim = oks(test_gt_kps, test_gt_kps, det_scale, kps_visibilities, kps_weights)
    print(f'similarity for ground truth: {gt_sim}')
    pred_sim = oks(test_gt_kps, test_pred_kps, det_scale, kps_visibilities, kps_weights)
    print(f'similarity for predicted: {pred_sim}')

    print(os.getcwd())
    annotation_data = load_annotations('../CarnivoreID-1/metadata_cleaned.json')

    annotations = annotation_data.get('annotations')

    annotations = filter_annotations(annotations)

    for index, annot in enumerate(annotations):
        show_image_with_keypoints(annot, bb=True)
        if 0 < index:
            break
