import json

import numpy as np

if __name__ == "__main__":
    index_map = {'left-eye': 0, 'right-eye': 1, 'left-earbase': 2, 'right-earbase': 3, 'nose': 4, 'throat': 5,
                 'tailbase': 6, 'withers': 7, 'L-F-elbow': 8, 'R-F-elbow': 9, 'L-B-knee': 10, 'R-B-knee': 11,
                 'L-F-wrist': 12, 'R-F-wrist': 13, 'L-B-ankle': 14, 'R-B-ankle': 15, 'L-F-paw': 16, 'R-F-paw': 17,
                 'L-B-paw': 18, 'R-B-paw': 19
                 }

    with open('../CarnivoreID/metadata_cleaned.json') as json_file:
        data = json.load(json_file)
    with open('../keypoints.json') as json_file:
        animal_data = json.load(json_file)

    data['categories'] = [{'supercategory': 'animal',
                   'id': 1,
                   'name': 'lynx',
                   'keypoints': ['left_eye', 'right_eye', 'left_ear', 'right_ear', 'nose', 'throat',
                                 'tailbase', 'withers', 'left_front_elbow',
                                 'right_front_elbow', 'left_back_elbow', 'right_back_elbow', 'left_front_knee',
                                 'right_front_knee', 'left_back_knee', 'right_back_knee', 'left_front_paw',
                                 'right_front_paw', 'left_back_paw', 'right_back_paw'],
                   'skeleton': [[0, 1], [0, 2], [1, 2], [0, 3], [1, 4], [2, 17], [18, 19], [5, 9], [6, 10], [7, 11], [8, 12], [9, 13], [10, 14], [11, 15], [12, 16]]}]

    possible_indices = [i for i in range(len(data.get('images')))]
    for index, annotation in enumerate(data.get('annotations')):
        annotation['orig_id'] = annotation.get('image_id')
        annotation['image_id'] = index
        annotation['id'] = index
        id = annotation.get('orig_id')
        annotation['category_id'] = 1
        for real_index, current_index in enumerate(possible_indices):
            if data.get('images')[current_index].get('id') == id:
                data.get('images')[current_index]['orig_id'] = id
                data.get('images')[current_index]['image_id'] = index
                data.get('images')[current_index]['id'] = index
                possible_indices.pop(real_index)
                break
        keypoints = np.zeros((20, 3))
        for keypoint_name, keypoint in annotation['keypoints'].items():
            index_in_array = [index_map[keypoint_name]]
            keypoints[index_in_array, 0:2] = keypoint
            keypoints[index_in_array, 2] = 1
        annotation['old_kps'] = annotation['keypoints']
        annotation['keypoints'] = keypoints.tolist()

    indexes_to_drop = []
    for index, annotation in enumerate(data.get('images')):
        if type(annotation.get('id')) != int:
            indexes_to_drop.append(index)
    data['images'] = [i for j, i in enumerate(data.get('images')) if j not in indexes_to_drop]

    json_object = json.dumps(data, indent=4)

    with open('../CarnivoreID/metadata_converted.json', 'w') as json_file:
        json_file.write(json_object)

    a = 5
