import time
import numpy as np
from typing import Tuple, Dict
from pyquaternion import Quaternion

from nuscenes import NuScenes

from nuscenes.eval.common.data_classes import EvalBoxes
from nuscenes.eval.tracking.data_classes import TrackingBox
from nuscenes.eval.detection.data_classes import DetectionBox
from nuscenes.eval.detection.algo import accumulate, calc_ap, calc_tp
from nuscenes.eval.detection.constants import TP_METRICS
from nuscenes.eval.detection.data_classes import DetectionConfig, DetectionMetricDataList, DetectionMetrics

from nuscenes.utils.data_classes import Box
from nuscenes.utils.geometry_utils import points_in_box


eval_detection_configs = {
    'cvpr_2019': {
        'class_range': {
            'car': 50,
            'truck': 50,
            'bus': 50,
            'trailer': 50,
            'construction_vehicle': 50,
            'pedestrian': 40,
            'motorcycle': 40,
            'bicycle': 40,
            'traffic_cone': 30,
            'barrier': 30
          },
        'dist_fcn': 'center_distance',
        'dist_ths': [0.5, 1.0, 2.0, 4.0],
        'dist_th_tp': 2.0,
        'min_recall': 0.1,
        'min_precision': 0.1,
        'max_boxes_per_sample': 500,
        'mean_ap_weight': 5
    },

    'cvpr_2019_longrange': {
        'class_range': {
            'car': 50,
            'truck': 50,
            'bus': 50,
            'trailer': 50,
            'construction_vehicle': 50,
            'pedestrian': 40,
            'motorcycle': 40,
            'bicycle': 40,
            'traffic_cone': 30,
            'barrier': 30
          },
        'dist_fcn': 'center_distance',
        'dist_ths': [0.5, 1.0, 2.0, 4.0],
        'dist_th_tp': 2.0,
        'min_recall': 0.1,
        'min_precision': 0.1,
        'max_boxes_per_sample': 500,
        'mean_ap_weight': 5
    }
}

def evaluate(gt_boxes,pred_boxes, cfg) -> Tuple[DetectionMetrics, DetectionMetricDataList]:
        """
        Performs the actual evaluation.
        :return: A tuple of high-level and the raw metric data.
        """
        
        start_time = time.time()

        # -----------------------------------
        # Step 1: Accumulate metric data for all classes and distance thresholds.
        # -----------------------------------
        datasaver = {}
        metric_data_list = DetectionMetricDataList()
        for class_name in cfg.class_names:
            # print('Class: ', class_name)
            datasaver[class_name] = {}
            for dist_th in cfg.dist_ths:
                # print('Distance: ', dist_th)
                md, data = accumulate(gt_boxes, pred_boxes, class_name, cfg.dist_fcn_callable, dist_th)
                metric_data_list.set(class_name, dist_th, md)
                datasaver[class_name][dist_th] = data
                

        # -----------------------------------
        # Step 2: Calculate metrics from the data.
        # -----------------------------------

        metrics = DetectionMetrics(cfg)
        for class_name in cfg.class_names:
            for dist_th in cfg.dist_ths:
                metric_data = metric_data_list[(class_name, dist_th)]
                ap = calc_ap(metric_data, cfg.min_recall, cfg.min_precision)
                metrics.add_label_ap(class_name, dist_th, ap)

            for metric_name in TP_METRICS:
                metric_data = metric_data_list[(class_name, cfg.dist_th_tp)]
                if class_name in ['traffic_cone'] and metric_name in ['attr_err', 'vel_err', 'orient_err']:
                    tp = np.nan
                elif class_name in ['barrier'] and metric_name in ['attr_err', 'vel_err']:
                    tp = np.nan
                else:
                    tp = calc_tp(metric_data, cfg.min_recall, metric_name)
                metrics.add_label_tp(class_name, metric_name, tp)

        metrics.add_runtime(time.time() - start_time)

        return metrics, metric_data_list, datasaver


def config_factory(configuration_name: str) -> DetectionConfig:
    """
    Creates a DetectionConfig instance that can be used to initialize a NuScenesEval instance.
    :param configuration_name: Name of desired configuration in eval_detection_configs.
    :return: DetectionConfig instance.
    """

    assert configuration_name in eval_detection_configs.keys(), \
        'Requested unknown configuration {}'.format(configuration_name)

    return DetectionConfig.deserialize(eval_detection_configs[configuration_name])


def filter_eval_boxes(nusc: NuScenes,
                      eval_boxes: EvalBoxes,
                      max_dist: Dict[str, float],
                      rang: str,
                      verbose: bool = False) -> EvalBoxes:
    """
    Applies filtering to boxes. Distance, bike-racks and points per box.
    :param nusc: An instance of the NuScenes class.
    :param eval_boxes: An instance of the EvalBoxes class.
    :param max_dist: Maps the detection name to the eval distance threshold for that class.
    :param verbose: Whether to print to stdout.
    """
    # Retrieve box type.
    assert len(eval_boxes.boxes) > 0
    first_key = list(eval_boxes.boxes.keys())[0]
    box = eval_boxes.boxes[first_key][0]
    if isinstance(box, DetectionBox):
        class_field = 'detection_name'
    elif isinstance(box, TrackingBox):
        class_field = 'tracking_name'
    else:
        raise Exception('Error: Invalid box type: %s' % box)

    # Accumulators for number of filtered boxes.
    total, dist_filter, point_filter, bike_rack_filter = 0, 0, 0, 0
    for ind, sample_token in enumerate(eval_boxes.sample_tokens):

        # Filter on distance first.
        total += len(eval_boxes[sample_token])
        if rang == 'long':
            eval_boxes.boxes[sample_token] = [box for box in eval_boxes[sample_token] if
                                            box.ego_dist < max_dist[box.__getattribute__(class_field)] and box.ego_dist > 2*max_dist[box.__getattribute__(class_field)]/3] # ändra tillbaka
        elif rang == 'medium':
            eval_boxes.boxes[sample_token] = [box for box in eval_boxes[sample_token] if
                                            box.ego_dist < 2*max_dist[box.__getattribute__(class_field)]/3 and box.ego_dist > max_dist[box.__getattribute__(class_field)]/3]
        elif rang == 'short':
            eval_boxes.boxes[sample_token] = [box for box in eval_boxes[sample_token] if
                                            box.ego_dist < max_dist[box.__getattribute__(class_field)]/3]
        else:
            eval_boxes.boxes[sample_token] = [box for box in eval_boxes[sample_token] if
                                            box.ego_dist < max_dist[box.__getattribute__(class_field)]]
        dist_filter += len(eval_boxes[sample_token])

        # Then remove boxes with zero points in them. Eval boxes have -1 points by default.
        eval_boxes.boxes[sample_token] = [box for box in eval_boxes[sample_token] if not box.num_pts == 0]
        point_filter += len(eval_boxes[sample_token])

        # Perform bike-rack filtering.
        sample_anns = nusc.get('sample', sample_token)['anns']
        bikerack_recs = [nusc.get('sample_annotation', ann) for ann in sample_anns if
                         nusc.get('sample_annotation', ann)['category_name'] == 'static_object.bicycle_rack']
        bikerack_boxes = [Box(rec['translation'], rec['size'], Quaternion(rec['rotation'])) for rec in bikerack_recs]
        filtered_boxes = []
        for box in eval_boxes[sample_token]:
            if box.__getattribute__(class_field) in ['bicycle', 'motorcycle']:
                in_a_bikerack = False
                for bikerack_box in bikerack_boxes:
                    if np.sum(points_in_box(bikerack_box, np.expand_dims(np.array(box.translation), axis=1))) > 0:
                        in_a_bikerack = True
                if not in_a_bikerack:
                    filtered_boxes.append(box)
            else:
                filtered_boxes.append(box)

        eval_boxes.boxes[sample_token] = filtered_boxes
        bike_rack_filter += len(eval_boxes.boxes[sample_token])

    if verbose:
        print("=> Original number of boxes: %d" % total)
        print("=> After distance based filtering: %d" % dist_filter)
        print("=> After LIDAR points based filtering: %d" % point_filter)
        print("=> After bike rack filtering: %d" % bike_rack_filter)

    return eval_boxes