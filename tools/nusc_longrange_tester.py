
import json, time, os
from nuscenes.nuscenes import NuScenes
from modfunctions import evaluate,config_factory, filter_eval_boxes
from nuscenes.eval.common.loaders import load_prediction, load_gt,add_center_dist
from nuscenes.eval.detection.data_classes import DetectionBox

# modelversion = 'v2_no_xy_flip'

# modelversion = 'no_gt_sampler'                 
# modelversion = 'baseline_no_fade'
# modelversion = 'baseline_no_fade_2x'                
# modelversion = 'baseline_with_fade'
modelversion = 'baseline_with_fade_2x'            
# modelversion = 'v3_fps_ds'
# modelversion = 'v3_fps_ds_fade'                      
# modelversion = 'v3_random_ds'                
# modelversion = 'v3_random_ds_fade'
# modelversion = 'v4_random_ds'
# modelversion = 'v4_random_ds_fade'           
# modelversion = 'v4_longrange_bias'           
# modelversion = 'v4_longrange_bias_with_fade'
# modelversion = 'v5_no_fade'                  
# modelversion = 'v5_with_fade'                
# modelversion = 'v6_no_fade'                  
# modelversion = 'v6_with_fade'

pp = ''
# pp = 'pp/'

output_dir = 'results/filtered/' + pp + modelversion

if not os.path.isdir(output_dir):
    os.mkdir(output_dir)

# res_path = 'results/HPC/v1/pts_bbox/results_nusc.json'
# res_path = 'results/HPC/v2_no_xy_flip/pts_bbox/results_nusc.json'
# res_path = 'results/HPC/v3_random_ds/pts_bbox/results_nusc.json'
res_path = 'results/HPC/' + pp + modelversion + '/pts_bbox/results_nusc.json'
ranges = ['short', 'medium', 'long', 'all']
# longrange = False
# mediumrange = False
# shortrange = True

verbose = False

# if longrange:
cfg = config_factory("cvpr_2019_longrange")
# else:

# cfg = config_factory("cvpr_2019")
nusc = NuScenes("v1.0-trainval","data/nuscenes",verbose=verbose)



final_data = {}
start = time.time()
for rang in ranges:

    print('Loading boxes:')
    
    pred_boxes, meta = load_prediction(res_path,cfg.max_boxes_per_sample,DetectionBox, verbose=verbose)
    gt_boxes = load_gt(nusc,'val',DetectionBox,True)

    print('Adding center distances: ')
    pred_boxes = add_center_dist(nusc, pred_boxes)
    gt_boxes = add_center_dist(nusc, gt_boxes)
    
    print('Filtering boxes: ')
    print(rang)
    filtered_pred_boxes = filter_eval_boxes(nusc, pred_boxes, cfg.class_range, rang=rang, verbose=verbose)
    filtered_gt_boxes = filter_eval_boxes(nusc, gt_boxes, cfg.class_range, rang=rang, verbose=verbose)

    pred_data = {
                'car': 0,
                'truck': 0,
                'bus': 0,
                'trailer': 0,
                'construction_vehicle': 0,
                'pedestrian': 0,
                'motorcycle': 0,
                'bicycle': 0,
                'traffic_cone': 0,
                'barrier': 0
    }
    gt_data = {
                'car': 0,
                'truck': 0,
                'bus': 0,
                'trailer': 0,
                'construction_vehicle': 0,
                'pedestrian': 0,
                'motorcycle': 0,
                'bicycle': 0,
                'traffic_cone': 0,
                'barrier': 0
    }
    pred_boxes_list = [box for box in filtered_pred_boxes.all]
    gt_boxes_list = [box for box in filtered_gt_boxes.all]

    for ind, box in enumerate(pred_boxes_list):
        # print(box)
        pred_data[box.detection_name] += 1

    for ind, box in enumerate(gt_boxes_list):
        gt_data[box.detection_name] += 1

    # print(pred_boxes_list[100])
    print(pred_data)
    print(gt_data)


    print('Evaluating: ')
    print(rang)
    metrics, metric_data_list, data = evaluate(gt_boxes,pred_boxes,cfg)
    metrics_summary = metrics.serialize()
    final_data[rang] = data
    final_data[rang]['num_gt_objects'] = gt_data
    final_data[rang]['metrics_summary'] = metrics_summary
    print('mAP:  %.4f' % (metrics_summary['mean_ap']))
    err_name_mapping = {
        'trans_err': 'mATE',
        'scale_err': 'mASE',
        'orient_err': 'mAOE',
        'vel_err': 'mAVE',
        'attr_err': 'mAAE'
    }
    for tp_name, tp_val in metrics_summary['tp_errors'].items():
        print('%s: %.4f' % (err_name_mapping[tp_name], tp_val))
    print('NDS:  %.4f' % (metrics_summary['nd_score']))
    print('Eval time: %.1fs' % metrics_summary['eval_time'])

    # Print per-class metrics.
    print()
    print('Per-class results:')
    print('Object Class\t\tAP\tATE\tASE\tAOE\tAVE\tAAE')
    class_aps = metrics_summary['mean_dist_aps']
    class_tps = metrics_summary['label_tp_errors']
    for class_name in class_aps.keys():
        if class_name == 'construction_vehicle':
            print('%s\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f'
                % (class_name, class_aps[class_name],
                    class_tps[class_name]['trans_err'],
                    class_tps[class_name]['scale_err'],
                    class_tps[class_name]['orient_err'],
                    class_tps[class_name]['vel_err'],
                    class_tps[class_name]['attr_err']))
        elif class_name == 'pedestrian' or class_name == 'motorcycle' or class_name == 'traffic_cone':
            print('%s\t\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f'
                % (class_name, class_aps[class_name],
                    class_tps[class_name]['trans_err'],
                    class_tps[class_name]['scale_err'],
                    class_tps[class_name]['orient_err'],
                    class_tps[class_name]['vel_err'],
                    class_tps[class_name]['attr_err']))
        else:
            print('%s\t\t\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f'
                    % (class_name, class_aps[class_name],
                        class_tps[class_name]['trans_err'],
                        class_tps[class_name]['scale_err'],
                        class_tps[class_name]['orient_err'],
                        class_tps[class_name]['vel_err'],
                        class_tps[class_name]['attr_err']))


# Dump the metric data, meta and metrics to disk.

print('Saving metrics to: %s' % output_dir)
metrics_summary = metrics.serialize()
# metrics_summary['meta'] = self.meta.copy()
with open(os.path.join(output_dir, modelversion + '_metrics_summary_filtered.json'), 'w') as f:
    json.dump(metrics_summary, f, indent=2)
with open(os.path.join(output_dir, modelversion + '_metrics_details_filtered.json'), 'w') as f:
    json.dump(metric_data_list.serialize(), f, indent=2)
with open(os.path.join(output_dir, modelversion + '_statistics.json'), 'w') as f:
    json.dump(final_data, f, indent=2)

# Print high-level metrics.
print('mAP:  %.4f' % (metrics_summary['mean_ap']))
err_name_mapping = {
    'trans_err': 'mATE',
    'scale_err': 'mASE',
    'orient_err': 'mAOE',
    'vel_err': 'mAVE',
    'attr_err': 'mAAE'
}
for tp_name, tp_val in metrics_summary['tp_errors'].items():
    print('%s: %.4f' % (err_name_mapping[tp_name], tp_val))
print('NDS:  %.4f' % (metrics_summary['nd_score']))
print('Eval time: %.1fs' % metrics_summary['eval_time'])

# Print per-class metrics.
print()
print('Per-class results:')
print('Object Class\t\tAP\tATE\tASE\tAOE\tAVE\tAAE')
class_aps = metrics_summary['mean_dist_aps']
class_tps = metrics_summary['label_tp_errors']
for class_name in class_aps.keys():
    if class_name == 'construction_vehicle':
        print('%s\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f'
            % (class_name, class_aps[class_name],
                class_tps[class_name]['trans_err'],
                class_tps[class_name]['scale_err'],
                class_tps[class_name]['orient_err'],
                class_tps[class_name]['vel_err'],
                class_tps[class_name]['attr_err']))
    elif class_name == 'pedestrian' or class_name == 'motorcycle' or class_name == 'traffic_cone':
        print('%s\t\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f'
            % (class_name, class_aps[class_name],
                class_tps[class_name]['trans_err'],
                class_tps[class_name]['scale_err'],
                class_tps[class_name]['orient_err'],
                class_tps[class_name]['vel_err'],
                class_tps[class_name]['attr_err']))
    else:
        print('%s\t\t\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f'
                % (class_name, class_aps[class_name],
                    class_tps[class_name]['trans_err'],
                    class_tps[class_name]['scale_err'],
                    class_tps[class_name]['orient_err'],
                    class_tps[class_name]['vel_err'],
                    class_tps[class_name]['attr_err']))
end = time.time()
print('Everything finished in ',end-start, ' seconds')
