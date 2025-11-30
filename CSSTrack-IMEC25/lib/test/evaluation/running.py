import numpy as np
import multiprocessing
import os
import sys
from itertools import product
from collections import OrderedDict
from lib.test.evaluation import Sequence, Tracker
import torch


def _save_tracker_output(seq: Sequence, tracker: Tracker, output: dict):
    """Saves the output of the tracker."""

    if not os.path.exists(tracker.results_dir):
        print("create tracking result dir:", tracker.results_dir)
        os.makedirs(tracker.results_dir)

    '''2021.1.5 create new folder for these two datasets'''
    if seq.dataset in ['trackingnet', 'got10k']:
        base_results_path = os.path.join(tracker.results_dir, seq.name)
        if not os.path.exists(os.path.join(base_results_path)):
            os.makedirs(os.path.join(base_results_path))
        base_results_path = os.path.join(tracker.results_dir, seq.name, seq.name)
    else:
        base_results_path = os.path.join(tracker.results_dir, seq.name)

    def save_bb(file, data):
        tracked_bb = np.array(data).astype(int)
        np.savetxt(file, tracked_bb, delimiter=',', fmt='%d')

    def save_time(file, data):
        exec_times = np.array(data).astype(float)
        np.savetxt(file, exec_times, delimiter='\t', fmt='%f')

    def save_score(file, data):
        scores = np.array(data).astype(float)
        np.savetxt(file, scores, delimiter='\t', fmt='%.2f')

    def _convert_dict(input_dict):
        data_dict = {}
        for elem in input_dict:
            for k, v in elem.items():
                if k in data_dict.keys():
                    data_dict[k].append(v)
                else:
                    data_dict[k] = [v, ]
        return data_dict

    for key, data in output.items():
        # If data is empty
        if not data:
            continue

        if key == 'target_bbox':
            if isinstance(data[0], (dict, OrderedDict)):
                data_dict = _convert_dict(data)

                for obj_id, d in data_dict.items():
                    bbox_file = '{}_{}.txt'.format(base_results_path, obj_id)
                    save_bb(bbox_file, d)
            else:
                # Single-object mode
                if 'got10k' in seq.dataset:
                    bbox_file = '{}_001.txt'.format(base_results_path)
                else:
                    bbox_file = '{}.txt'.format(base_results_path)
                save_bb(bbox_file, data)

        if key == 'all_boxes':
            if isinstance(data[0], (dict, OrderedDict)):
                data_dict = _convert_dict(data)

                for obj_id, d in data_dict.items():
                    bbox_file = '{}_{}_all_boxes.txt'.format(base_results_path, obj_id)
                    save_bb(bbox_file, d)
            else:
                # Single-object mode
                bbox_file = '{}_all_boxes.txt'.format(base_results_path)
                save_bb(bbox_file, data)

        if key == 'all_scores':
            if isinstance(data[0], (dict, OrderedDict)):
                data_dict = _convert_dict(data)

                for obj_id, d in data_dict.items():
                    bbox_file = '{}_{}_all_scores.txt'.format(base_results_path, obj_id)
                    save_score(bbox_file, d)
            else:
                # Single-object mode
                print("saving scores...")
                bbox_file = '{}_all_scores.txt'.format(base_results_path)
                save_score(bbox_file, data)

def run_sequence(seq: Sequence, tracker: Tracker, debug=False, num_gpu=8, checkpoint_path=''):
    """Runs a tracker on a sequence."""
    '''2021.1.2 Add multiple gpu support'''

    def _results_exist():
        if seq.object_ids is None:
            if seq.dataset in ['trackingnet', 'got10k']:
                base_results_path = os.path.join(tracker.results_dir, seq.name, seq.name)
                bbox_file = '{}_001.txt'.format(base_results_path)
            else:
                bbox_file = '{}/{}.txt'.format(tracker.results_dir, seq.name)
            return os.path.isfile(bbox_file)
        else:
            bbox_files = ['{}/{}_{}.txt'.format(tracker.results_dir, seq.name, obj_id) for obj_id in seq.object_ids]
            missing = [not os.path.isfile(f) for f in bbox_files]
            return sum(missing) == 0


    print('Tracker: {} {} {} ,  Sequence: {}'.format(tracker.name, tracker.parameter_name, tracker.run_id, seq.name))

    output, gtArr = tracker.run_sequence(seq, debug=debug, checkpoint_path=checkpoint_path)

    sys.stdout.flush()
 
    _save_tracker_output(seq, tracker, output)
    return output, gtArr


def run_dataset(dataset, trackers, debug=False, threads=0, num_gpus=8, checkpoint_path=''):
    """Runs a list of trackers on a dataset.
    args:
        dataset: List of Sequence instances, forming a dataset.
        trackers: List of Tracker instances.
        debug: Debug level.
        threads: Number of threads to use (default 0).
    """
    multiprocessing.set_start_method('spawn', force=True)

    print('Evaluating {:4d} trackers on {:5d} sequences'.format(len(trackers), len(dataset)))

    multiprocessing.set_start_method('spawn', force=True)

    if threads == 0:
        mode = 'sequential'
    else:
        mode = 'parallel'

    mode = 'sequential'

    detArr = []
    gtArr = []
    video_nameLis = []
    if True:
        for seq in dataset:
            print ('len(trackers) = ', len(trackers))
            for tracker_info in trackers:
                out, gt = run_sequence(seq, tracker_info, debug=debug, checkpoint_path=checkpoint_path)
    elif mode == 'parallel':
        param_list = [(seq, tracker_info, debug, num_gpus) for seq, tracker_info in product(dataset, trackers)]
        with multiprocessing.Pool(processes=threads) as pool:
            pool.starmap(run_sequence, param_list)
    print('Done')
