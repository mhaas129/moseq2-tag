from moseq2_viz.util import parse_index
from moseq2_viz.model.util import(relabel_by_usage, get_syllable_slices, parse_model_results)
from functools import partial
from sys import platform
import os
import multiprocessing as mp
import numpy as np
import joblib
import tqdm
import warnings


def get_syl_slices(model,index):
    index_file = os.path.join(index)
    model_path = os.path.join(model)

    max_syllable=100
    count='usage'


    if platform in ['linux', 'linux2']:
        print('Setting CPU affinity to use all CPUs...')
        cpu_count = psutil.cpu_count()
        proc = psutil.Process()
        proc.cpu_affinity(list(range(cpu_count)))

    # need to handle h5 intelligently here...

    if model_path.endswith('.p') or model_path.endswith('.pz'):
        model_fit = parse_model_results(joblib.load(model_path))
        labels = model_fit['labels']

        if 'train_list' in model_fit:
            label_uuids = model_fit['train_list']
        else:
            label_uuids = model_fit['keys']
    elif model_fit.endswith('.h5'):
        # load in h5, use index found using another function
        pass

    info_parameters = ['model_class', 'kappa', 'gamma', 'alpha']
    info_dict = {k: model_fit['model_parameters'][k] for k in info_parameters}

    # convert numpy dtypes to their corresponding primitives
    for k, v in info_dict.items():
        if isinstance(v, (np.ndarray, np.generic)):
            info_dict[k] = info_dict[k].item()

    info_dict['model_path'] = model_path
    info_dict['index_path'] = index_file

    labels, _ = relabel_by_usage(labels, count=count)

    index, sorted_index = parse_index(index_file)

    # uuid in both the labels and the index
    uuid_set = set(label_uuids) & set(sorted_index['files'].keys())

    # make sure the files exist
    uuid_set = [uuid for uuid in uuid_set if os.path.exists(sorted_index['files'][uuid]['path'][0])]

    # harmonize everything...
    labels = [label_arr for label_arr, uuid in zip(labels, label_uuids) if uuid in uuid_set]
    label_uuids = [uuid for uuid in label_uuids if uuid in uuid_set]
    sorted_index['files'] = {k: v for k, v in sorted_index['files'].items() if k in uuid_set}

    with mp.Pool() as pool:
        slice_fun = partial(get_syllable_slices,
            labels=labels,
            label_uuids=label_uuids,
            index=sorted_index)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", tqdm.TqdmSynchronisationWarning)
            slices = list(tqdm.tqdm(pool.imap(slice_fun, range(max_syllable)), total=max_syllable))
    return slices