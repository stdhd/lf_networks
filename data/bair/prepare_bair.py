import argparse
import cv2
import numpy as np
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import os
import imageio
import h5py
import glob
import pandas as pd
import hashlib
import random
import io


class ACTION_MISMATCH:
    ERROR = 0
    PAD_ZERO = 1
    CLEAVE = 2


class STATE_MISMATCH:
    ERROR = 0
    PAD_ZERO = 1
    CLEAVE = 2

def default_loader_hparams():
    return {
        'target_adim': 4,
        'target_sdim': 5,
        'state_mismatch': STATE_MISMATCH.ERROR,  # TODO make better flag parsing
        'action_mismatch': ACTION_MISMATCH.ERROR,  # TODO make better flag parsing
        'img_size': [48, 64],
        'cams_to_load': [0],
        'impute_autograsp_action': True,
        'load_annotations': False,
        'zero_if_missing_annotation': False,
        'load_T': 0  # TODO implement error checking here for jagged reading
    }


def load_camera_imgs(cam_index, file_pointer, file_metadata, target_dims, start_time=0, n_load=None):
    cam_group = file_pointer['env']['cam{}_video'.format(cam_index)]
    old_dims = file_metadata['frame_dim']
    length = file_metadata['img_T']
    encoding = file_metadata['img_encoding']
    image_format = file_metadata['image_format']

    if n_load is None:
        n_load = length

    old_height, old_width = old_dims
    target_height, target_width = target_dims
    resize_method = cv2.INTER_CUBIC
    if target_height * target_width < old_height * old_width:
        resize_method = cv2.INTER_AREA

    images = np.zeros((n_load, target_height, target_width, 3), dtype=np.uint8)
    if encoding == 'mp4':
        buf = io.BytesIO(cam_group['frames'][:].tostring())
        img_buffer = [img for t, img in enumerate(imageio.get_reader(buf, format='mp4')) if
                      start_time <= t < n_load + start_time]
    elif encoding == 'jpg':
        img_buffer = [cv2.imdecode(cam_group['frame{}'.format(t)][:], cv2.IMREAD_COLOR)[:, :, ::-1]
                      for t in range(start_time, start_time + n_load)]
    else:
        raise ValueError("encoding not supported")

    for t, img in enumerate(img_buffer):
        if (old_height, old_width) == (target_height, target_width):
            images[t] = img
        else:
            images[t] = cv2.resize(img, (target_width, target_height), interpolation=resize_method)

    if image_format == 'RGB':
        return images
    elif image_format == 'BGR':
        return images[:, :, :, ::-1]
    raise NotImplementedError


def load_states(file_pointer, meta_data, hparams):
    s_T, sdim = meta_data['state_T'], meta_data['sdim']
    if hparams.target_sdim == sdim:
        return file_pointer['env']['state'][:]

    elif sdim < hparams.target_sdim and hparams.state_mismatch & STATE_MISMATCH.PAD_ZERO:
        pad = np.zeros((s_T, hparams.target_sdim - sdim), dtype=np.float32)
        return np.concatenate((file_pointer['env']['state'][:], pad), axis=-1)

    elif sdim > hparams.target_sdim and hparams.state_mismatch & STATE_MISMATCH.CLEAVE:
        return file_pointer['env']['state'][:][:, :hparams.target_sdim]

    else:
        raise ValueError("file sdim - {}, target sdim - {}, pad behavior - {}".format(sdim, hparams.target_sdim,
                                                                                      hparams.state_mismatch))


def load_actions(file_pointer, meta_data, hparams):
    a_T, adim = meta_data['action_T'], meta_data['adim']
    if hparams.target_adim == adim:
        return file_pointer['policy']['actions'][:]

    elif hparams.target_adim == adim + 1 and hparams.impute_autograsp_action and meta_data['primitives'] == 'autograsp':
        action_append, old_actions = np.zeros((a_T, 1)), file_pointer['policy']['actions'][:]
        next_state = file_pointer['env']['state'][:][1:, -1]

        high_val, low_val = meta_data['high_bound'][-1], meta_data['low_bound'][-1]
        midpoint = (high_val + low_val) / 2.0

        for t, s in enumerate(next_state):
            if s > midpoint:
                action_append[t, 0] = high_val
            else:
                action_append[t, 0] = low_val
        return np.concatenate((old_actions, action_append), axis=-1)

    elif adim < hparams.target_adim and hparams.action_mismatch & ACTION_MISMATCH.PAD_ZERO:
        pad = np.zeros((a_T, hparams.target_adim - adim), dtype=np.float32)
        return np.concatenate((file_pointer['policy']['actions'][:], pad), axis=-1)

    elif adim > hparams.target_adim and hparams.action_mismatch & ACTION_MISMATCH.CLEAVE:
        return file_pointer['policy']['actions'][:][:, :hparams.target_adim]

    else:
        raise ValueError("file adim - {}, target adim - {}, pad behavior - {}".format(adim, hparams.target_adim,
                                                                                      hparams.action_mismatch))


def load_annotations(file_pointer, metadata, hparams, cams_to_load):
    old_height, old_width = metadata['frame_dim']
    target_height, target_width = hparams.img_size
    scale_height, scale_width = target_height / float(old_height), target_width / float(old_width)
    annot = np.zeros((metadata['img_T'], len(cams_to_load), target_height, target_width, 2), dtype=np.float32)
    if metadata.get('contains_annotation', False) != True and hparams.zero_if_missing_annotation:
        return annot

    assert metadata['contains_annotation'], "no annotations to load!"
    point_mat = file_pointer['env']['bbox_annotations'][:].astype(np.int32)

    for t in range(metadata['img_T']):
        for n, chosen_cam in enumerate(cams_to_load):
            for obj in range(point_mat.shape[2]):
                h1, w1 = point_mat[t, chosen_cam, obj, 0] * [scale_height, scale_width] - 1
                h2, w2 = point_mat[t, chosen_cam, obj, 1] * [scale_height, scale_width] - 1
                h, w = int((h1 + h2) / 2), int((w1 + w2) / 2)
                annot[t, n, h, w, obj] = 1
    return annot


def load_data(f_name, file_metadata, hparams, rng=None):
    rng = random.Random(rng)

    assert os.path.exists(f_name) and os.path.isfile(f_name), "invalid f_name"
    with open(f_name, 'rb') as f:
        buf = f.read()
    assert hashlib.sha256(buf).hexdigest() == file_metadata[
        'sha256'], "file hash doesn't match meta-data. maybe delete pkl and re-generate?"

    with h5py.File(io.BytesIO(buf)) as hf:
        start_time, n_states = 0, min([file_metadata['state_T'], file_metadata['img_T'], file_metadata['action_T'] + 1])
        assert n_states > 1, "must be more than one state in loaded tensor!"
        if 1 < hparams.load_T < n_states:
            start_time = rng.randint(0, n_states - hparams.load_T)
            n_states = hparams.load_T

        assert all([0 <= i < file_metadata['ncam'] for i in hparams.cams_to_load]), "cams_to_load out of bounds!"
        images, selected_cams = [], []
        for cam_index in hparams.cams_to_load:
            images.append(load_camera_imgs(cam_index, hf, file_metadata, hparams.img_size, start_time, n_states)[None])
            selected_cams.append(cam_index)
        images = np.swapaxes(np.concatenate(images, 0), 0, 1)

        actions = load_actions(hf, file_metadata, hparams).astype(np.float32)[start_time:start_time + n_states - 1]
        states = load_states(hf, file_metadata, hparams).astype(np.float32)[start_time:start_time + n_states]

        #if hparams.load_annotations:
        #    annotations = load_annotations(hf, file_metadata, hparams, selected_cams)[start_time:start_time + n_states]
        #    return images, actions, states, annotations

    return images, actions, states


class MetaDataContainer:
    def __init__(self, base_path, meta_data):
        self._meta_data = meta_data
        self._base_path = base_path

    def get_file_metadata(self, fname):
        fname = fname.split('/')[-1]
        return self._meta_data.loc[fname]

    def select_objects(self, obj_class_name):
        if isinstance(obj_class_name, str):
            return self._meta_data[[obj_class_name in x for x in self._meta_data['object_classes']]]
        return self._meta_data[[set(obj_class_name) == set(x) for x in self._meta_data['object_classes']]]

    @property
    def frame(self):
        return self._meta_data

    @property
    def files(self):
        return ['{}/{}'.format(self._base_path, f) for f in self.frame.index]

    def get_shuffled_files(self, rng=None):
        files = ['{}/{}'.format(self._base_path, f) for f in self.frame.index]
        if rng:
            rng.shuffle(files)
        else:
            random.shuffle(files)
        return files

    @property
    def base_path(self):
        return self._base_path

    def __getitem__(self, arg):
        return MetaDataContainer(self._base_path, self._meta_data[arg])

    def __contains__(self, item):
        return item in self._meta_data

    def __repr__(self):
        return repr(self._meta_data)

    def __str__(self):
        return str(self._meta_data)

    def __eq__(self, other):
        return self._meta_data == other

    def __ne__(self, other):
        return self._meta_data != other

    def __lt__(self, other):
        return self._meta_data < other

    def __le__(self, other):
        return self._meta_data <= other

    def __gt__(self, other):
        return self._meta_data > other

    def __ge__(self, other):
        return self._meta_data >= other

    def keys(self):
        return self._meta_data.keys()

    def __len__(self):
        return len(self._meta_data)

def get_metadata_frame(files):
    if isinstance(files, str):
        base_path = files
        files = sorted(glob.glob('{}/*.hdf5'.format(files)))
        if not files:
            raise ValueError('no hdf5 files found!')

        if os.path.exists('{}/meta_data.pkl'.format(base_path)):
            meta_data = pd.read_pickle('{}/meta_data.pkl'.format(base_path), compression='gzip')

            registered_fnames = set([f for f in meta_data.index])
            loaded_fnames = set([f.split('/')[-1] for f in files])

            if loaded_fnames == registered_fnames:
                return meta_data
            os.remove('{}/meta_data.pkl'.format(base_path))
            print('regenerating meta_data file!')
    elif isinstance(files, (list, tuple)):
        base_path = None
        files = sorted(files)
    else:
        raise ValueError("Must be path to files or list/tuple of filenames")

    with Pool(cpu_count()) as p:
        meta_data = list(tqdm(p.imap(load_metadata_dict, files), total=len(files)))

    data_frame = pd.DataFrame(meta_data, index=[f.split('/')[-1] for f in files])
    if base_path:
        data_frame.to_pickle("{}/meta_data.pkl".format(base_path), compression='gzip')
    return data_frame


def load_metadata(files):
    base_path = files
    if isinstance(files, (tuple, list)):
        base_path = ''
    else:
        files = base_path = os.path.expanduser(base_path)

    return MetaDataContainer(base_path, get_metadata_frame(files))


def load_metadata_dict(fname):
    if not os.path.exists(fname) or not os.path.isfile(fname):
        raise IOError("can't find {}".format(fname))
    buf = open(fname, 'rb').read()

    with h5py.File(io.BytesIO(buf)) as hf:
        meta_data_dict = {'file_version': hf['file_version'][()]}

        meta_data_dict['sha256'] = hashlib.sha256(buf).hexdigest()
        meta_data_dict['sdim'] = hf['env']['state'].shape[1]
        meta_data_dict['state_T'] = hf['env']['state'].shape[0]

        meta_data_dict['adim'] = hf['policy']['actions'].shape[1]
        meta_data_dict['action_T'] = hf['policy']['actions'].shape[0]

        # assumes all cameras have same attributes (if they exist)
        n_cams = hf['env'].attrs.get('n_cams', 0)
        if n_cams:
            meta_data_dict['ncam'] = n_cams

            if hf['env'].attrs['cam_encoding'] == 'mp4':
                meta_data_dict['frame_dim'] = hf['env']['cam0_video']['frames'].attrs['shape'][:2]
                meta_data_dict['img_T'] = hf['env']['cam0_video']['frames'].attrs['T']
                meta_data_dict['img_encoding'] = 'mp4'
                meta_data_dict['image_format'] = hf['env']['cam0_video']['frames'].attrs['image_format']
            else:
                meta_data_dict['frame_dim'] = hf['env']['cam0_video']['frame0'].attrs['shape'][:2]
                meta_data_dict['image_format'] = hf['env']['cam0_video']['frame0'].attrs['image_format']
                meta_data_dict['img_encoding'] = 'jpg'
                meta_data_dict['img_T'] = len(hf['env']['cam0_video'])

        # TODO: remove misc field and shift all to meta-data
        for k in hf['misc'].keys():
            assert k not in meta_data_dict, "key {} already present!".format(k)
            meta_data_dict[k] = hf['misc'][k][()]

        for k in hf['metadata'].attrs.keys():
            assert k not in meta_data_dict, "key {} already present!".format(k)
            meta_data_dict[k] = hf['metadata'].attrs[k]

        if 'low_bound' not in meta_data_dict and 'low_bound' in hf['env']:
            meta_data_dict['low_bound'] = hf['env']['low_bound'][0]

        if 'high_bound' not in meta_data_dict and 'high_bound' in hf['env']:
            meta_data_dict['high_bound'] = hf['env']['high_bound'][0]

        return meta_data_dict


""" sample metadata
>>> database = load_metadata('../robonet_hdf5_dir')
>>> database.frame.T['stanford_franka_corr_noise_traj1097.hdf5']
file_version                                                     b'0.1.0'
sha256                  d67a389cc110a6dddf7f03fe10cba9b3d2a4268695290f...
sdim                                                                    5
state_T                                                                31
adim                                                                    4
action_T                                                               30
ncam                                                                    3
frame_dim                                                      [240, 320]
img_T                                                                  31
img_encoding                                                          mp4
image_format                                                          RGB
traj_ok                                                              True
action_space                                                  x,y,z,theta
background                                                            lab
bin_insert                                                           none
bin_type                                                        metal_bin
camera_configuration                                          multiview_3
camera_type                                                 Logitech C920
contains_annotation                                                 False
environment_size              [0.35, 0.4, 0.09999999999999999, 0.6, 3.14]
gripper                                                              hand
object_batch            b'/home/panda1/data_robonet/home/panda1/ros_ws...
object_classes                                                  [b'toys']
policy_desc             random policy, with correlated gaussian noise ...
primitives                                                      autograsp
robot                                                              franka
term_t                                                                 29
low_bound                                 [0.35, -0.2, 0.05, -0.3, -1.57]
high_bound                                    [0.7, 0.2, 0.15, 0.3, 1.57]
Name: stanford_franka_corr_noise_traj1097.hdf5, dtype: object
"""


def load_data(f_name, file_metadata, hparams):
    # modified hdf5_loader.load_data
    with open(f_name, 'rb') as f:
        buf = f.read()
    assert hashlib.sha256(buf).hexdigest() == file_metadata['sha256'], \
        "file hash doesn't match meta-data. maybe delete pkl and re-generate?"

    with h5py.File(io.BytesIO(buf)) as hf:
        start_time, n_states = 0, min([file_metadata['state_T'],
                                       file_metadata['img_T'],
                                       file_metadata['action_T'] + 1])
        assert n_states > 1, "must be more than one state in loaded tensor!"
        images, selected_cams = [], []
        for cam_index in range(file_metadata['ncam']):
            images.append(load_camera_imgs(
                cam_index, hf, file_metadata,
                hparams['img_size'], start_time, n_states)[None])
            selected_cams.append(cam_index)
        images = np.swapaxes(np.concatenate(images, 0), 0, 1)
        actions = None #load_actions(hf, file_metadata, hparams).astype(
            #np.float32)[start_time:start_time + n_states-1]
        states = None #load_states(hf, file_metadata, hparams).astype(
            #np.float32)[start_time:start_time + n_states]

        #if hparams.load_annotations:
            #annotations = load_annotations(
            #    hf, file_metadata,
            #    hparams, selected_cams)[start_time:start_time + n_states]
          #  return images, actions, states, None
    return images, actions, states


def iterate_dataset(hdf5_dir):
    hparams = default_loader_hparams()
    hparams['img_size'] = [-1, -1]
    hparams['action_mismatch'] = 3
    hparams['state_mismatch'] = 3
    #hparams = HParams(**hparams)

    meta_dataset = load_metadata(hdf5_dir)  # dir path is fed
    for fname in tqdm(meta_dataset.frame.index):
        # fname: stanford_franka_traj67.hdf5
        meta_data = meta_dataset.get_file_metadata(fname)
        frame_dim = meta_data['frame_dim']
        assert 0 < frame_dim[0] and 0 < frame_dim[1]
        hparams['img_size'] = frame_dim  # set size to every video
        fpath = os.path.join(meta_dataset.base_path, fname)
        data = load_data(fpath, meta_data, hparams)
        # images, actions, states = data
        # [(31, 1, 480, 640, 3), (30, 4), (31, 5)]
        yield (fname, fpath, meta_data) + tuple(data)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hdf5-dir', required=True)
    parser.add_argument('--out-dir', required=True)
    args = parser.parse_args()

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    MP4_FPS = 5.0

    for data in iterate_dataset(hdf5_dir=args.hdf5_dir):
        fname, fpath, meta_data, images, actions, states = data
        # shapes: [(31, 1, 240, 320, 3), (30, 4), (31, 5)]
        # image shape (T, N_VIEWS, H, W, C): (31, 1, 240, 320, 3)
        # action shape (T - 1, ACTION_DIM): (30, 4)
        # robot state shape (T, STATE_DIM): (31, 5)
        T, n_views, H, W, C = images.shape
        images = images[:, :, :, :, ::-1]  # toRGB
        for i_view in range(n_views):
            a_video = images[:, i_view]
            out_video_path = os.path.join(
                args.out_dir, '{}.cam{:02d}.mp4'.format(fname, i_view))
            writer = cv2.VideoWriter(out_video_path, fourcc, MP4_FPS, (W, H))
            for a_frame in a_video:
                writer.write(a_frame)
            writer.release()


if __name__ == '__main__':
    main()