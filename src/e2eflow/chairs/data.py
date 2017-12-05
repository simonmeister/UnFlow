import os
import sys
import re

import numpy as np
from PIL import Image

from ..core.data import Data
from ..util import tryremove
from shutil import copyfile, rmtree
from urllib.request import urlretrieve


class ChairsData(Data):
    URL = 'http://lmb.informatik.uni-freiburg.de/data/FlyingChairs/FlyingChairs.zip'
    TRAIN_VAL_URL = 'http://lmb.informatik.uni-freiburg.de/resources/datasets/FlyingChairs/FlyingChairs_train_val.txt'
    dirs = ['flying_chairs']

    def __init__(self, data_dir, stat_log_dir=None,
                 development=True, fast_dir=None):
        super().__init__(data_dir, stat_log_dir,
                         development=development,
                         fast_dir=fast_dir)

    def _fetch_if_missing(self):
        local_path = os.path.join(self.data_dir, 'flying_chairs')
        train_val_path = os.path.join(local_path, 'FlyingChairs_train_val.txt')
        if not os.path.isdir(local_path):
            did_download = True
            self._download_and_extract(self.URL, local_path)
            urlretrieve(self.TRAIN_VAL_URL, train_val_path)
        else:
            did_download = False

        data_path = os.path.join(local_path, 'FlyingChairs_release', 'data')
        os.makedirs(os.path.join(local_path, 'image'), exist_ok=True)
        os.makedirs(os.path.join(local_path, 'flow'), exist_ok=True)
        os.makedirs(os.path.join(local_path, 'test_image'), exist_ok=True)

        if os.path.isdir(data_path):
            print('>> converting chairs data to .png')
            train_val_repeated = []
            train_val = []
            with open(train_val_path) as f:
                for line in f:
                    training = int(line.strip()) == 1
                    train_val_repeated.extend([training, training])
                    train_val.extend([training])
            # Convert .ppm to .png and split data into image and flow directory
            im_files = [f for f in os.listdir(data_path) if
                        re.match(r'[0-9]+.*\.ppm', f)]
            im_files.sort()
            flow_files = [f for f in os.listdir(data_path) if
                          re.match(r'[0-9]+.*\.flo', f)]
            flow_files.sort()
            for t, f in zip(train_val_repeated, im_files):
                name, ext = os.path.splitext(f)
                path = os.path.join(data_path, f)

                im = Image.open(path)
                folder = 'image' if t else 'test_image'
                im.save(os.path.join(local_path, folder, name + '.png'),
                        'PNG')
            for t, f in zip(train_val, flow_files):
                path = os.path.join(data_path, f)
                if not t:
                    copyfile(path, os.path.join(local_path, 'flow', f))
            if did_download:
                rmtree(data_path)
            print('>> processed chairs data')

    def get_raw_dirs(self):
       return [os.path.join(self.current_dir, 'flying_chairs', 'image')]
