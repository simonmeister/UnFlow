import os
import sys

import numpy as np
import matplotlib.image as mpimg

from ..core.data import Data
from ..util import tryremove


class SintelData(Data):
    SINTEL_URL = 'http://files.is.tue.mpg.de/sintel/MPI-Sintel-complete.zip'

    dirs = ['sintel']

    def __init__(self, data_dir, stat_log_dir=None,
                 development=True, fast_dir=None):
        super().__init__(data_dir, stat_log_dir,
                         development=development,
                         fast_dir=fast_dir)

    def _fetch_if_missing(self):
        local_path = os.path.join(self.data_dir, 'sintel')
        if not os.path.isdir(local_path):
            self._download_and_extract(self.SINTEL_URL, local_path)

    def get_raw_dirs(self):
        dirs = []
        for folder in ['training/clean', 'training/final', 'test/clean', 'test/final']:
            top_dir = os.path.join(self.current_dir, 'sintel/' + folder)
            for sub_dir in os.listdir(top_dir):
              dirs.append(os.path.join(top_dir, sub_dir))
        return dirs
