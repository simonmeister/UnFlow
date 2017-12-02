import os
import sys

import numpy as np
import matplotlib.image as mpimg

from ..core.data import Data
from ..util import tryremove


class MiddleburyData(Data):
    MDB_FLOW_URL = 'http://vision.middlebury.edu/flow/data/comp/zip/other-gt-flow.zip'
    MDB_COLOR_URL = 'http://vision.middlebury.edu/flow/data/comp/zip/other-color-twoframes.zip'
    MDB_EVAL_URL = 'http://vision.middlebury.edu/flow/data/comp/zip/eval-color-twoframes.zip'

    dirs = ['middlebury']

    def __init__(self, data_dir, stat_log_dir=None,
                 development=True, fast_dir=None):
        super().__init__(data_dir, stat_log_dir,
                         development=development,
                         fast_dir=fast_dir)

    def _fetch_if_missing(self):
        local_path = os.path.join(self.data_dir, 'middlebury')
        if not os.path.isdir(local_path):
            self._download_and_extract(self.MDB_FLOW_URL, local_path)
            self._download_and_extract(self.MDB_COLOR_URL, local_path)
            self._download_and_extract(self.MDB_EVAL_URL, local_path)

        for name in ['Beanbags', 'DogDance', 'MiniCooper', 'Walking']:
          tryremove(os.path.join(local_path, 'other-data', name))

    def get_raw_dirs(self):
        raise NotImplementedError("Can not train on middlebury")
