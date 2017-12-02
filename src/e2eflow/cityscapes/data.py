import os
import sys

import numpy as np
import matplotlib.image as mpimg

from ..core.data import Data
from ..util import tryremove


class CityscapesData(Data):
    dirs = ['cs']

    def __init__(self, data_dir, stat_log_dir=None,
                 development=True, fast_dir=None):
        super().__init__(data_dir, stat_log_dir,
                         development=development,
                         fast_dir=fast_dir)

    def _fetch_if_missing(self):
        pass

    def get_raw_dirs(self):
       top_dir = os.path.join(self.current_dir, 'cs', 'leftImg8bit_sequence_trainvaltest')
       if not os.path.isdir(top_dir):
         raise RuntimeError(
             "Cityscapes data missing.\n"
             "Download 'leftImg8bit_sequence_trainvaltest.zip (324GB)' "
             "from https://www.cityscapes-dataset.com/ and store in <data_dir>/cs.")
       dirs = []
       splits = os.listdir(top_dir)
       for split in splits:
           split_path = os.path.join(top_dir, split)
           cities = os.listdir(split_path)
           for city in cities:
               city_path = os.path.join(split_path, city)
               dirs.append(city_path)
       return dirs
