import os
import sys

import numpy as np
import matplotlib.image as mpimg

from ..core.data import Data
from ..util import tryremove

URL = 'http://synthia-dataset.cvc.uab.cat/SYNTHIA_SEQS/'
SEQS = [ # SUMMER and WINTER from sequences `1 - 6`
    'SYNTHIA-SEQS-01-SUMMER',
    'SYNTHIA-SEQS-01-WINTER',
    'SYNTHIA-SEQS-02-SUMMER',
    'SYNTHIA-SEQS-02-WINTER',
    'SYNTHIA-SEQS-04-SUMMER',
    'SYNTHIA-SEQS-04-WINTER',
    'SYNTHIA-SEQS-05-SUMMER',
    'SYNTHIA-SEQS-05-WINTER',
    'SYNTHIA-SEQS-06-SUMMER',
    'SYNTHIA-SEQS-06-WINTER'
]

DEV_SEQS = ['SYNTHIA-SEQS-01-SUMMER']


class SynthiaData(Data):
    dirs = ['synthia']

    def __init__(self, data_dir, stat_log_dir=None,
                 development=True, fast_dir=None):
        super().__init__(data_dir, stat_log_dir,
                         development=development,
                         fast_dir=fast_dir)

    def _fetch_if_missing(self):
        self._maybe_get_synthia()

    def get_raw_dirs(self):
        root_dir = os.path.join(self.current_dir, 'synthia')
        dirs = []
        seqs = os.listdir(root_dir)
        for seq in seqs:
            seq_dir = os.path.join(root_dir, seq, seq, 'RGB', 'Stereo_Left')
            views = os.listdir(seq_dir)
            for view in views:
                view_dir = os.path.join(seq_dir, view)
                dirs.extend([view_dir])
        return dirs

    def _maybe_get_synthia(self):
        seqs = DEV_SEQS if self.development else SEQS
        for seq in seqs:
            root_dir = os.path.join(self.data_dir, 'synthia')
            url = URL + seq + '.rar'
            url_dir = os.path.join(root_dir, seq)
            if not os.path.isdir(url_dir):
                self._download_and_extract(url, url_dir, 'rar')

            # Remove unused directories
            tryremove(os.path.join(url_dir, seq, 'GT'))
            tryremove(os.path.join(url_dir, seq, 'Depth'))
            tryremove(os.path.join(url_dir, seq, 'CameraParams'))
            tryremove(os.path.join(url_dir, 'RGB', 'Stereo_Right'))
