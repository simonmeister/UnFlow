"""Utility functions for providing data directories."""
import os
import sys
import zipfile
import rarfile
from urllib.request import FancyURLopener
import shutil

import numpy as np
import matplotlib.image as mpimg


class Data():
    # Should be a list containing all subdirectories of the main data dir which
    # belong to this dataset
    dirs = None

    def __init__(self, data_dir, stat_log_dir,
                 development=True, fast_dir=None):
        self.development = development
        self.data_dir = data_dir
        self.stat_log_dir = stat_log_dir
        if not os.path.isdir(data_dir):
            os.makedirs(data_dir)

        self._fetch_if_missing()

        self.fast_dir = fast_dir
        if fast_dir:
            print(">> Copying files to {}".format(fast_dir))
            for d in self.dirs:
                src = os.path.join(data_dir, d)
                dst = os.path.join(fast_dir, d)
                if not os.path.isdir(dst):
                    shutil.copytree(src, dst)
                    print(">> Copied {}".format(d))
            self.current_dir = fast_dir
        else:
            self.current_dir = data_dir

        if stat_log_dir:
            self.stat_log_file = os.path.join(stat_log_dir,
                                              self.__class__.__name__ + ".txt")
            self._ensure_statistics()

    def __del__(self):
        pass
        #if self.fast_dir:
        #    print(">> Removing files from {}".format(self.fast_dir))
        #    for d in self.dirs:
        #        shutil.rmtree(os.path.join(self.fast_dir, d))

    def clear_statistics(self):
        """Delete saved statistics file if present."""
        if self.stat_log_dir and os.path.isfile(self.stat_log_file):
            os.remove(self.stat_log_file)

    def _ensure_statistics(self):
        """Make sure we know the dataset statistics."""
        if os.path.isfile(self.stat_log_file):
            vals = np.loadtxt(self.stat_log_file)
            self.mean = vals[0]
            self.stddev = vals[1]
        else:
            print(">> Computing statistics (mean, variance) for {}"
                  .format(self.__class__.__name__))
            mean, stddev = self.compute_statistics(self.get_raw_files())
            self.mean = mean
            self.stddev = stddev
            os.makedirs(self.stat_log_dir, exist_ok=True)
            np.savetxt(self.stat_log_file, [mean, stddev])
            print(">> Statistics complete")

    def get_raw_dirs(self):
        """Should return a list of all dirs containing training images.

        Note: self.current_dir should be used for loading input data.
        """
        raise NotImplementedError()

    def get_raw_files(self):
        files = []
        for d in self.get_raw_dirs():
            for path in os.listdir(d):
                files.append(os.path.join(d, path))
        return files

    def _fetch_if_missing(self):
        """A call to this must make subsequent calls to get_raw_files succeed.
        All subdirs of data_dir listed in self.dirs must exist after this call.
        """
        raise NotImplementedError()

    def _download_and_extract(self, url, extract_to, ext='zip'):
        def _progress(count, block_size, total_size):
            if total_size > 0:
                print('\r>> Downloading %s %.1f%%' % (url,
                      float(count * block_size) / float(total_size) * 100.0), end=' ')
            else:
                print('\r>> Downloading %s' % (url), end=' ')
            sys.stdout.flush()
        urlretrieve = FancyURLopener().retrieve
        local_zip_path = os.path.join(self.data_dir, 'tmp.' + ext)
        urlretrieve(url, local_zip_path, _progress)
        sys.stdout.write("\n>> Finished downloading. Unzipping...\n")
        if ext == 'zip':
            with zipfile.ZipFile(local_zip_path, "r") as zip_ref:
                zip_ref.extractall(extract_to)
        else:
            with rarfile.RarFile(local_zip_path, "r") as zip_ref:
                zip_ref.extractall(extract_to)

        sys.stdout.write(">> Finished unzipping.\n")
        os.remove(local_zip_path)

        self.clear_statistics()

    def compute_statistics(self, files):
        """Use welford's method to compute mean and variance of the given
        dataset.

        See https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Online_algorithm."""

        assert len(files) > 1

        n = 0
        mean = np.zeros(3)
        M2 = np.zeros(3)
        for j, filename in enumerate(files):
            #TODO ensure the pixel values are 0..255
            im = np.reshape(mpimg.imread(filename) * 255, [-1, 3])
            for i in range(np.shape(im)[1]):
                n = n + 1
                delta = im[i] - mean
                mean += delta / n
                M2 += delta * (im[i] - mean)
            sys.stdout.write('\r>> Processed %.1f%%' % (
                float(j) / float(len(files)) * 100.0))
            sys.stdout.flush()
        var = M2 / (n - 1)
        stddev = np.sqrt(var)
        return np.float32(mean), np.float32(stddev)
