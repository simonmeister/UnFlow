import os
import subprocess
import datetime
from shutil import copyfile, rmtree

import tensorflow as tf

from .util import config_dict


class Experiment():
    def __init__(self, name, overwrite=False):
        global_config = config_dict()
        log_dir = os.path.join(global_config['dirs']['log'], 'ex', name)
        dirs = global_config['dirs']

        train_dir = os.path.join(log_dir, 'train')
        eval_dir = os.path.join(log_dir, 'eval')
        save_dir = os.path.join(dirs['checkpoints'], name)

        def _init_dirs():
            os.makedirs(log_dir)
            os.makedirs(save_dir)
            os.makedirs(train_dir)
            os.makedirs(eval_dir)

        # Experiment already exists
        if os.path.isdir(log_dir):
            if overwrite:
                rmtree(log_dir)
                if os.path.isdir(save_dir):
                    rmtree(save_dir)
                _init_dirs()

            else:
                if not os.path.isdir(save_dir):
                    os.makedirs(save_dir)
                    # Copy stored checkpoint in case intermediate checkpoints
                    # were deleted
                    ckpt = self._copy_latest_checkpoint(log_dir, save_dir)
                    if not ckpt:
                        raise RuntimeError('Failed to restore "{}".'
                                           'Use --overwrite=True to clear.'
                                           .format(name))
                    print('Warning: intermediate checkpoints could not be restored.')
        else:
            _init_dirs()

        config_path = os.path.join(log_dir, 'config.ini')
        if not os.path.isfile(config_path) or overwrite:
            copyfile('../config.ini', config_path)
        config = config_dict(config_path)

        self.train_dir = train_dir
        self.eval_dir = eval_dir
        self.save_dir = save_dir
        self.log_dir = log_dir
        self.name = name
        self.config = config

    def latest_checkpoint(self):
        return tf.train.latest_checkpoint(self.save_dir)

    def _copy_latest_checkpoint(self, src, dst, reset_global_step=False):
        ckpt = tf.train.latest_checkpoint(src)
        if ckpt:
            ckpt_base = os.path.basename(ckpt)
            new_base = 'model.ckpt-0' if reset_global_step else ckpt_base
            with open(os.path.join(dst, 'checkpoint'), 'w') as f:
                f.write('model_checkpoint_path: "' + new_base + '"\n')
                f.write('all_model_checkpoint_paths: "' + new_base + '"\n')
            for filename in os.listdir(src):
                if ckpt_base in filename:
                    new_filename = filename.replace(ckpt_base, new_base)
                    copyfile(os.path.join(src, filename),
                             os.path.join(dst, new_filename))
        return ckpt

    def conclude(self):
        """Move final checkpoint to the permanent log dir."""
        ckpt = self._copy_latest_checkpoint(self.save_dir, self.log_dir)
        if not ckpt:
            print('Warning: no checkpoints written')
