import os
import subprocess
import configparser
from shutil import rmtree
import tensorflow as tf


CONFIG_PATH = '../config.ini'
#TMP_DIR = '/tmp/e2eflow'


def upload_gdrive(upload_dir, gdrive_filename):
    # search for file in gdrive and capture id if it already exists
    lst_lines = subprocess.Popen(['../scripts/gdrive', 'list'],
                                 stdout=subprocess.PIPE)
    existing_id = None
    for line in lst_lines.stdout:
        splits = line.split()
        if str(splits[1], 'utf-8') == gdrive_filename:
            existing_id = str(splits[0], 'utf-8')
    tmp_path = os.path.join('/tmp', gdrive_filename)
    if os.path.isfile(tmp_path):
        os.remove(tmp_path)
    p = subprocess.Popen(['/usr/bin/zip', '-r', tmp_path, upload_dir])
    p.wait()
    if existing_id:
        p = subprocess.Popen(['../scripts/gdrive', 'update',
                              existing_id, tmp_path])
    else:
        p = subprocess.Popen(['../scripts/gdrive', 'upload',
                              '--name', gdrive_filename,
                               tmp_path])
    p.wait()
    os.remove(tmp_path)


def config_dict(config_path=CONFIG_PATH):
    """Returns the config as dictionary,
    where the elements have intuitively correct types.
    """

    config = configparser.ConfigParser()
    config.read(config_path)

    d = dict()
    for section_key in config.sections():
        sd = dict()
        section = config[section_key]
        for key in section:
            val = section[key]
            try:
                sd[key] = int(val)
            except ValueError:
                try:
                    sd[key] = float(val)
                except ValueError:
                    try:
                        sd[key] = section.getboolean(key)
                    except ValueError:
                        sd[key] = val
        d[section_key] = sd
    return d


def convert_input_strings(config_dct, dirs):
    if 'manual_decay_iters' in config_dct and 'manual_decay_lrs' in config_dct:
        iters_lst = config_dct['manual_decay_iters'].split(',')
        lrs_lst = config_dct['manual_decay_lrs'].split(',')
        iters_lst = [int(i) for i in iters_lst]
        lrs_lst = [float(l) for l in lrs_lst]
        config_dct['manual_decay_iters'] = iters_lst
        config_dct['manual_decay_lrs'] = lrs_lst
        config_dct['num_iters'] = sum(iters_lst)

    if 'finetune' in config_dct:
        finetune = []
        for name in config_dct['finetune'].split(","):
            ckpt_dir = os.path.join(dirs['checkpoints'], name)
            ckpt = tf.train.get_checkpoint_state(ckpt_dir)
            if ckpt is None:
              ckpt_dir = os.path.join(dirs['log'], 'ex', name)
              ckpt = tf.train.get_checkpoint_state(ckpt_dir)
            assert ckpt, "Could not load experiment " + name
            finetune.append(ckpt)
        config_dct['finetune'] = finetune


def tryremove(name, file=False):
    try:
        if file:
            os.remove(name)
        else:
            rmtree(name)
    except OSError:
        pass
