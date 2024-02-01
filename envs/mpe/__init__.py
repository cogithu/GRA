

import socket
from absl import flags
import os.path as osp
import imp



FLAGS = flags.FLAGS
FLAGS(['train_sc.py'])
def load(name):
    pathname = osp.join(osp.dirname(__file__), name)
    return imp.load_source('', pathname)

