from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import tempfile
import csv
import time
import pickle
import glob
import cv2
import scipy
import os
import Cython
from distutils.core import setup
from Cython.Build import cythonize
from ctypes import *

import tensorflow as tf
import numpy as np

import random

import numpy as np
from matplotlib import *
from scipy.misc import imread, imsave
from tensorflow.contrib.learn.python.learn.datasets import base

from six.moves import xrange  # pylint: disable=redefined-builtin

from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import random_seed
from tensorflow.python.platform import gfile
from tensorflow.contrib.learn.python import __init__
from tensorflow.python.saved_model.loader_impl import maybe_saved_model_directory
from tensorflow.contrib.learn.python.learn.datasets import base

from tqdm import tqdm

from tensorflow.python.ops import gen_math_ops

FLAGS = None
DEFAULT_PADDING = 'SAME'