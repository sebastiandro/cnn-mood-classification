# Check if we have Tensorflow GPU
# tf.__version__sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
import tensorflow as tf
config=tf.ConfigProto(log_device_placement=True)
config.gpu_options.allow_growth=True

tf.__version__sess = tf.Session(config=config)
from tensorflow.python.client import device_lib

def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']

get_available_gpus()

