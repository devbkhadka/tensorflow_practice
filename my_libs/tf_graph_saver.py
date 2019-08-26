from datetime import datetime
from os import path
import tensorflow as tf

class ScalerGraphSaver():
    '''
    This class logs graph summary in a directory which can be viewed using tensorboard command
    tensorboard --logdir <dir path>
    '''
    def __init__(self, dir_name):
        root_dir = path.join("tf_logs",dir_name)
        now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
        self.log_dir = "{}/run_{}/".format(root_dir, now)
        self.scalars = {}
        
        
    def __enter__(self):
        self.file_writer = tf.compat.v1.summary.FileWriter(self.log_dir, tf.get_default_graph())
        return self
    
    def __exit__(self, type_, value, traceback):
        self.file_writer.close()
        
    
    def log_summary(self, name, scalar, step, feed_dict=None):
        scalar_summary = self.scalars.get(name, tf.compat.v1.summary.scalar(name, scalar))
        self.scalars[name] = scalar_summary
        mse_str = scalar_summary.eval(feed_dict=feed_dict)
        self.file_writer.add_summary(mse_str, step)
        
 

class ScalerGraphSaver2():
    '''
    This class logs graph summary in a directory which can be viewed using tensorboard command
    tensorboard --logdir <dir path>
    '''
    def __init__(self, dir_name):
        root_dir = path.join("tf_logs",dir_name)
        now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
        self.log_dir = "{}/run_{}/".format(root_dir, now)
        self.scalars = {}
        
        
    def __enter__(self):
        self.file_writer = tf.compat.v1.summary.FileWriter(self.log_dir, tf.get_default_graph())
        return self
    
    def __exit__(self, type_, value, traceback):
        self.file_writer.close()
        
    
    def get_summary_op(self, name, scalar):
        return tf.compat.v1.summary.scalar(name, scalar)
        
    def log_summary(self, summary_text, step):
        self.file_writer.add_summary(summary_text, step)
        
        