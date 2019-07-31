from datetime import datetime
from os import path
import tensorflow as tf

class ScalerGraphSaver():
    '''
    class logs graph summary in a directory which can be viewed using tensorboard command
    tensorboard --logdir <dir path>
    '''
    def __init__(self, dir_name, scaler):
        root_dir = path.join("tf_logs",dir_name)
        now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
        self.log_dir = "{}/run_{}/".format(root_dir, now)
        self.mse_summary = tf.summary.scalar("MSE", scaler)
        
        
    def __enter__(self):
        self.file_writer = tf.summary.FileWriter(self.log_dir, tf.get_default_graph())
        return self
    
    def __exit__(self, type_, value, traceback):
        self.file_writer.close()
        
        
    def log_summary(self,step, feed_dict=None):
        mse_str = self.mse_summary.eval(feed_dict=feed_dict)
        self.file_writer.add_summary(mse_str, step)
        return mse_str