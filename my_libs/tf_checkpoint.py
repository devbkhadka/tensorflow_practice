import tensorflow as tf
from os import path

class CheckpointSaver():
    
    def __init__(self):
        self._checkpoint_path = "tf_checkpoints"
        self.sess = tf.get_default_session()
        self.saver = tf.train.Saver()
    
    def save_checkpoint(self, name):
        chk_path = "{}/{}.ckpt".format(self._checkpoint_path, name)
        self.saver.save(self.sess, chk_path)

    def restore_checkpoint(self, name):
        chk_path = "{}/{}.ckpt".format(self._checkpoint_path, name)
        if path.exists(chk_path+".index"):
            self.saver.restore(self.sess, chk_path)
            return True
        
        return False