import tensorflow as tf
from os import path

class CheckpointSaver():
    
    def __init__(self, name):
        checkpoint_path = "tf_checkpoints"
        self.chk_path = "{}/{}.ckpt".format(checkpoint_path, name)
        self.sess = tf.get_default_session()
        self.saver = tf.train.Saver()
    
    def save_checkpoint(self):
        self.saver.save(self.sess, self.chk_path)

    def restore_checkpoint(self):
        if path.exists(self.chk_path+".index"):
            self.saver.restore(self.sess, self.chk_path)
            return True
        
        return False