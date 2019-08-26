import tensorflow as tf
from tensorflow.contrib.layers import variance_scaling_initializer as he_initializer
from tensorflow.nn import sparse_softmax_cross_entropy_with_logits as softmax_xentropy
from tensorflow.layers import dense
import numpy as np


def get_leaky_relu(alpha):
    return lambda z, name=None: tf.maximum(alpha*z,z, name=name)
    

def get_connected_layers(x, n_hidden_layers, n_neurons, n_ouputs, activation=tf.nn.elu,
                                   batch_norm_momentum=None, dropout_rate=None, is_training=None):
    

    initializer = he_initializer()
    
    with tf.name_scope("DNN"):
        inputs = x
        for l in range(n_hidden_layers):
            if dropout_rate is not None:
                ## this function will set inputs to zero with dropout rate probability
                ## and divides remaining inputs with dropout rate
                inputs = tf.layers.dropout(inputs, dropout_rate, training=is_training, 
                                  name=("dropout%d"%l))
                
            inputs = tf.layers.dense(inputs, n_neurons, kernel_initializer=initializer,
                           name="hidden%d"%(l+1), activation=activation)
            
            if batch_norm_momentum is not None:
                inputs = tf.layers.batch_normalization(inputs, momentum=batch_norm_momentum,
                                training=is_training)
            
            inputs = activation(inputs, name="hiden%d_out"%(l+1))
            
        output = tf.layers.dense(inputs, n_ouputs, name="output")
        
    return output
        


def get_softmax_xentropy_loss(logits,y):
    with tf.name_scope("loss"):
        xentropy = softmax_xentropy(labels=y, logits=logits)
        return tf.reduce_mean(xentropy, name="mean_loss")

def get_optimizer_op(optimizer, loss, learning_rate=0.01):
    with tf.name_scope("train"):
        optimizer =  optimizer(learning_rate=learning_rate)
        optimizer_op = optimizer.minimize(loss, name="optimizer_op")
    return optimizer_op

def get_validation_score(logits,y):
    with tf.name_scope("validation"):
        preds = tf.nn.in_top_k(logits,y,1)
        return tf.reduce_mean(tf.cast(preds, dtype=np.float32), name="validation_score")
    
def get_batch(x,y,batch_size):
    n_batches = len(y)//batch_size + 1
    for i in range(n_batches):
        indxes = np.random.choice(len(y), size=batch_size, replace=False)
        yield x[indxes], y[indxes]


from sklearn.base import BaseEstimator, TransformerMixin
from my_libs.tf_graph_saver import ScalerGraphSaver2
from sklearn.exceptions import NotFittedError



class DNN_Classifier(BaseEstimator, TransformerMixin):
    def __init__(self, n_hidden_layers=None, n_neurons=None, n_outputs=None, 
                 activation=tf.nn.elu, optimizer=tf.train.AdamOptimizer,  learning_rate=0.01, 
                 batch_norm_momentum=None, batch_size=50, dropout_rate=None):
        self.n_hidden_layers = n_hidden_layers
        self.n_neurons = n_neurons
        self.learning_rate = learning_rate
        self.activation = activation
        self.optimizer = optimizer
        self.batch_norm_momentum = batch_norm_momentum
        self.batch_size = batch_size
        self.dropout_rate = dropout_rate
        self.n_outputs = n_outputs
        self._session = None
        self._graph = None
        
        
    def _create_graph(self):                      
        
        tf.reset_default_graph()
        self._graph = tf.Graph()
        with self._graph.as_default():
        
            self._x = tf.placeholder(shape=(None, 28*28), dtype=np.float32,name="x")
            self._y = tf.placeholder(shape=(None), dtype=np.int32,name="y")

            self._is_training = tf.placeholder_with_default(False,shape=(), name="is_training")


            self._dnn = get_connected_layers(self._x, self.n_hidden_layers, self.n_neurons, 
                                       self.n_outputs, activation=self.activation, 
                                       batch_norm_momentum=self.batch_norm_momentum, 
                                       dropout_rate=self.dropout_rate, is_training=self._is_training)
            self._loss = get_softmax_xentropy_loss(self._dnn, self._y)
            self._optimizer_op = get_optimizer_op(self.optimizer, self._loss, 
                                                  self.learning_rate)
            self._validation_score = get_validation_score(self._dnn, self._y)

            self._y_proba = tf.nn.softmax(self._dnn, name="y_proba")

            self._batch_norm_update_ops = self._graph.get_collection(tf.GraphKeys.UPDATE_OPS)
            self._saver = tf.train.Saver()
            
            
        
    def _save_params(self):
        with self._graph.as_default():
            global_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
            
        vars_n_values = {global_var.op.name:value for global_var, value in \
                 zip(global_vars,self._session.run(global_vars))}
        self._saved_params =  vars_n_values
        
    
    def _restore_params(self):
        var_names = list(self._saved_params.keys())
        
        ## get assign operations for all variables
        assign_ops = {var_name:self._graph.get_operation_by_name("%s/Assign"%var_name) 
                      for var_name in var_names}
        ## get initialization values of all variables
        init_values = {var_name: assign_op.inputs[1]  for var_name, assign_op 
                       in assign_ops.items()}
        
        ## get feed_dict for all values
        feed_dict = {init_values[var_name]:self._saved_params[var_name] 
                     for var_name in var_names}
        
        
        self._session.run(assign_ops, feed_dict=feed_dict)
        
    
    def fit(self,x,y,x_val,y_val):
        n_epoches = 500
        max_epoches_wo_progress = 100  
        
        best_score=0
        best_epoch=0
        
        self._initialize_session_and_graph() 
        with self._session.as_default() as sess:
            # sess.run(self._init)
            
            graph_saver = ScalerGraphSaver2("DNN_GridSearch")
            loss_summary = graph_saver.get_summary_op("loss", self._loss)
            score_summary = graph_saver.get_summary_op("accuracy_score", self._validation_score)
            
            with graph_saver:
        
                for epoch in range(n_epoches):
                    for batch_x, batch_y in get_batch(x,y,self.batch_size):
                        ops = [self._loss, loss_summary, self._optimizer_op]
                        if self._batch_norm_update_ops is not None:
                            ops.append(self._batch_norm_update_ops)

                        results = sess.run(ops , feed_dict={self._x:batch_x, self._y:batch_y, 
                                               self._is_training:True})
                        loss = results[0]
                        loss_summary_text = results[1]



                    score, score_summary_text = sess.run([self._validation_score, score_summary], 
                                     feed_dict={self._x:x_val, self._y:y_val})
                    graph_saver.log_summary(loss_summary_text, epoch)
                    graph_saver.log_summary(score_summary_text, epoch)
                    
                    if epoch%50 == 0:
                        print("epoch %d, score %f, loss %f"%(epoch, score, loss))

                    if score > best_score:
                        best_score = score
                        best_epoch = epoch
                        self._save_params()
                    elif (epoch - best_epoch)>max_epoches_wo_progress:
                        print("No progress for %d epoches."%max_epoches_wo_progress)
                        break
                
            self._restore_params()
            print("Reverting back to epoch %d \
                    with %f score" %(best_epoch, best_score))
            self._score = best_score 
            return self
            
                    
    
    def predict_proba(self,x):
        if self._session is None:
            raise NotFittedError("%s is not fitted yet" \
                                                    %self.__class__.__name__)
        
        return self._session.run(self._y_proba, feed_dict={self._x:x, 
                                                           self._is_training:False})
            
    
    def predict(self,x):
        return np.argmax(self.predict_proba(x), axis=1)
    
    def score(self, x_val=None, y_val=None):
        
        score=self._session.run(self._validation_score, 
                             feed_dict={self._x:x_val, self._y:y_val})
        print("validation score: %f", score)
        return score
    
    def _get_save_path(self, name):
        return "tf_checkpoints/%s"%name
    
    def save(self,name):
        self._saver.save(self._session, self._get_save_path(name))
    
    def _restore_graph(self, name):
        tf.reset_default_graph()
        self._imported_meta = tf.train.import_meta_graph("%s.meta"%self._get_save_path(name))
        graph = tf.get_default_graph()
        self._x = graph.get_tensor_by_name("x:0")
        self._y = graph.get_tensor_by_name("y:0")
        self._loss = graph.get_tensor_by_name("loss/mean_loss:0")
        
        self._validation_score = graph.get_tensor_by_name("validation/validation_score:0")
        self._y_proba = graph.get_tensor_by_name("y_proba:0")
        self._is_training = graph.get_tensor_by_name("is_training:0")
        self._graph = graph


    def _initialize_session(self):
        if self._session: 
            self._session.close()
        
        self._session = tf.Session(graph=self._graph)
        with self._graph.as_default():
            init = tf.global_variables_initializer()
        self._session.run(init)


    def _restore_session(self, name):       
        self._initialize_session()
        self._imported_meta.restore(self._session, self._get_save_path(name))
    
    def _initialize_session_and_graph(self):
        self._create_graph()
        self._initialize_session()
    
    def restore(self, name):
        self._restore_graph(name)
        self._restore_session(name)