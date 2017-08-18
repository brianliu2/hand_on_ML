# -*- coding: utf-8 -*-
"""
This is the implementation of exercise 1 in deep learning
chapter

@author: xliu
"""

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.exceptions import NotFittedError
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

'''
A class of deep neural network 
1.  __init__: self-container
2.  _dnn: hidden layers
3.  _build_graph: architecture of dnn with dropout, hidden layers, batch norm
				  and activation function
4.  _close_session: closs session activated, because only one session can exist
5.  _get_model_param: retrieve all paramters from dnn model
6.  _restore_model_param: restore/reload all parameters rather than save then 
                         restore a ckpt model
7.  fit: function to build dnn
8.  predict_proba: produce predicted probabilities for each instance for all classes/labels
9.  predict: produce a single predicted class for each instance
10. save: a functin to save the final model
'''
class DNNClassifier(BaseEstimator, ClassifierMixin):
	'''
	1. initialize the self container
	'''
	def __init__(self, n_hidden_layers = 5, num_units_per_layer = [100, 100, 70, 50, 30],\
				  activationFcn = tf.nn.relu, optimizer = tf.train.AdamOptimizer,\
				  kernel_initializer = tf.contrib.layers.variance_scaling_initializer(),\
				  learning_rate = 0.01, n_epochs = 100, batch_size = 20,\
				  batch_norm_momentum = 0.9, batch_norm_use = None,\
				  dropout_rate = 0.5, dropout_use = None, random_state = None):
		
		self.n_hidden_layers = n_hidden_layers
		self.num_units_per_layer = num_units_per_layer
		self.activationFcn = activationFcn
		self.kernel_initializer = kernel_initializer
		self.learning_rate = learning_rate
		self.n_epochs = n_epochs
		self.batch_size = batch_size
		self.optimizer = optimizer
		self.batch_norm_momentum = batch_norm_momentum
		self.batch_norm_use = batch_norm_use
		self.dropout_rate = dropout_rate
		self.dropout_use  = dropout_use
		self.random_state = random_state
		
		# Important: to pre-allocate a session for executing actions after
		# graph is constructed
		self._session = None
	
	'''
	2. _dnn: an externally uncallable function to construct hidden layers
	
	--- proceduer: dropout (if capable) -> matrix multiplication -> batch norm (if capable) -> activation
	
	--- Input: Inputs (the first inputs is X while the rest are 'outputs' from 
	                   the previous layer)
	'''
	def _dnn(self, inputs):
		for layer in range(self.n_hidden_layers):
			# 2.1 if dropout is used
			if self.dropout_use:
				inputs = tf.layers.dropout(inputs, rate = self.dropout_rate, training = self._training)
			
			# 2.2 construct the matrix multi
			inputs = tf.layers.dense(inputs, self.num_units_per_layer[layer],\
					  kernel_initializer = self.kernel_initializer,\
					  name = 'hidden_%d'%(layer+1))
		
			# 2.3 if batch normalization is used 
			if self.batch_norm_use:
				inputs = tf.layers.batch_normalization(inputs, momentum = self.batch_norm_momentum,\
						  training = self._training)
		
			# 2.4 layer output after passing through activation function
			inputs = self.activationFcn(inputs, name = 'hidden_%d_output'%(layer + 1))
		return inputs
	
	'''
	3. _build_graph: an externally uncallable function to construct the computing
	                 graph for dnn
	   
	   --- Inputs: 
				dim_input:  the dimension of attributes for each instance in feature variable
				n_classes:  the number of types of response variable
	   --- Outputs:
				this is a uncallable function, so it typically has no outputs
				but fill all stuffs in self-container
	'''
	def _build_graph(self, n_inputs, n_classes):
		# 3.1 if random_state is specified, then we set tf and np
		if self.random_state:
			tf.set_random_seed(self.random_state)
			np.random.seed(self.random_state)
		
		# 3.2 pre-allocate computing node for training data
		X = tf.placeholder(dtype = tf.float32, shape = (None, n_inputs), name = 'X')
		y = tf.placeholder(dtype = tf.int64, shape = (None), name = 'y')
		
		# 3.3 create indicator for batch_norm or drop out if neccessary
		if self.batch_norm_use is not None or self.dropout_use is not None:
			self._training = tf.placeholder_with_default(False, shape = (), name = 'training')
		else:
			self._training = None
		
		# 3.4 create layers until the last fully connect layer by using _dnn function
		dnn_hidden_layers = self._dnn(X)
		
		# 3.5 create the fully connected layer
		logits = tf.layers.dense(dnn_hidden_layers, self.n_classes_, \
								  kernel_initializer = self.kernel_initializer,\
								  name = 'logits')
		
		# 3.6 create the score layer (here, we use softmax because mnist is a multi-class problem)
		scores = tf.nn.softmax(logits, name = 'scores')
		
		
		# 3.7 evaluate cross_entropy and loss
#		cross_entropy = tf.losses.sparse_softmax_cross_entropy(labels = y, logits = logits)
		cross_entropy = tf.losses.sparse_softmax_cross_entropy(labels = y, logits = logits)
		loss = tf.reduce_mean(cross_entropy, name = 'loss')
		
		# 3.8 specify optimizer and train_obj
		optimizer = self.optimizer(learning_rate = self.learning_rate)
		train_obj = optimizer.minimize(loss)
		
		# 3.9 evaluate the accuracy of current dnn
		correct = tf.nn.in_top_k(predictions = logits, targets = y, k = 1)
		accuracy = tf.reduce_mean(tf.cast(correct, tf.float32), name = 'accuracy')
		
		# 3.10 specify initilizer for global variables and saver
		init = tf.global_variables_initializer()
		saver = tf.train.Saver()
		
		# 3.11 fill all stuffs in self-container
		self._X, self._y = X, y
		self._scores, self._loss = scores, loss
		self._train_obj = train_obj
		self._accuracy  = accuracy
		self._init, self._saver = init, saver
	
	'''
	4. _close_session: closs session activated, because only one session can exist
	'''
	def close_session(self):
		if self._session:
			self._session.close()
	
	'''
	5. get_model_param: retrieve all paramters from dnn model
	'''
	def _get_model_param(self):
		# 5.1 we need to get the default/current graph's global variables/parameters
		with self._graph.as_default():
			model_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
		
		# 5.2 create a dictionary to store global parameters and their values in current computing graph
		model_params_dict = {param.op.name: value for param, value in \
							   zip(model_params, self._session.run(model_params))}
		return model_params_dict
	
	'''
	6. _restore_model_param: restore/reload all parameters rather than save then 
                             restore a ckpt model
	   --- Input:
				model_params_dict: global variables existing in the current graph 
				and they are obtained from _get_model_param()
	   --- Output:
				No output, and we need to fill all information in self-container
	'''
	def _restore_model_param(self, model_params_dict):
		# 6.1 get all parameter names from dictionary
		model_params_name = list(model_params_dict.keys())
		
		# 6.2 (create a node without execution) assign operation to 
		#      update/retrieve global parameter values
		assign_ops = {model_param_name:  self._graph.get_operation_by_name(model_param_name + '/Assign')\
					   for model_param_name in model_params_name}
			
		# 6.3 the concrete values that we want to set to global parameters via assign operation
		init_values = {model_param_name: assign_op.inputs[1] \
						for model_param_name, assign_op in assign_ops.items()}
		
		# 6.4 In order to execute assign operation after having computing node and specific values
		#     we need to create a dictionary so that we can sess.run(operation, feed_dict={param: value})
		feed_dict = {init_values[model_param_name]: model_params_dict[model_param_name] 
					  for model_param_name in model_params_name}
		
		# 6.5 run the operation through session
		self._session.run(assign_ops, feed_dict)
	
	'''
	7. fit: function to build dnn
	--- Inputs:
			X: features from training data
			y: responses from training data
			X_valid: features from validation data, we use early stopping when validation is available
			y_valid: responses from validation data
	
	--- Output:
			fill all stuffs in self-container
	'''
	def fit(self, X, y, X_valid = None, y_valid = None):
		# 7.1 close the existed session if there is one
		self.close_session()
		
		# 7.2 retrieve information about training data
		n_inputs = X.shape[1]
		n_classes = len(np.unique(y))
		classes   = np.unique(y)
		self.n_classes_ = n_classes
		self.classes_ = classes
		
		# 7.3 retrieve algorithmic settings from self-container for performing fit procedure
		n_epochs = self.n_epochs
		batch_size = self.batch_size
		
		# 7.4.1 Translate the labels vector to a vector of sorted class indices, containing
		# integers from 0 to n_outputs - 1.
		# For example, if y is equal to [8, 8, 9, 5, 7, 6, 6, 6], then the sorted class
		# labels (self.classes_) will be equal to [5, 6, 7, 8, 9], and the labels vector
		# will be translated to [3, 3, 4, 0, 2, 1, 1, 1]
		class_to_index = {label: idx for idx, label in enumerate(self.classes_)}
		
		# 7.4.2 we need to fill the translation from class to index to self-container,
		# because we will re-use when we do prediction for test data
		self.class_to_index_ = class_to_index
		
		# 7.4.3 we translate the response variables into indice and create an array to store 
		y = np.array([self.class_to_index_[label] for label in y], dtype = np.int32)
		
		
		# 7.5 we need to create a computing graph to execute all actions
		self._graph = tf.Graph()

		# 7.6 we now have an empty computing graph, within this graph, we need to construct
		# dnn to be built and see if there is extra update operations existed for batch norm
		with self._graph.as_default():
			self._build_graph(n_inputs, n_classes)
			# see there is extra update operations for batch norm or drop out
			extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
		
		# 7.7 create the session to execute all actions 
		# with specifying the current computing graph
		self._session = tf.Session(graph = self._graph)
		
		# 7.8 initialize settings for early stopping
		max_check_without_progress = 20
		check_without_progress = 0
		best_loss_valid = np.infty
		best_params = None
		
		# 7.9 --- Main loop
		with self._session.as_default() as sess:
			
			# 7.9.0 initialize global variables
			sess.run(self._init)			
			
			# 7.9.1 loop over epochs
			for epoch in range(self.n_epochs):
				# ---------------------------------------------------------------------- #
				# ----------------------   start one epoch ----------------------------- #
				
				# 7.9.1.1 shuffle indice of training data
				shuffled_idx = np.random.permutation(len(X))
				
				# 7.9.1.2 split shuffled indice to generate mini batch sets
				for idx in np.array_split(shuffled_idx, (len(X) // self.batch_size)):
					# 7.9.1.2.1 get the current mini batch based the index					
					X_batch, y_batch = X[idx], y[idx]
					
					# 7.9.1.2.2 construct feed_dict for executing train_obj
					feed_dict = {self._X: X_batch, self._y: y_batch}
					
					# 7.9.1.2.3 if we use batch norm or drop out, 
					# then we set training in feed_dict to True
					if self._training is not None:
						feed_dict[self._training] = True
					
					# 7.9.1.2.4 execute the train_obj
					sess.run(self._train_obj, feed_dict = feed_dict)
					
					# 7.9.1.2.5 if there is extra_update_ops, then we 
					# execute extra_update_ops to update for batch norm
					if extra_update_ops is not None:
						sess.run(extra_update_ops, feed_dict)
				# ----------------------   end one epoch   ------------------------------- #
				# ------------------------------------------------------------------------ #
				
				# 7.9.1.3 In this section, we need to either evaluate loss_valid if capable
				# and determine whether we should early stopping
				# or evaluate loss_train if there is no validation data provided.
				if X_valid is not None and y_valid is not None:
					# 7.9.1.3.1 evaluate validation loss and accuracy
					loss_valid, acc_valid = sess.run([self._loss, self._accuracy],\
														 feed_dict = {self._X: X_valid, self._y: y_valid})
					
					# 7.9.1.3.2 print out the training status
					print('Epoch {}\tValidation Loss: {:.6f}\tBest Loss: {:.6f}\tValidation Accuracy: {:.2f}%'\
					.format(epoch, loss_valid, best_loss_valid, acc_valid * 100))
					
					# 7.9.1.3.3 determine whether we should early stopping training
					if loss_valid > best_loss_valid:
						best_loss_valid = loss_valid
						check_without_progress = 0
						best_params = self._get_model_param()
					else:
						check_without_progress += 1
						if check_without_progress > max_check_without_progress:
							print('Early Stopping !')
							break
				
				# 7.9.1.4 if there is no validation data, then we evaluate training loss and accuracy
				else:
					loss_train, acc_training = sess.run([self._loss, self._accuracy],\
					feed_dict = {self._X: X_batch, self._y: y_batch})
					print('Epoch {}\tTraining Loss: {:.6f}\tTraining Accuracy: {:.2f}%'\
					.format(epoch, loss_train, acc_training * 100))
					
				
		# 7. 10 If we use early stopping then rollback to the best model found
		if best_params:
			self._restore_model_param(best_params)
		
		return self
	
	'''
	8. predict_proba: produce predicted probabilities for each instance for all classes/labels
	
	Note: we need to make sure that there is a session being built/trained, 
	      if there is no such session, we raise exception to report the error.
	      otherwise, we evaluate the 'score' within running a session.
	
	
	--- Inputs: 
			X: the features of test data
	--- Outputs:
			y_proba: the probabilities of each instance being each class in problem
	'''
	def predict_proba(self, X_test):
		if not self._session():
			raise NotFittedError('This %s instance is not fitted yet' %(self.__class__.__name__))
		else:
			with self._session.as_default():
				y_pred_proba = self._scores.eval(feed_dict = {self._X: X_test})
		return y_pred_proba
	
	'''
	9. predict: produce a concrete predicted class for an instance
	
	Note: the predicted values are indice, and we need to translate back to class labels
	
	--- Inputs: the features of test data
	--- Outputs: the predicted classes for test data
	'''
	def predict(self, X_test):
		y_pred_proba  = self.predict_proba(X_test)
		y_pred_indice = np.argmax(y_pred_proba, axis = 1)
		y_pred = np.array([self.class_to_index_[y_pred_index] for y_pred_index in y_pred_indice], dtype = np.int32)
		return y_pred
	
	'''
	10. save: to save the final model to a specified path
	'''
	def save(self, path):
		self._saver.save(self._session, path)
		
		
if __name__ == '__main__':
	mnist = input_data.read_data_sets('./dataset')
	X_train = mnist.train.images[mnist.train.labels < 5]
	y_train = mnist.train.labels[mnist.train.labels < 5]
	X_valid = mnist.validation.images[mnist.validation.labels < 5]
	y_valid = mnist.validation.labels[mnist.validation.labels < 5]
	X_test  = mnist.test.images[mnist.test.labels < 5]
	y_test  = mnist.test.labels[mnist.test.labels < 5]
	
	random_state = 42
	dnn_clf = DNNClassifier(random_state = random_state)
	dnn_clf.fit(X_train, y_train, X_valid = X_valid, y_valid = y_valid)
	print('ok')
	
	
	
	
	
	
	
	
	








































