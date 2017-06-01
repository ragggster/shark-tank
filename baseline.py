
import python_speech_features
import tensorflow as tf 
import numpy as np
import scipy.io.wavfile as wav
import os
from os import listdir
from os.path import isfile, join, exists
import wave
from scikits.audiolab import Sndfile
from scikits import audiolab

import dill as pickle
from feature_extractor import *
from tqdm import tqdm



LR = 0.0005
NUM_EPOCHS = 3000
REG = 0.0001
VAL_SPLIT = 0.15

#RNN Params
RNN_Units = 50
RUN_ON_FINAL_RNN_STATE = True
SOFTMAX = False
LSTM = False
BATCH_NORM = True

LIMIT_DATA_POINTS = 1500

BATCHES = LIMIT_DATA_POINTS//10


ADD_LAYER = False
ADD_LAYER_U = 10

#CNN PARAMS
L1_FILTER_PARAMS = {'filters' : 64, 'kernel_size': 6, 'strides': 2}
L2_FILTER_PARAMS = {'filters' : 32, 'kernel_size': 4, 'strides': 2}
L3_FILTER_PARAMS = {'filters' : 16, 'kernel_size': 3, 'strides': 1}


LABELS_FN = './data/labels.p'

CLASSIFY = 'label_code' #needs to be a key in the labels dictionaries


# def get_label(line):
# 	potential_labels = line.split(' ')[0] #ALTER THE INDEX TO CHANGE WHAT YOU ARE PREDICTING!!!!
# 	return float(potential_labels)
def get_label(labels, fn):
	name = fn.split('.')[0]
	entry = labels[name + '.wav']
	return entry[CLASSIFY]


class Baseline():
	def __init__(self):
		pass

	def setup(self):
		self.setup_input()
		# self.setup_hybrid_graph()
		# self.setup_cnn_graph()
		self.setup_rnn_graph()
		self.setup_loss_and_train()

	def setup_input(self):
		self.X_placeholder = tf.placeholder(tf.float32, (None, self.max_time, self.n_features))
		self.y_placeholder = tf.placeholder(tf.float32, (None, ))
		self.seq_lens_placeholder = tf.placeholder(tf.int32, (None, ))


	def setup_rnn(self, X, seq_lens = None):
		if LSTM:
			cell = tf.contrib.rnn.LSTMCell(RNN_Units, activation = tf.nn.relu, cell_clip= 50.0) #COULD TRY OTHER CELLS
		else:
			cell = tf.contrib.rnn.BasicRNNCell(RNN_Units, activation = tf.nn.relu) #COULD TRY OTHER CELLS

		rnn_outputs, final_state = tf.nn.dynamic_rnn(cell, X, seq_lens, dtype=tf.float32)
		
		if RUN_ON_FINAL_RNN_STATE:
			if LSTM:
				final_outputs = final_state.h #final hidden state
			else:
				final_outputs = final_state
		else:
			final_outputs = tf.contrib.layers.flatten(rnn_outputs)
		return final_outputs


	'''
	Sets up a final affine layer that feeds into either a softmax loss or a binary loss
	'''
	def final_affine_layer_and_loss(self, inputs):
		if SOFTMAX:
			self.outputs = tf.squeeze(tf.contrib.layers.fully_connected(inputs, num_outputs = 2, activation_fn = None, biases_initializer = tf.zeros_initializer()))
			
			degenerate_score = tf.nn.sparse_softmax_cross_entropy_with_logits(labels = tf.cast(self.y_placeholder, tf.int32), logits= 0.5*tf.ones_like(self.outputs))
			unreg_losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels = tf.cast(self.y_placeholder, tf.int32), logits = self.outputs)
			self.unreg_loss = tf.reduce_mean(unreg_losses)
			self.advantage = tf.reduce_mean(degenerate_score - unreg_losses)
		else:
			self.outputs = tf.squeeze(tf.contrib.layers.fully_connected(inputs, num_outputs = 1, activation_fn = None, biases_initializer = tf.zeros_initializer()))

			# degenerate_score = tf.losses.mean_squared_error(self.y_placeholder, 0.5*tf.ones_like(self.y_placeholder))
			# self.unreg_losses = tf.losses.mean_squared_error(self.y_placeholder, self.outputs) #PLAY AROUND WITH
			
			adjusted_y = (self.y_placeholder-0.5)*2.0 #[0, 1] => [-1, 1]
			degenerate_score = tf.losses.hinge_loss(self.y_placeholder, tf.zeros_like(self.outputs))
			unreg_losses = tf.losses.hinge_loss(self.y_placeholder, self.outputs) #PLAY AROUND WITH
			
			self.unreg_loss = tf.reduce_mean(unreg_losses)
			self.advantage = tf.reduce_mean(degenerate_score - unreg_losses)

	#####################################
	# 			GRAPH SETUP$ 			#
	# These can be anything that uses 	#
	# the placeholders and outputs:  	#
	# self.unreg_loss(unregularized)	#
	# self.outputs (the results)		#
	# IF these two things are given,	#
	# most other methods should continue#
	# to work or require minimal 		#
	# modification						#

	def setup_cnn_graph(self):
		print self.X_placeholder.get_shape().as_list()
		conv1 = tf.layers.conv1d(self.X_placeholder,filters = L1_FILTER_PARAMS['filters'],
		 											kernel_size=L1_FILTER_PARAMS['kernel_size'],
		 											strides=L1_FILTER_PARAMS['strides'],
		 											activation = tf.nn.relu)
		

		print 'c1', conv1.get_shape().as_list()
		if BATCH_NORM:
			conv1 = tf.layers.batch_normalization(conv1)
		conv2 = tf.layers.conv1d(conv1, 			filters = L2_FILTER_PARAMS['filters'],
		 											kernel_size=L2_FILTER_PARAMS['kernel_size'],
		 											strides=L2_FILTER_PARAMS['strides'],
		 											activation = tf.nn.relu)
		print 'c2', conv2.get_shape().as_list()
		if BATCH_NORM:
			conv2 = tf.layers.batch_normalization(conv1)
		conv3 = tf.layers.conv1d(conv2,	 			filters = L3_FILTER_PARAMS['filters'],
		 											kernel_size=L3_FILTER_PARAMS['kernel_size'],
		 											strides=L3_FILTER_PARAMS['strides'],
		 											activation = tf.nn.relu)#*(L3_FILTER_PARAMS.values()), use_bias = True)

		print 'c3', conv3.get_shape().as_list()

		
		self.final_affine_layer_and_loss(tf.contrib.layers.flatten(conv3))


	def setup_hybrid_graph(self):
		conv1 = tf.layers.conv1d(self.X_placeholder, filters = L1_FILTER_PARAMS['filters'],
		 											kernel_size=L1_FILTER_PARAMS['kernel_size'],
		 											strides=L1_FILTER_PARAMS['strides'],
		 											activation = tf.nn.relu)
		

		print 'c1', conv1.get_shape().as_list()
		conv2 = tf.layers.conv1d(conv1, filters = L2_FILTER_PARAMS['filters'],
		 											kernel_size=L2_FILTER_PARAMS['kernel_size'],
		 											strides=L2_FILTER_PARAMS['strides'],
		 											activation = tf.nn.relu)
		
		rnn_outputs = self.setup_rnn(conv2)
		self.final_affine_layer_and_loss(rnn_outputs)



	def setup_rnn_graph(self):
		rnn_outputs = self.setup_rnn(self.X_placeholder, self.seq_lens_placeholder)
		if BATCH_NORM:
			rnn_outputs = tf.layers.batch_normalization(rnn_outputs)
		self.to_check = rnn_outputs
		if ADD_LAYER:
			rnn_outputs = tf.layers.dense(rnn_outputs, ADD_LAYER_U, activation = tf.nn.relu)
		self.final_affine_layer_and_loss(rnn_outputs)

	
		
		
	########## END SETUPS ###############
	#####################################

	def setup_loss_and_train(self):
		self.l2_loss = REG*tf.reduce_sum([tf.nn.l2_loss(v) for v in tf.trainable_variables()])
		self.loss = self.unreg_loss + self.l2_loss
		global_step = tf.Variable(0, trainable=False)
		decaying_lr = tf.train.exponential_decay(LR, global_step, 1000000, 0.95, staircase=True)
		optimizer = tf.train.AdamOptimizer(decaying_lr)
		self.train_step = optimizer.minimize(self.loss, global_step=global_step)
		print "\n----\n%i variables total\n----\n"%(sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()]))

	def get_batch(self, X, y, seq_lens, batch=True, batches = BATCHES):
		total_examples = len(seq_lens)	
		if not batch:
			yield (X, y, seq_lens)
		batch_len = total_examples//batches
		divide = 0
		while divide < total_examples:
			next_divide = min([divide+batch_len, total_examples])
			yield (X[divide:next_divide], y[divide:next_divide], seq_lens[divide:next_divide])
			divide = next_divide

	'''
	Iterates over all of the files present in 
	features_dir and attempts to load them as numpy files.
	Each file corresponds to one data point which is matched up 
	by file name with a label and these two are then joined together.
	'''
	def prepare(self, features_dir, labels_fn):
		feature_files = np.random.permutation(get_files_in_dir(features_dir))
		#features = get_files_in_dir(features_dir)
		num_features = None
		with open(labels_fn) as lf:
			labels = pickle.load(lf)

		X_unpadded, y = [], []
		print 'loading data'


		for i, feature_fn in enumerate(tqdm(feature_files)):
			if i > LIMIT_DATA_POINTS:
				break
			try:
				y.append(get_label(labels, feature_fn))
				with open(join(features_dir, feature_fn)) as ffn:
					data = np.loadtxt(ffn, delimiter=',')
					# print 'data loaded w/ shape:', data.shape
					
					if num_features is None:
						num_features = data.shape[1]
					assert(num_features == data.shape[1])

					X_unpadded.append(data.flatten())
			except IOError as e:
				print "ERROR", e
				continue


		# This part of the code reshapes X into a temporal series 
		# after padding (it was flattened previously to more easily pad)
		print 'formating data'
		N = len(y)
		y = np.array(y)
		seq_lens = map(len, X_unpadded)
		max_X = max(seq_lens)
		X = np.zeros((N, max_X))
		for n in range(N):
			x_len = seq_lens[n]
			X[n, 0:x_len] += np.array(X_unpadded[n])
		X = X.reshape((N, -1, num_features))
		assert(X.shape[1] == max_X/num_features)

		self.max_time = max_X/num_features
		self.n_features = num_features
		return X, y, seq_lens

	def split_data(self, X, y, seq_lens):
		seq_lens = np.array(seq_lens)
		N = len(seq_lens)
		N_val = int(VAL_SPLIT*N)
		val_r = np.random.choice(range(N), (N_val,))
		train_r = set(range(N)) - set(val_r)
		val_r = np.array(val_r)
		train_r = np.array(list(train_r), dtype = np.int)
		X_tr = X[train_r]
		X_val = X[val_r]
		y_tr = y[train_r]
		y_val = y[val_r]
		sq_tr = seq_lens[train_r]
		sq_val = seq_lens[val_r]
		return X_tr, X_val, y_tr, y_val, sq_tr, sq_val

	def get_accuracy(self, Y, pred):
		correct = []
		for i, y in enumerate(Y):
			
			if SOFTMAX:
				pred_i = np.argmax(pred, axis=1)[i]
			else:
				pred_i = round(max([0, min([1, pred[i]])]))
			
			if pred_i == y:
				correct.append(1)
			else:
				correct.append(0)

		return sum(correct)*1.0/len(correct)

	

	def center_on_train_data(self, X_train, X_val):
		n_features = X_train.shape[2]
		means = np.mean(np.reshape(X_train, (-1, n_features)), axis = 0)
		return X_train - means, X_val - means

	def run_baseline(self, features_dir, labels_file):
		X, y, seq_lens = self.prepare(features_dir, labels_file)
		self.setup()

		X_train, X_val, y_train, y_val, seq_lens_train, seq_lens_val = self.split_data(X, y, seq_lens)
		X_train, X_val = self.center_on_train_data(X_train, X_val)
		with tf.Session() as session:
			session.run(tf.global_variables_initializer())
			for epoch in range(NUM_EPOCHS):
				for batch in self.get_batch(X_train, y_train, seq_lens_train):
					X_batch, y_batch, seq_lens_batch = batch
					
					train_feed_dict = {self.X_placeholder: X_batch, self.y_placeholder : y_batch, self.seq_lens_placeholder : seq_lens_batch}

					_, l2_loss, unreg_loss  = session.run([self.train_step, self.l2_loss, self.unreg_loss], train_feed_dict)
					

					total_loss = l2_loss + unreg_loss

					val_feed_dict = {self.X_placeholder: X_val, self.seq_lens_placeholder: seq_lens_val, self.y_placeholder : y_val}
					adv, val_loss, output = session.run([self.advantage, self.loss, self.outputs], val_feed_dict)

					accuracy = self.get_accuracy(y_val, output)
					print "accuracy: :", accuracy

					#Advantage is the performance above a degenerate algorithm (predict one class every time)
					#print epoch, 'val_loss', val_loss, '(unreg:', val_unreg_loss, ')', ' -- train_loss: ', total_loss, '(', "unreg:", unreg_loss, ')'
					print adv ,unreg_loss, zip(output[0:7], y_val[0:7])

if __name__ == '__main__':
	baseline = Baseline()
	baseline.run_baseline(MFCC_DIR, LABELS_FN)

