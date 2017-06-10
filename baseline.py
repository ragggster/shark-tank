
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

SAVE_FILE = 'results_meta_init_state_w_pooling_PReLU.csv'


LR = 0.0001
NUM_EPOCHS = 3000
REG = 0.0001
VAL_SPLIT = 0.2

#RNN Params
RNN_Units = 80
RUN_ON_FINAL_RNN_STATE = True
SOFTMAX = False
LSTM = True
BATCH_NORM = True

FEED_META_INTO_RNN = True

LIMIT_DATA_POINTS = 7894

BATCHES = LIMIT_DATA_POINTS//50

ADD_LAYER = False
ADD_LAYER_U = 10

#CNN PARAMS
L1_FILTER_PARAMS = {'filters' : 64, 'kernel_size': 6, 'strides': 3}
# L1_FILTER_PARAMS = {'filters' : 16, 'kernel_size': 6, 'strides': 3}
L2_FILTER_PARAMS = {'filters' : 32, 'kernel_size': 4, 'strides': 2}
L3_FILTER_PARAMS = {'filters' : 16, 'kernel_size': 3, 'strides': 1}
# L3_FILTER_PARAMS = {'filters' : 64, 'kernel_size': 3, 'strides': 1}

LABELS_FN = './data/labels.p'

CLASSIFY = 'label_code' #needs to be a key in the labels dictionaries


# def get_label(line):
# 	potential_labels = line.split(' ')[0] #ALTER THE INDEX TO CHANGE WHAT YOU ARE PREDICTING!!!!
# 	return float(potential_labels)
def get_label(labels, fn):
	name = fn.split('.')[0]
	entry = labels[name + '.wav']
	return entry[CLASSIFY]


industry_categories = ['Food and Beverage',
'Fashion / Beauty',
'Lifestyle / Home',
'Children / Education',
'Fitness / Sports / Outdoors',
'Software / Tech',
'Healthcare',
'Pet Products',
'Uncertain / Other',
'Business Services',
'Media / Entertainment',
'Travel',
'Green/CleanTech',
'Automotive',
'Fitness / Sports',
'Consumer Products']

def get_metainfo(labels, fn):
	name = fn.split('.')[0]
	entry = labels[name + '.wav']
	if entry['gender'] == 'Female':
		gender = [1.0, 0, 0]
	elif entry['gender'] == "Male":
		gender = [0, 1.0, 0]
	else:
		gender = [0, 0, 1.0]
	industry = np.zeros((len(industry_categories),))
	industry[industry_categories.index(entry['industry'])] = 1.0
	# gender = (-1 if entry['gender'] == 'Female' else 1)
	return np.concatenate([gender, industry])


class Baseline():
	def __init__(self, features_dir, labels_file):
		X, y, meta, seq_lens = self.prepare(features_dir, labels_file)
		self.n_meta_features = meta.shape[1]
		self.X_train, self.X_val, self.y_train, self.y_val, self.seq_lens_train, self.seq_lens_val, self.meta_train, self.meta_val = self.split_data(X, y, seq_lens,meta)
		self.X_train, self.X_val = self.center_on_train_data(self.X_train, self.X_val)

	def setup(self):
		self.setup_input()
		self.setup_hybrid_graph()
		# self.setup_cnn_graph()
		# self.setup_rnn_graph()
		self.setup_loss_and_train()

	def run_baseline(self):
		
		self.setup()

		with open(SAVE_FILE, 'w') as output_f:
			with tf.Session() as session:
				session.run(tf.global_variables_initializer())
				for epoch in range(NUM_EPOCHS):
					for i, batch in enumerate(self.get_batch(self.X_train, self.y_train, self.seq_lens_train, self.meta_train)):
						X_batch, y_batch, seq_lens_batch, meta_batch = batch
						
						train_feed_dict = {self.X_placeholder: X_batch, self.y_placeholder : y_batch, self.seq_lens_placeholder : seq_lens_batch, self.meta_placeholder: meta_batch}

						_, l2_loss, unreg_loss, dec_lr  = session.run([self.train_step, self.l2_loss, self.unreg_loss, self.decaying_lr], train_feed_dict)
						print 'lr:', dec_lr

						total_loss = l2_loss + unreg_loss

						val_feed_dict = {self.X_placeholder: self.X_val, self.seq_lens_placeholder: self.seq_lens_val, self.y_placeholder : self.y_val,  self.meta_placeholder: self.meta_val}
						adv, val_loss, output = session.run([self.advantage, self.loss, self.outputs], val_feed_dict)

						accuracy = self.get_accuracy(self.y_val, output)
						print "accuracy: %f \t degenerate accuracy: %f" %(accuracy, self.get_degenerate_accuracy(self.y_val, output) )
						output_f.write('%f, %f, %f\n' %(accuracy, val_loss, unreg_loss))
						
						#Advantage is the performance above a degenerate algorithm (predict one class every time)
						#print epoch, 'val_loss', val_loss, '(unreg:', val_unreg_loss, ')', ' -- train_loss: ', total_loss, '(', "unreg:", unreg_loss, ')'
						print adv ,unreg_loss, zip(output[0:7], self.y_val[0:7])
						if i % 50 == 0:
							output_f.flush()
					output_f.flush()
	def get_degenerate_accuracy(self, truth, output):
		if min(truth) == 0:
			return max([self.get_accuracy(truth, np.ones_like(output)), self.get_accuracy(truth, np.zeros_like(output))])
		if min(truth) == -1:
			return max([self.get_accuracy(truth, np.ones_like(output)), self.get_accuracy(truth, np.zeros_like(output) -1)])
		else:
			print ("wtf is going on...")

	def prepare(self, features_dir, labels_fn):
		feature_files = np.random.permutation(get_files_in_dir(features_dir))
		#features = get_files_in_dir(features_dir)
		num_features = None
		with open(labels_fn) as lf:
			labels = pickle.load(lf)

		X_unpadded, y, meta = [], [], []
		print 'loading data'

		for i, feature_fn in enumerate(tqdm(feature_files)):
			if i > LIMIT_DATA_POINTS:
				break
			try:
				meta.append(get_metainfo(labels, feature_fn))
				y.append(get_label(labels, feature_fn))
				with open(join(features_dir, feature_fn)) as ffn:
					data = np.loadtxt(ffn, delimiter=',')
					# print 'data loaded w/ shape:', data.shape
					
					if num_features is None:
						num_features = data.shape[1] #should always be 13
					assert(num_features == data.shape[1])

					X_unpadded.append(data.flatten())
			except IOError as e:
				print "\n\nERROR", e, "\n\n"
				continue
			except KeyError as e:
				print "\n\nnon-existent entry:", e, "\n\n"
				continue

		# This part of the code reshapes X into a temporal series 
		# after padding (it was flattened previously to more easily pad)
		print 'formating data'
		N = len(y)
		y = np.array(y)
		meta = np.array(meta)
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
		return X, y, meta, seq_lens

	

	def setup_input(self):
		self.X_placeholder = tf.placeholder(tf.float32, (None, self.max_time, self.n_features))
		self.y_placeholder = tf.placeholder(tf.float32, (None, ))
		self.meta_placeholder = tf.placeholder(tf.float32, (None, self.n_meta_features))
		self.seq_lens_placeholder = tf.placeholder(tf.int32, (None, ))


	def setup_rnn(self, X, seq_lens = None):
		if LSTM:
			cell = tf.contrib.rnn.LSTMCell(RNN_Units, activation = tf.nn.tanh, cell_clip= 50000.0) 
		else:
			cell = tf.contrib.rnn.BasicRNNCell(RNN_Units, activation = tf.nn.relu) 
		if FEED_META_INTO_RNN:
			
			init_inputs = tf.layers.dense(self.meta_placeholder, 2*RNN_Units)
			if LSTM:
				c, h = tf.split(init_inputs, 2, axis  = 1)
				init_state = tf.contrib.rnn.LSTMStateTuple(c = c, h = h)
			else:
				init_state = init_inputs
			rnn_outputs, final_state = tf.nn.dynamic_rnn(cell, X, seq_lens, initial_state = init_state, dtype=tf.float32)
		else:
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
			vanilla_outputs = tf.squeeze(tf.contrib.layers.fully_connected(inputs, num_outputs = 2, activation_fn = None, biases_initializer = tf.zeros_initializer()))
			meta_layer = 0*tf.layers.dense(self.meta_placeholder, 2)
			
			self.outputs = tf.layers.dense(tf.concat([vanilla_outputs, meta_layer], axis = -1), 2)


			degenerate_score = tf.nn.sparse_softmax_cross_entropy_with_logits(labels = tf.cast(self.y_placeholder, tf.int32), logits= 0.5*tf.ones_like(self.outputs))
			unreg_losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels = tf.cast(self.y_placeholder, tf.int32), logits = self.outputs)
			self.unreg_loss = tf.reduce_mean(unreg_losses)
			self.advantage = tf.reduce_mean(degenerate_score - unreg_losses)
		else:
			vanilla_outputs = tf.contrib.layers.fully_connected(inputs, num_outputs = 1, activation_fn = None, biases_initializer = tf.zeros_initializer())
			meta_layer = 0.0*tf.layers.dense(self.meta_placeholder, 1) #zeroed out as we no longer use it

			self.outputs = tf.squeeze(tf.layers.dense(tf.concat([vanilla_outputs, meta_layer], axis = -1), 1))
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
		conv1 = tf.layers.max_pooling1d(conv1, 4, 2)

		print 'c1', conv1.get_shape().as_list()
		if BATCH_NORM:
			conv1 = tf.layers.batch_normalization(conv1)
		conv2 = tf.layers.conv1d(conv1, 			filters = L2_FILTER_PARAMS['filters'],
		 											kernel_size=L2_FILTER_PARAMS['kernel_size'],
		 											strides=L2_FILTER_PARAMS['strides'],
		 											activation = tf.nn.relu)

		conv2 = tf.layers.max_pooling1d(conv2, 2, 2)
		if BATCH_NORM:
			conv2 = tf.layers.batch_normalization(conv1)
		conv3 = tf.layers.conv1d(conv2,	 			filters = L3_FILTER_PARAMS['filters'],
		 											kernel_size=L3_FILTER_PARAMS['kernel_size'],
		 											strides=L3_FILTER_PARAMS['strides'],
		 											activation = tf.nn.relu)#*(L3_FILTER_PARAMS.values()), use_bias = True)

		print 'c3', conv3.get_shape().as_list()

		
		self.final_affine_layer_and_loss(tf.contrib.layers.flatten(conv3))


	
	def setup_hybrid_graph(self):
 		class PReLU():
			def __init__(self):
				self.P = tf.Variable(0.001)

 
			def __call__(self, x):
				self.P = tf.clip_by_value(self.P, 0.00001, 0.2)
				return tf.maximum(self.P*x, x)



		conv1 = tf.layers.conv1d(self.X_placeholder, filters = L1_FILTER_PARAMS['filters'],
		 											kernel_size=L1_FILTER_PARAMS['kernel_size'],
		 											strides=L1_FILTER_PARAMS['strides'],
		 											activation = tf.nn.relu)
		
		conv1 = tf.layers.max_pooling1d(conv1, 2, 2)
		if BATCH_NORM:
			conv1 = tf.layers.batch_normalization(conv1)
		conv2 = tf.layers.conv1d(conv1, filters = L2_FILTER_PARAMS['filters'],
		 											kernel_size=L2_FILTER_PARAMS['kernel_size'],
		 											strides=L2_FILTER_PARAMS['strides'],
		 											activation = tf.nn.relu)
		
		conv2 = tf.layers.max_pooling1d(conv2, 2, 2)
		if BATCH_NORM:
			conv2 = tf.layers.batch_normalization(conv2)
		conv3 = tf.layers.conv1d(conv2,	 			filters = L3_FILTER_PARAMS['filters'],
		 											kernel_size=L3_FILTER_PARAMS['kernel_size'],
		 											strides=L3_FILTER_PARAMS['strides'],
		 											activation = tf.nn.relu)#*(L3_FILTER_PARAMS.values()), use_bias = True)
		if BATCH_NORM:
			conv3 = tf.layers.batch_normalization(conv3)
		rnn_outputs = self.setup_rnn(conv3)
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
		self.decaying_lr = tf.train.exponential_decay(LR, global_step, 500, 0.97, staircase=True)
		optimizer = tf.train.AdamOptimizer(self.decaying_lr)
		self.train_step = optimizer.minimize(self.loss, global_step=global_step)
		print "\n----\n%i variables total\n----\n"%(sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()]))

	def get_batch(self, X, y, seq_lens, meta, batch=True, batches = BATCHES):
		total_examples = len(seq_lens)	
		if not batch:
			yield (X, y, seq_lens)
		batch_len = total_examples//batches
		divide = 0
		while divide < total_examples:
			next_divide = min([divide+batch_len, total_examples])
			yield (X[divide:next_divide], y[divide:next_divide], seq_lens[divide:next_divide], meta[divide:next_divide])
			divide = next_divide

	'''
	Iterates over all of the files present in 
	features_dir and attempts to load them as numpy files.
	Each file corresponds to one data point which is matched up 
	by file name with a label and these two are then joined together.
	'''
	

	def split_data(self, X, y, seq_lens, meta):
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

		meta_tr = meta[train_r]
		meta_val = meta[val_r]
		sq_tr = seq_lens[train_r]
		sq_val = seq_lens[val_r]
		return X_tr, X_val, y_tr, y_val, sq_tr, sq_val, meta_tr, meta_val

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

	

if __name__ == '__main__':
	baseline = Baseline(MFCC_DIR, LABELS_FN)
	baseline.run_baseline()

# 