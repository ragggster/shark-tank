
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
from unpickle import unpickle


SEASON = 4
DOWNSAMPLE = 1000
EXTRACT = False


RNN_Units = 5
LR = 0.0001
NUM_EPOCHS = 3000
REG = 0.000001
VAL_SPLIT = 0.1
SOFTMAX = True
RUN_ON_FINAL_RNN_STATE = True

DATA_DIR = './audio-scraping/season%s-pitches' %(SEASON)
MFCC_DIR = './data/mfcc' 
META_FILE = './audio-scraping/season%s-pitches-metadata.p' % (SEASON)
LABELS_DIR = './data/labels/'

def get_files_in_dir(directory):
	# http://stackoverflow.com/questions/3207219/how-to-list-all-files-of-a-directory
	return [f for f in listdir(directory) if (isfile(join(directory, f)) and f[0] != '.')]

def get_label(line):
	potential_labels = line.split(' ')[0] #ALTER THE INDEX TO CHANGE WHAT YOU ARE PREDICTING!!!!
	return float(potential_labels)

class MFCC_Extractor():
	def __init__(self, input_dir):
		self.input_fns = get_files_in_dir(input_dir)

	def write_features(self, meta_fn, output_dir):
		# Write features as csv WITH headers so we can pick which to classify later
		# load data
		if not exists(output_dir):
			os.makedirs(output_dir)

		for pitch_audio_fn in self.input_fns:
			
			full_file = join(DATA_DIR, pitch_audio_fn)
			open_wav = Sndfile(full_file)
			rate = open_wav.samplerate*DOWNSAMPLE
			
			sig = open_wav.read_frames(open_wav.nframes)

			# open_wav  = wave.open(full_file, 'r')
			# rate = open_wav.getframerate()
			# sig = open_wav.readframes(open_wav.getnframes())
			#(rate,sig) = wav.read(join(DATA_DIR, pitch_audio_fn))
			mfcc_features = python_speech_features.mfcc(sig, rate)
			print(np.prod(mfcc_features.shape))
			with open(join(output_dir, pitch_audio_fn.split('.')[0]), 'w')	as output_fn:
				np.savetxt(output_fn, mfcc_features, delimiter= ',')

			print "Extracted MFCC features for %s into a csv" %(pitch_audio_fn)

class Baseline():
	def __init__(self):
		pass


	def setup_graph(self, max_time, features):
		self.X_placeholder = tf.placeholder(tf.float32, (None, max_time, features))
		self.y_placeholder = tf.placeholder(tf.float32, (None, ))
		self.seq_lens_placeholder = tf.placeholder(tf.float32, (None, ))
		
		rnn_cell = tf.contrib.rnn.BasicRNNCell(RNN_Units, activation = tf.nn.relu) #COULD TRY OTHER CELLS

		rnn_outputs, final_state = tf.nn.dynamic_rnn(rnn_cell, self.X_placeholder, self.seq_lens_placeholder, dtype=tf.float32)

		if(RUN_ON_FINAL_RNN_STATE):
			final_inputs = final_state
		else:
			final_inputs = tf.contrib.layers.flatten(rnn_outputs)

		if SOFTMAX:
			self.outputs = tf.contrib.layers.fully_connected(final_inputs, num_outputs = 2, activation_fn = None, biases_initializer = tf.zeros_initializer())
			self.unreg_loss = tf.losses.sparse_softmax_cross_entropy(tf.cast(self.y_placeholder, tf.int32), self.outputs)
		else:
			self.outputs = tf.contrib.layers.fully_connected(final_inputs, num_outputs = 1, activation_fn = None, biases_initializer = tf.zeros_initializer())
			self.unreg_loss = tf.losses.mean_squared_error(self.y_placeholder, tf.squeeze(self.outputs)) #PLAY AROUND WITH

		

		self.l2_loss = REG*tf.reduce_sum([tf.nn.l2_loss(v) for v in tf.trainable_variables()])
		self.loss = self.unreg_loss #+ self.l2_loss

		optimizer = tf.train.AdamOptimizer(LR)
		#optimizer = tf.train.GradientDescentOptimizer(LR)
		self.train_step = optimizer.minimize(self.loss)


	def get_batch(self, X, y, seq_lens, batch=True):
		total_examples = len(seq_lens)	
		if not batch:
			yield (X, y, seq_lens)
		yield (X[0:total_examples//2], y[0:total_examples//2], seq_lens[0:total_examples//2])
		yield (X[total_examples//2:], y[total_examples//2:], seq_lens[total_examples//2:])

	def prepare(self, features_dir, labels_dir):
		labels = get_files_in_dir(labels_dir)
		#features = get_files_in_dir(features_dir)
		num_fns = None
		X_unpadded, y = [], []
		for label_fn in labels:
			try:
				with open(join(features_dir, label_fn.split('.')[0] )) as ffn:
					data = np.loadtxt(ffn, delimiter=',')
					assert(data.shape[1] == 13)
					num_fns = data.shape[1]
					X_unpadded.append(data.flatten())
				with open(join(labels_dir, label_fn)) as lfn:
					y.append(get_label(lfn.readline()))
			except IOError:
				continue

		N = len(y)
		y = np.array(y)
		seq_lens = map(len, X_unpadded)
		max_X = max(seq_lens)
		X = np.zeros((N, max_X))

		for n in range(N):
			x_len = seq_lens[n]
			X[n, 0:x_len] += np.array(X_unpadded[n])
		X = X.reshape((N, -1, num_fns))
		assert(X.shape[1] == max_X/num_fns)

		self.setup_graph(max_X/num_fns, num_fns)
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


	def run_baseline(self, features_dir, labels_dir):
		X, y, seq_lens = self.prepare(features_dir, labels_dir)
		X_train, X_val, y_train, y_val, seq_lens_train, seq_lens_val = self.split_data(X, y, seq_lens)


		with tf.Session() as session:
			session.run(tf.global_variables_initializer())
			for epoch in range(NUM_EPOCHS):
				for batch in self.get_batch(X_train, y_train, seq_lens_train):
					X_batch, y_batch, seq_lens_batch = batch
					train_feed_dict = {self.X_placeholder: X_batch, self.y_placeholder : y_batch, self.seq_lens_placeholder : seq_lens_batch}

					_, l2_loss, unreg_loss = session.run([self.train_step, self.l2_loss, self.unreg_loss], train_feed_dict)
					total_loss = l2_loss + unreg_loss

					val_feed_dict = {self.X_placeholder: X_val, self.seq_lens_placeholder: seq_lens_val, self.y_placeholder : y_val}
					val_unreg_loss, val_loss, output = session.run([self.unreg_loss, self.loss, tf.nn.softmax(self.outputs)], val_feed_dict)
					#print epoch, 'test_loss', val_loss, '(unreg:', val_unreg_loss, ')', '\t--train_loss: ', total_loss, '(', "unreg:", unreg_loss, ')'
					print val_unreg_loss, zip(output[0:5], y_val[0:5])

if __name__ == '__main__':
	
	if(EXTRACT):
		MFCC_Extractor(DATA_DIR).write_features(META_FILE, MFCC_DIR)

	baseline = Baseline()
	baseline.run_baseline(MFCC_DIR, LABELS_DIR)