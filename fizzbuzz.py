import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


class FizzBuzz():

	def __init__(self):
		# some parameters
		self.EMBEDDINGS_SIZE = 100
		self.HIDDEN_UNITS = 100
		self.DROPOUT = 0.75
		self.LEARNING_RATE = 0.0001
		self.NUM_LABELS = 4
		self.NUM_EPOCHS = 20
		self.BATCH_SIZE = 100
		self.LAMBDA = 0.0001

		# create data and model
		self._load_data()
		self._generate_graph()
	
	def _load_data(self):
		self.label2id = {"other":0, "fizz":1,"buzz":2, "fizz buzz":3}
		self.id2label = {i:j for (j,i) in self.label2id.items()}

		self.X = [[int(digit) for digit in number] for number in list(map(str,list(range(1,9999))))]
		self.lengths = np.asarray([len(x) for x in self.X], dtype=np.int32)
		self.max_length = max(self.lengths)
		self.X = np.asarray([np.concatenate((x, np.ones((self.max_length - len(x))) * 10)) for x in self.X], dtype=np.int32)
		self.y = np.asarray([self._vectorize_y(i) for i in range(1,9999)])

		self.train = (self.X[100:], self.y[100:], self.lengths[100:])
		self.test = (self.X[:100], self.y[:100], self.lengths[:100])
		return self

	def _vectorize_y(self, number):
		if number % 3 == 0 and number % 5 == 0:
			return self._create_one_hot(self.label2id["fizz buzz"])
		elif number % 3 == 0:
			return self._create_one_hot(self.label2id["fizz"])
		elif number % 5 == 0:
			return self._create_one_hot(self.label2id["buzz"])
		else:
			return self._create_one_hot(self.label2id["other"])

	def _create_one_hot(self, x):
		one_hot = np.zeros(self.NUM_LABELS)
		one_hot[x] = 1
		return one_hot

	def _encode_sentence(self, is_training):
		"""
		encode the input sequence by running LSTMs over all digits (=timesteps)
		returns encoded representation of the input sequence
		"""


		with tf.variable_scope("encode", reuse=tf.AUTO_REUSE):
			if is_training:
				cell_fw = tf.contrib.rnn.LSTMCell(self.HIDDEN_UNITS)
				cell_fw = tf.contrib.rnn.DropoutWrapper(cell_fw, output_keep_prob=self.DROPOUT)
				cell_bw = tf.contrib.rnn.LSTMCell(self.HIDDEN_UNITS)
				cell_bw = tf.contrib.rnn.DropoutWrapper(cell_bw, output_keep_prob=self.DROPOUT)
			else:
				cell_fw = tf.contrib.rnn.LSTMCell(self.HIDDEN_UNITS)
				cell_bw = tf.contrib.rnn.LSTMCell(self.HIDDEN_UNITS)

			# run the LSTM
			(_, _), (state1, state2) = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, self.embedded_inputs, sequence_length=self.sequence_lengths, dtype=tf.float32)
			# collect output
			encoded = tf.concat([state1.h, state2.h], axis=-1)
		return encoded

	def _classify(self, encoded, is_training):
		"""
		some classification layers on top
		input is encoded sequence, 
		returns logits
		"""
		with tf.variable_scope("classify", reuse=tf.AUTO_REUSE):
			hidden_layer = tf.layers.dense(inputs=encoded, kernel_initializer=tf.orthogonal_initializer, units=self.HIDDEN_UNITS, activation=tf.nn.relu)
			hidden_layer = tf.layers.dropout(hidden_layer, rate=1-self.DROPOUT, training=is_training)
			hidden_layer = tf.layers.dense(inputs=hidden_layer, kernel_initializer=tf.orthogonal_initializer, units=self.HIDDEN_UNITS, activation=tf.nn.relu)
			hidden_layer = tf.layers.dropout(hidden_layer, rate=1-self.DROPOUT, training=is_training)
			logits = tf.layers.dense(inputs=hidden_layer, units=self.NUM_LABELS)
		return logits



	def _generate_graph(self):

		# some placeholders
		self.inputs = tf.placeholder(tf.int32, shape=[None, None])
		self.labels = tf.placeholder(tf.int32, shape=[None, self.NUM_LABELS])
		self.sequence_lengths = tf.placeholder(tf.int32, shape=[None])

		# embedding lookup
		self.embeddings = tf.get_variable("embedding", [11, self.EMBEDDINGS_SIZE], initializer=tf.orthogonal_initializer, dtype=tf.float32, trainable=True)
		self.embedded_inputs = tf.nn.embedding_lookup(self.embeddings, self.inputs)

		# encode sequence
		self.encoded = self._encode_sentence(True)

		# perform classification on top
		self.logits = self._classify(self.encoded, True)	

		# training
		self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.labels))
		self.train_step = tf.train.AdamOptimizer().minimize(self.cost)

		# inference
		self.encoded_inference = self._encode_sentence(False)
		self.logits_inference = self._classify(self.encoded_inference, False)
		self.predictions = tf.cast(tf.argmax(tf.nn.softmax(self.logits_inference), axis=-1), tf.int32)
		self.accuracy = tf.reduce_mean(tf.where(tf.equal(tf.cast(tf.argmax(self.labels, axis=-1), tf.int32), self.predictions), x=tf.ones_like(self.predictions,dtype=tf.float32),y=tf.zeros_like(self.predictions, dtype=tf.float32)))

		return self

	def fit(self):
		init = tf.global_variables_initializer()
		with tf.Session() as sess:
			sess.run(init)
			for epoch in range(self.NUM_EPOCHS):
				# shuffle everything
				p = np.random.permutation(len(self.train[0]))
				self.train = ([array[p] for array in self.train])
				for batch in range(len(self.train[0]) // self.BATCH_SIZE):
					this_batch = self._generate_batch(batch)
					#print (this_batch[0][:3], this_batch[1][:3], this_batch[2][:3])
					sess.run(self.train_step, feed_dict={self.inputs:this_batch[0], self.labels:this_batch[1], self.sequence_lengths:this_batch[2]})

				print ("epoch %d, train accuracy:" % epoch, sess.run(self.accuracy, feed_dict={self.inputs:self.train[0], self.labels:self.train[1], self.sequence_lengths:self.train[2]}))
			predictions = sess.run(self.predictions, feed_dict={self.inputs:self.test[0], self.labels:self.test[1], self.sequence_lengths:self.test[2]})
			self.digits = sess.run(self.embeddings)
			return sess.run(self.accuracy, feed_dict={self.inputs:self.test[0], self.labels:self.test[1], self.sequence_lengths:self.test[2]}), predictions, self.digits
	def _generate_batch(self, batch):
		start, stop = batch * self.BATCH_SIZE, (batch + 1) * self.BATCH_SIZE
		return [array[start:stop] for array in self.train]


	"""
	def _normalize(inputs, epsilon = 1e-6):	
		mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)
		normalized = (inputs - mean) / ( (variance + epsilon) ** (.5) )
		outputs = normalized		
		return outputs
	def pca(self, A):


		# Compute the mean along each axes
		A = A[:-1]
		x_mean = np.mean(A[:, 0])
		y_mean = np.mean(A[:, 1])

		AC = np.copy(A)
		# Center data
		AC[:, 0] = AC[:, 0] - x_mean
		AC[:, 1] = AC[:, 1] - y_mean
		C = np.matmul(AC.T, AC)
		u, s, v = np.linalg.svd(C)
		E = np.matmul(AC, u[:2].T)

		fig, axes = plt.subplots()
		axes.axhline(0, ls='--', linewidth=0.5)
		axes.axvline(0, ls='--', linewidth=0.5)
		#axes.set_xlim(-1.0, 1.0)
		#axes.set_ylim(-1.0, 1.0)
		axes.scatter(E[:, 0], E[:, 1], s=1e2, label='digits')
		axes.set_title('digit embeddings')
		axes.legend()
		for i in range(10):
			axes.annotate(i, (E[:,0][i],E[:,1][i]))
		
		plt.show()
	"""

if __name__ == "__main__":
	fizzbuzz = FizzBuzz()
	test_acc, predictions, A = fizzbuzz.fit()
	print ("training finished, test acc:",test_acc)
	print ([fizzbuzz.id2label[label] if label != 0 else i + 1 for i,label in enumerate(predictions)])
	
	"""
	training finished, test acc: 1.0
	[1, 2, 'fizz', 4, 'buzz', 'fizz', 7, 8, 'fizz', 'buzz', 11, 'fizz', 13, 14, 'fizz buzz', 16, 17, 'fizz', 19, 'buzz', 'fizz', 22, 23, 'fizz', 'buzz', 26, 'fizz', 28, 29, 'fizz buzz', 31, 32, 		'fizz', 34, 'buzz', 'fizz', 37, 38, 'fizz', 'buzz', 41, 'fizz', 43, 44, 'fizz buzz', 46, 47, 'fizz', 49, 'buzz', 'fizz', 52, 53, 'fizz', 'buzz', 56, 'fizz', 58, 59, 'fizz buzz', 61, 62, 'fizz', 		64, 'buzz', 'fizz', 67, 68, 'fizz', 'buzz', 71, 'fizz', 73, 74, 'fizz buzz', 76, 77, 'fizz', 79, 'buzz', 'fizz', 82, 83, 'fizz', 'buzz', 86, 'fizz', 88, 89, 'fizz buzz', 91, 92, 'fizz', 94, 		'buzz', 'fizz', 97, 98, 'fizz', 'buzz']
	"""
	#fizzbuzz.pca(A)

