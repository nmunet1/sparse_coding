import bisect
import math
import os
import random
import struct as st

import h5py
import numpy as np

from itertools import compress
from numpy.random import RandomState, randint

from sklearn import decomposition as dcmp

from plotutils import plotFeatureArrays


class SparseCoding(object):
	'''
	Sparse coding parent class
	'''

	def __init__(self, imshape = (.4, .4), xmargin = .1, ymargin = .125, 
		rng = True):

		self.isempty = True
		self.sc_model = None

		if rng:
			self.rng = RandomState(randint(100000))
		else:
			self.rng = None

		self.X_train = np.asarray([])
		self.X_test = np.asarray([])
		self.X_train_pp = np.asarray([])
		self.X_test_pp = np.asarray([])
		self.x_shape = None

		# Initialize plotting parameters
		self.imshape = imshape
		self.xmargin = xmargin
		self.ymargin = ymargin

	def fit(self, completion = 2, alpha = .5, n_iter = 500, batch_size = 50):
		if self.X_train_pp.size == 0:
			return None

		dict_size = round(completion * self.X_train_pp.shape[1])

		self.sparse_model = dcmp.MiniBatchDictionaryLearning(n_components = dict_size, alpha = alpha,
			fit_algorithm = 'cd', n_iter = n_iter, random_state = self.rng, batch_size = batch_size)
		self.sparse_model.fit(X_train_pp)

	def reconstruct(self):
		pass

	def losses(self):
		pass

	def save(self):
		pass


class SpectrogramSC(SparseCoding):

	def __init__(self):
		SparseCoding.__init__(self, imheight = .8)

		self.spect_path = None
		self.feat_path = None
		self.label_path = None
		self.fnames_path = None

		self.pca_model = None
		self.category_names = []
		self.category_idxs = {}

		self.train_labels = []
		self.test_labels = []

	def readin(self, path):
		self.isempty = False

		# Set paths for data and metadata
		self.feat_path = os.path.join(path, 'acoustic_features_setA.h5')
		self.label_path = os.path.join(path, 'label_names')
		self.spect_path = os.path.join(path, 'sound_arrays.h5')
		self.fnames_path = os.path.join(path, 'spectrogram_file_names')

		# Filter out spectrograms that have inappropriate number of labels or contain inf/nan values
		with h5py.File(self.feat_path) as feat:
			label_idxs = np.where(np.sum(np.array(feat['labels']), axis = 1) == 1)[0]

		with h5py.File(self.spect_path) as spect:
			sound_idxs = np.where(np.all(np.isfinite(np.array(spect['spectograms'])), axis = (1,2)))[0]

		valid_idxs = np.intersect1d(label_idxs, sound_idxs, assume_unique = True)

		# Get list of sound category names
		self.category_names = []
		with open(self.label_path) as names:
			for line in names:
				self.category_names.append(line[:-1].strip)

		# Compile dict with indices of spectrograms organized by sound category
		self.category_idxs = {}
		with h5py.File(self.feat_path) as feat:
			sound_categories = features['labels']

			# Get array of indices for each category
			for cat_i, cat_name in enumerate(self.category_names):
				self.category_idxs[cat_name] = np.intersect1d(valid_idxs, 
					list(compress(range(valid_idxs.size), sound_categories[:,cat_i])),
					assume_unique = True)

		self.subsample()


	def subsample(self, categories = None, max_samples = 1000, train_prop = .8, seed = True):
		if self.isempty:
			return None

		if seed:
			random.seed(13)

		if categories is None:
			categories = self.category_idxs.keys()

		# (Sub)sample indices from each category, concatenate, and partition into training and test sets
		train_idxs = []
		test_idxs = []
		for category in categories:
			idxs = self.category_idxs[category]

			if idxs.size > max_samples:
				n_samples = max_samples
			else:
				n_samples = idxs.size

			samples = random.sample(list(idxs), n_samples)
			train_idxs += samples[:round(train_prop*n_samples)]
			test_idxs += samples[round(train_prop*n_samples):]

		train_idx.sort()
		test_idxs.sort()

		# Get training and test data arrays
		with h5py.File(self.spect_path) as spect:
			self.X_train = np.array(spect['spectograms'][train_idxs,:,:])
			self.X_test = np.array(spect['spectograms'][test_idxs,:,:])

		self.x_shape = self.X_train.shape[1:]

		# Get file name labels for training and test data
		self.train_labels = []
		self.test_labels = []
		with open(self.fnames_path) as fnames:
			i = 0
			for line in fnames:
				if i in train_idxs:
					self.train_labels.append(line.split('/')[-1].split('.')[0])
				elif i in test_idxs:
					self.test_labels.append(line.split('/')[-1].split('.')[0])

				i += 1

	def preprocess(self, fit = True, n_components = .98):
		if self.isempty:
			return None

		if self.pca_model is None:
			fit = True

		if fit:
			self.pca_model = dcmp.PCA(n_components = n_components, whiten = True, random_state = self.rng)
			self.X_train_pp = pca_model.fit_transform(self.X_train - np.mean(self.X_train,0))
		else:
			self.X_train_pp = pca_model.transform(self.X_train - np.mean(self.X_train,0))

		self.X_test_pp = pca_model.transform(self.X_test - np.mean(self.X_test,0))

	def plotData(self, dataset = 'Train'):
		if self.isempty:
			return None

		if dataset == 'Train':
			plotFeatureArrays(self.X_train, self.x_shape, tiled = True, 
				tiles_psn = (self.xmargin, self.ymargin)+self.imshape,
				xlims = (0, 100), ylims = (250, 10000),
				titles = self.train_labels, xlabel = 'Time (ms)', ylabel = 'Frequency (Hz)', 
				noise_floor = 50, extent = (0, 99, 0, 79952), origin = 'lower')
		elif dataset == 'Test':
			plotFeatureArrays(self.X_test, self.x_shape, tiled = True, 
				tiles_psn = (self.xmargin, self.ymargin)+self.imshape,
				xlims = (0, 100), ylims = (250, 10000),
				titles = self.test_labels, xlabel = 'Time (ms)', ylabel = 'Frequency (Hz)', 
				noise_floor = 50, extent = (0, 99, 0, 79952), origin = 'lower')

	def plotDictionary(self):
		if self.sc_model is None:
			return None

		components = self.pca_model.inverse_transform(self.sparse_model.components_)

		plotFeatureArrays(components, self.x_shape, tiled = True, 
			tiles_psn = (self.xmargin, self.ymargin) + self.imshape,
			xlims = (0, 100), ylims = (250, 10000),
			xlabel = 'Time (ms)', ylabel = 'Frequency (Hz)', 
			extent = (0, 99, 0, 79952), origin = 'lower')


class NaturalImageSC(SparseCoding):

	def __init__(self):
		SparseCoding.__init__(self)