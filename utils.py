import random
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

def plotFeatureArrays(featureVecs, array_shape, n_plots = (5, 5), tile_pad = None, 
	aspect = None, xlims = None, ylims = None, xlabel = None, ylabel = None, titles = None, 
	noise_floor = None, extent = None, origin = None, cbar = False, seed_id = None):
	
	n_samples = featureVecs.shape[0] # Total number of feature vectors
	sample_titles = []

	if tile_pad is None:
		tile_pad = (0.1, 0.1)

		if titles is not None:
			tile_pad[1] += 0.15

		if cbar:
			tile_pad[0] += .4
			tile_pad[1] += .05

	# Set random number generator seed
	if seed_id is None:
		random.seed(seed_id)

	if cbar:
		cbar_mode = 'each'
		cbar_pad = .03
	else:
		cbar_mode = None
		cbar_pad = None

	if type(n_plots) is tuple and len(n_plots) == 2:
		# Select random samples from feat
		sample_idxs = random.sample(range(n_samples), np.prod(n_plots))

		fig = plt.figure()
		grid = ImageGrid(fig, 111, nrows_ncols = n_plots, axes_pad = tile_pad, share_all = True, 
			cbar_mode = cbar_mode, cbar_pad = cbar_pad)

		for ax, idx in zip(grid, sample_idxs):
			arr = featureVecs[idx,:].reshape(array_shape)

			# set signal limits
			maxSig = arr.max()
			if noise_floor is not None:
				minSig = maxSig - noise_floor
				arr[arr < minSig] = minSig
			minSig = arr.min()

			#left = tile_psn[0]*(c+1) + tile_psn[2]*c
			#bottom = tile_psn[1]*(r+1) + tile_psn[3]*r

			#plt.axes((left, bottom)+tile_psn[2:])
			#print(plt.gca().get_position())

			im = ax.imshow(arr, aspect = aspect, extent = extent, interpolation = 'nearest', origin = origin,
				cmap = 'binary', vmin = minSig, vmax = maxSig)
			
			if cbar:
				cax.colorbar(im, ticks = [minSig, maxSig])
				cax.tick_params(length = 0)
				cax.set_yticklabels(['%.02f' % minSig, '%.02f' % maxSig], fontsize = 8)

			if xlims is not None:
				ax.set_xlim(xlims)

			if ylims is not None:
				ax.set_ylim(ylims)

			if titles is not None:
				ax.set_title(titles[idx], pad = 3, fontsize = 10)

		if xlabel is not None:
			grid.axes_llc.set_xlabel(xlabel, labelpad = 3)
			# set each x label
			#for c in range(n_plots[1]):
			# 	grid.axes_row[-1][c].set_xlabel(xlabel)
		else:
			grid.axes_llc.set_xticks([])

		if ylabel is not None:
			grid.axes_llc.set_ylabel(ylabel, labelpad = 3)
		else:
			grid.axes_llc.set_yticks([])

		plt.show()

	elif type(n_plots) is int:
		sample_idxs = np.array(random.sample(range(n_samples), n_plots))

		for idx in sample_idxs:
			arr = featureVecs[idx,:].reshape(array_shape)

			# set signal limits
			maxSig = arr.max()
			if noise_floor is not None:
				minSig = maxSig - noise_floor
				arr[arr < minSig] = minSig
			minSig = arr.min()

			plt.figure()
			# plt.axes(tile_psn)

			plt.imshow(arr, aspect = aspect, extent = extent, interpolation = 'nearest', origin = origin, 
				cmap = 'binary', vmin = minSig, vmax = maxSig)
			
			if cbar:
				cbar_ = plt.colorbar(ticks = [minSig, maxSig])
				cbar_.ax.tick_params(length = 0)
				cbar_.ax.set_yticklabels(['%.02f' % minSig, '%.02f' % maxSig])

			if xlims is not None:
				plt.xlim(xlims)

			if ylims is not None:
				plt.ylim(ylims)

			if xlabel is not None:
				plt.xlabel(xlabel)
			else:
				plt.xticks([])

			if ylabel is not None:
				plt.ylabel(ylabel)
			else:
				plt.yticks([])

			if titles is not None:
				plt.title(titles[idx])

			plt.show()
