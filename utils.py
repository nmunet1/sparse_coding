import random
import numpy as np
from matplotlib import pyplot as plt

def plotFeatureArrays(featureVecs, array_shape, n_figs = 10, tiled = True, tiles_shape = (2,5),
	tile_psn = (.1, .125, .4, .4), xlims = None, ylims = None, titles = None, xlabel = None, ylabel = None,
	noise_floor = None, extent = None, origin = None, colorbar = True, seed_id = 22):
	
	n_samples = featureVecs.shape[0]
	if seed_id is not None:
		random.seed(seed_id)
	print(seed_id)

	if tiled:
		sample_idxs = np.array(random.sample(range(n_samples), np.prod(tiles_shape))).reshape(tiles_shape)

		plt.figure()

		for r in range(tiles_shape[0]):
			for c in range(tiles_shape[1]):
				idx = sample_idxs[r,c]
				arr = featureVecs[idx,:].reshape(array_shape)

				# set signal limits
				maxSig = arr.max()
				if noise_floor is not None:
					minSig = maxSig - noise_floor
					arr[arr < minSig] = minSig
				minSig = arr.min()

				left = tile_psn[0]*(c+1) + tile_psn[2]*c
				bottom = tile_psn[1]*(r+1) + tile_psn[3]*r

				plt.axes((left, bottom)+tile_psn[2:])
				#print(plt.gca().get_position())

				plt.imshow(arr, extent = extent, aspect = 'auto', interpolation = 'nearest', origin = origin,
					cmap = 'binary', vmin = minSig, vmax = maxSig)
				
				if colorbar:
					plt.colorbar()

				if xlims is not None:
					plt.xlim(xlims)

				if ylims is not None:
					plt.ylim(ylims)

				if xlabel is not None and r == 0:
					plt.xlabel(xlabel)
				else:
					plt.xticks([])

				if ylabel is not None and c == 0:
					plt.ylabel(ylabel)
				else:
					plt.yticks([])

				if titles is not None:
					plt.title(titles[idx])

		plt.show()

	else:
		sample_idxs = np.array(random.sample(range(n_samples), n_figs))

		for idx in sample_idxs:
			arr = featureVecs[idx,:].reshape(array_shape)

			# set signal limits
			maxSig = arr.max()
			if noise_floor is not None:
				minSig = maxSig - noise_floor
				arr[arr < minSig] = minSig
			minSig = arr.min()

			plt.figure()
			plt.axes(tile_psn)

			plt.imshow(arr, extent = extent, aspect = 'auto', interpolation = 'nearest', origin = origin,
				cmap = 'binary', vmin = minSig, vmax = maxSig)
			
			if colorbar:
				plt.colorbar()

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
