import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

def plotFeatureArrays(feature_vecs, array_shape, n_plots = (5, 5), tile_pad = None, 
	aspect = None, xlims = None, ylims = None, xlabel = None, ylabel = None, titles = None, 
	noise_floor = None, extent = None, origin = None, cbar = False):

	if tile_pad is None:
		tile_pad = (0.1, 0.1)

		if titles is not None:
			tile_pad[1] += 0.15

		if cbar:
			tile_pad[0] += .4
			tile_pad[1] += .05

	if cbar:
		cbar_mode = 'each'
		cbar_pad = .03
	else:
		cbar_mode = None
		cbar_pad = None

	if type(n_plots) is tuple and len(n_plots) == 2:
		fig = plt.figure()
		grid = ImageGrid(fig, 111, nrows_ncols = n_plots, axes_pad = tile_pad, share_all = True, 
			cbar_mode = cbar_mode, cbar_pad = cbar_pad)

		if titles is None:
			titles = np.full(np.prod(n_plots), None)

		for ax, arr, title in zip(grid, feature_vecs, titles):
			arr = arr.reshape(array_shape)

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

			if title is not None:
				ax.set_title(title, pad = 3, fontsize = 10)

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
		if titles is None:
			titles = np.full(n_plots, None)

		for arr, title in zip(feature_vecs, titles):
			arr = arr.reshape(array_shape)

			# set signal limits
			maxSig = arr.max()
			if noise_floor is not None:
				minSig = maxSig - noise_floor
				arr[arr < minSig] = minSig
			minSig = arr.min()

			plt.figure()
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

			if title is not None:
				plt.title(title)

			plt.show()
