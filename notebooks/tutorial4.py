import numpy as np
import pandas as pd

from pathlib import Path
import yaml

from wtie import grid, autotie, viz
from wtie.utils.datasets import tutorial

from pprint import pprint

import matplotlib.pyplot as plt

# import data
# data path
folder = Path('data/tutorial')
assert folder.exists()

inputs = tutorial.load_volve_data(folder, prestack=True)

# choose angle range
tmp_gather = grid.BasePrestackTrace.partial_stacking(inputs.seismic, n=1) # "super traces"
inputs.seismic = grid.BasePrestackTrace.decimate_angles(tmp_gather, 2, 36, 2) # decimate

# Load pretrained network to extract the wavelet
# neural network's weights
model_state_dict = folder / 'trained_net_state_dict.pt'
assert model_state_dict.is_file()

# network training paramters
with open(folder / 'network_parameters.yaml', 'r') as yaml_file:
    training_parameters = yaml.load(yaml_file, Loader=yaml.Loader)
    
wavelet_extractor = tutorial.load_wavelet_extractor(training_parameters, model_state_dict)

# Load synthetic modeling tool
modeler = tutorial.get_modeling_tool()

# Parameters for the search and optimization
print(autotie.tie_v1.__doc__)

# Logs processing
median_length_choice = dict(name="logs_median_size", type="choice",
                  values=[i for i in range(11,63,2)], value_type="int")

median_th_choice = dict(name="logs_median_threshold", type="range",
                  bounds=[0.1, 6.5], value_type="float")

std_choice = dict(name="logs_std", type="range",
                  bounds=[0.5, 5.5], value_type="float")


# bulk shift
table_t_shift_choice = dict(name="table_t_shift", type="range",
                  bounds=[-0.012, 0.012], value_type="float")


search_space = [median_length_choice,
                median_th_choice,
                std_choice,
                table_t_shift_choice
                ]

search_params = dict(num_iters=75, similarity_std=0.02, random_ratio=0.7) #70

import warnings
warnings.filterwarnings('ignore')

wavelet_scaling_params = dict(wavelet_min_scale=0.01, wavelet_max_scale=1.0, num_iters=35) #30


outputs = autotie.tie_v1(inputs,
                         wavelet_extractor,
                         modeler,
                         wavelet_scaling_params,
                         search_params=search_params,
                         search_space=search_space,
                         stretch_and_squeeze_params=None)

best_parameters, values = outputs.ax_client.get_best_parameters()
means, covariances = values
print(means)
print(covariances)

pprint(best_parameters)

outputs.plot_optimization_objective()

# visualize results
#near, mid, far
fig, axes = outputs.plot_wavelet(three_angles=[6,20,34], fmax=60, phi_max=55, figsize=(12,6))
for ax in axes[:3]:
    ax.set_ylim((-.075, .20))
for ax in axes[6:]:
    ax.set_ylim(-20,0)

# look at one wavelet
print(outputs.wavelet.angles)
angle_ = 6 # in degrees
viz.plot_wavelet(outputs.wavelet[angle_], fmax=65, phi_max=55, figsize=None);

fig,ax = viz.plot_prestack_wiggle_trace(outputs.wavelet, figsize=(7,4))
ax.set_ylim((-0.08,0.08))
fig.suptitle("")

outputs.seismic.angles

# all wavelets
fig, axes = plt.subplots(1,2, figsize=(10,6))
viz.plot_prestack_wiggle_trace(outputs.wavelet, fig_axes=(fig,axes[0]))
viz.plot_prestack_trace_as_pixels(outputs.wavelet, fig_axes=(fig,axes[1]),
                                  im_params=dict(cmap='RdBu', vmin=-0.15, vmax=0.15))

fig.tight_layout()

wigg_scale_ = 150
fig, axes = outputs.plot_tie_window(wiggle_scale=wigg_scale_, figsize=(12,7),
                                   decimate_wiggles=1, reflectivity_scale=15)

fig, axes, cbar = viz.plot_prestack_residual_as_pixels(outputs.seismic, outputs.synth_seismic,
                                                   im_params=dict(cmap='RdBu', vmin=-4e-2,vmax=4e-2))
axes[0].set_title("Real gather")
axes[1].set_title("Synthetic gather")

fig,ax = viz.plot_td_table(inputs.table, plot_params=dict(label='original'))
viz.plot_td_table(outputs.table,  plot_params=dict(label='modified'), fig_axes=(fig,ax))
ax.legend(loc='best')

# Automatic strech & squeeze
s_and_s_params = dict(window_length=0.040, max_lag=0.008, reference_angle=14)

outputs2 = autotie.stretch_and_squeeze(inputs,
                                       outputs,
                                       wavelet_extractor,
                                       modeler,
                                       wavelet_scaling_params,
                                       best_parameters,
                                       s_and_s_params)

fig, axes = outputs2.plot_wavelet(three_angles=[6,20,34], fmax=60, phi_max=55, figsize=(13,7))
for ax in axes[:3]:
    ax.set_ylim((-.08, .21))
for ax in axes[6:]:
    ax.set_ylim(-40,20)

fig,ax = viz.plot_prestack_wiggle_trace(outputs2.wavelet, figsize=(7,4))
ax.set_ylim((-0.08,0.08))
fig.suptitle("")

fig, axes = outputs2.plot_tie_window(wiggle_scale=wigg_scale_, figsize=(12,7),
                                   decimate_wiggles=1, reflectivity_scale=15)
axes[-1].set_title("Cross-correlation")

fig,ax = viz.plot_td_table(inputs.table, plot_params=dict(label='original'))
viz.plot_td_table(outputs2.table,  plot_params=dict(label='modified'), fig_axes=(fig,ax))
ax.legend(loc='best')

plt.show()