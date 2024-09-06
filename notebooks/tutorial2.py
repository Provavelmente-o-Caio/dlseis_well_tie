import numpy as np
import pandas as pd

from pathlib import Path
import yaml
from pprint import pprint

from wtie import grid, viz, tie
from wtie.utils.datasets import tutorial

import matplotlib.pyplot as plt

from wtie.optimize.wavelet import compute_expected_wavelet
from wtie.optimize import similarity

# data path
folder = Path('data/tutorial')

# inputs is a class that stores the useful input data to perform a well tie
inputs = tutorial.load_poseidon_data(folder) 
pprint(inputs.__dict__)
inputs.plot_inputs(figsize=(7.5,3.5));

# Load neural network to perform wavelet extraction
# neural network's pre-trained weights
model_state_dict = folder / 'trained_net_state_dict.pt'
assert model_state_dict.is_file()

# network training paramters
with open(folder / 'network_parameters.yaml', 'r') as yaml_file:
    training_parameters = yaml.load(yaml_file, Loader=yaml.Loader)

# Filter the well logs
wavelet_extractor = tutorial.load_wavelet_extractor(training_parameters, model_state_dict)

filtered_logset_md = tie.filter_md_logs(inputs.logset_md,
                                        median_size=21, threshold=2.0, std=2.0,std2=None)

viz.plot_logsets_overlay(inputs.logset_md, filtered_logset_md, figsize=(6,5))

# Depth to time conversion
logset_twt = tie.convert_logs_from_md_to_twt(filtered_logset_md,
                                             inputs.wellpath,
                                             inputs.table,
                                             dt=wavelet_extractor.expected_sampling)
viz.plot_logset(logset_twt, figsize=(6,5));

#compute velocities from Vp and checkshot tables
fig,ax = viz.plot_trace(logset_twt.Vp, plot_params=dict(label='Sonic log'))
viz.plot_trace(inputs.table.slope_velocity_twt(), fig_axes=(fig,ax), plot_params=dict(label='Checkshot velocity'))
ax.legend(loc='best')
ax.set_ylim((logset_twt.Vp.basis[0], logset_twt.Vp.basis[-1]))

# Compute vertical incidence reflectity
r0 = tie.compute_reflectivity(logset_twt)
viz.plot_reflectivity(r0)

# Interpolate seismic and find intersection with reflectivity
seismic_sinc = tie.resample_seismic(inputs.seismic, wavelet_extractor.expected_sampling)
seis_match, r0_match = tie.match_seismic_and_reflectivity(seismic_sinc, r0)

viz.plot_seismic_and_reflectivity(seis_match, r0_match, normalize=True, title='Real seismic and reflectivity')

pred_wlt = compute_expected_wavelet(evaluator=wavelet_extractor,
                                    seismic=seis_match,
                                    reflectivity=r0_match)
fig, axes = viz.plot_wavelet(pred_wlt, figsize=(6,5), fmax = 85, phi_max = 100)
axes[2].set_ylim((-90,5))

# Compute synthetic seismic
modeler = tutorial.get_modeling_tool()

synth_seismic = tie.compute_synthetic_seismic(modeler, pred_wlt, r0_match)
viz.plot_seismics(seis_match,
                  synth_seismic,
                  r0_match);

# Similarity between real and systhetic seismic
xcorr = similarity.traces_normalized_xcorr(seis_match, synth_seismic)
xcorr = grid.upsample_trace(xcorr, 0.001)
dxcorr = similarity.dynamic_normalized_xcorr(seis_match, synth_seismic)
print("Max coeff of %.2f at a lag of %.3f sec" % (xcorr.R, xcorr.lag))

# Visualize results
fig, axes = viz.plot_tie_window(logset_twt,
                    r0_match,
                    synth_seismic,
                    seis_match,
                    xcorr,
                    dxcorr,
                    figsize=(12,6),
                    wiggle_scale_syn=0.5,
                    wiggle_scale_real=110000
                    ) 
axes[0].locator_params(axis='y', nbins=16)
fig.tight_layout()

plt.show()