import warnings
from ast import Pass
from pathlib import Path
from pprint import pprint

import lasio
import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import segyio
import yaml
from matplotlib.ticker import MaxNLocator
import sys

from wtie import autotie, grid, viz
from wtie.processing.logs import despike, interpolate_nans
from wtie.utils.datasets import tutorial
from wtie.utils.datasets.utils import InputSet
import os

class PetrelInterface:
    """Interface for dinamically integrating the wtie plugin into a Petrel workflow"""

    def __init__(self, well_name):
        print(os.listdir("D:\\"))  # Verifique se o drive está acessível
        time.sleep(5)
        # defining a temporary path
        # change this dinamically once we're able t  o figure out how to export all of this dinamically from petrel
        # self.folder = Path('data/tutorial/Volve')
        self.folder = Path("D:\\Caio\\dlseis_well_tie_petrel\\dlseis_well_tie_petrel\\bin\\Debug\\dlseis_well_tie\\data\\tutorial")
        assert self.folder.exists()
        # self.trajectory_path = self.folder / 'volve_path_15_9-19_A.txt'
        # self.table_path = self.folder / 'volve_checkshot_15_9_19A.txt'
        # self.logs_path = self.folder / 'volve_159-19A_LFP.las'
        # self.seis_path = self.folder / 'volve_15_9_19A_gather.sgy'

        # # loading the las logs
        # self.las_logs = lasio.read(self.logs_path).df()

        # # loading the seismic data
        # with segyio.open(self.seis_path, 'r') as f:
        #     self.twt = f.samples / 1000
        #     self.seis = np.squeeze(segyio.tools.cube(f))
        # self.seis = grid.Seismic(self.seis, self.wtw, 'twt', name='aaa')

        # # loading whe wellpath
        # self.wellpath = pd.read_csv(self.trajectory_path, sep='\t')

        # # loading the time/depth table
        # self.table =

        # self.input_set = InputSet(self.las_logs, self.seis, self.wellpath, self.table)
        self.inputs = tutorial.load_poseidon_data(self.folder, well=well_name)
        self.las_logs = self.inputs.logset_md

    def auto_well_tie(self):
        # loading neural network
        print(self.folder / "trained_net_state_dict.pt")
        model_state_dict = self.folder / "trained_net_state_dict.pt"
        assert model_state_dict.is_file()
        with open(self.folder / "network_parameters.yaml", "r") as yaml_file:
            training_parameters = yaml.load(yaml_file, Loader=yaml.Loader)
        wavelet_extractor = tutorial.load_wavelet_extractor(
            training_parameters, model_state_dict
        )
        # Load sunthetic modeling tool
        modeler = tutorial.get_modeling_tool()

        # Logs processing
        median_length_choice = dict(
            name="logs_median_size",
            type="choice",
            values=[i for i in range(11, 63, 2)],
            value_type="int",
        )

        median_th_choice = dict(
            name="logs_median_threshold",
            type="range",
            bounds=[0.1, 5.5],
            value_type="float",
        )

        std_choice = dict(
            name="logs_std", type="range", bounds=[0.5, 5.5], value_type="float"
        )

        # bulk shift in seconds
        table_t_shift_choice = dict(
            name="table_t_shift",
            type="range",
            bounds=[-0.012, 0.012],
            value_type="float",
        )

        search_space = [
            median_length_choice,
            median_th_choice,
            std_choice,
            table_t_shift_choice,
        ]

        search_params = dict(num_iters=80, similarity_std=0.02)
        wavelet_scaling_params = dict(
            wavelet_min_scale=50000, wavelet_max_scale=500000, num_iters=60
        )

        warnings.filterwarnings("ignore")
        warnings.simplefilter("ignore")

        outputs = autotie.tie_v1(
            self.inputs,
            wavelet_extractor,
            modeler,
            wavelet_scaling_params,
            search_params=search_params,
            search_space=search_space,
            stretch_and_squeeze_params=None,
        )

        best_parameters, values = outputs.ax_client.get_best_parameters()
        means, covariances = values
        # Results from the well tie
        outputs.plot_optimization_objective()
        fig, axes = outputs.plot_wavelet(fmax=85, phi_max=15, figsize=(6, 5))
        axes[0].set_xlim((-0.1, 0.1))
        axes[2].set_ylim((-12.5, 12.5))
        fig.tight_layout()
        _scale = 120000
        fig, axes = outputs.plot_tie_window(wiggle_scale=_scale, figsize=(12.0, 7.5))
        fig, ax = viz.plot_td_table(
            self.inputs.table, plot_params=dict(label="original")
        )
        viz.plot_td_table(
            outputs.table, plot_params=dict(label="modified"), fig_axes=(fig, ax)
        )
        ax.legend(loc="best")
        plt.show()

        # Stretch and Squeeze
        s_and_s_params = dict(window_length=0.060, max_lag=0.010)  # in seconds

        outputs2 = autotie.stretch_and_squeeze(
            self.inputs,
            outputs,
            wavelet_extractor,
            modeler,
            wavelet_scaling_params,
            best_parameters,
            s_and_s_params,
        )
        fig, axes = outputs2.plot_wavelet(fmax=85, phi_max=25, figsize=(6, 5))
        axes[0].set_xlim((-0.1, 0.1))
        axes[2].set_ylim((-6.0, 12.5))
        fig.tight_layout()
        fig, axes = outputs2.plot_tie_window(wiggle_scale=_scale, figsize=(12.0, 7.5))
        axes[0].xaxis.set_major_locator(MaxNLocator(nbins=2))
        fig, ax = viz.plot_td_table(
            self.inputs.table, plot_params=dict(label="original")
        )
        viz.plot_td_table(
            outputs2.table, plot_params=dict(label="modified"), fig_axes=(fig, ax)
        )
        ax.legend(loc="best")
        fig, ax = viz.plot_warping(
            outputs.synth_seismic, outputs.seismic, outputs2.dlags
        )
        plt.show()
