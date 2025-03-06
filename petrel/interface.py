from ast import Pass
import numpy as np
import pandas as pd
import segyio
import lasio

from pathlib import Path
from pprint import pprint

from wtie import grid, viz
from wtie.processing.logs import interpolate_nans, despike

import matplotlib.pyplot as plt

class PetrelInterface:
    """Interface for dinamically integrating the wtie plugin into a Petrel workflow"""
    # defining a temporary path
    # change this dinamically once we're able to figure out how to export all of this dinamically from petrel
    def __init__(self):
        self.folder = Path('data/tutorial/Volve')
        self.trajectory_path = self.folder / 'volve_path_15_9-19_A.txt'
        self.table_path = self.folder / 'volve_checkshot_15_9_19A.txt'
        self.logs_path = self.folder / 'volve_159-19A_LFP.las'
        self.seis_path = self.folder / 'volve_15_9_19A_gather.sgy'

        assert self.folder.exists()

        # loading the las logs
        self.las_logs = lasio.read(self.logs_path).df()

        # loading the seismic data
        with segyio.open(self.seis_path, 'r') as f:
            print(f.samples.size) # number of time samples
            print(f.ilines) 
            print(f.xlines)
            print(f.offsets) # these are actually angles, from 0 to 45 degrees

    def plot_las_logs(self):
        # TODO: better define other logs to be plotted
        # Currently only VP, VS and RHO are selected, the user should be able to select any log

        log_dict = {}

        log_dict['Vp'] = grid.Log(self.las_logs.LFP_VP.values, self.las_logs.LFP_VP.index, 'md', name='Vp')
        log_dict['Vs'] = grid.Log(self.las_logs.LFP_VS.values, self.las_logs.LFP_VS.index, 'md', name='Vs')

        # Density contains some NaNs, I fill them with linear interpolation.
        log_dict['Rho'] = grid.Log(interpolate_nans(self.las_logs.LFP_RHOB.values), self.las_logs.LFP_RHOB.index, 'md', name='Rho')
        logset_md = grid.LogSet(log_dict)
        viz.plot_logset(logset_md)
        plt.show()

    def plot_seismic(self):
        Pass # type: ignore