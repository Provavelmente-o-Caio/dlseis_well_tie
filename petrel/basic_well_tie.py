import numpy as np
import pandas as pd
import segyio
import lasio
import yaml
import sys
import warnings
import json
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


from pathlib import Path

from wtie import grid, viz, autotie
from wtie.processing.logs import interpolate_nans, despike
from wtie.utils.datasets import tutorial
from wtie.utils.datasets.utils import InputSet


class Basic_well_tie:
    def __init__(self, logs_path, seis_path, path_path, table_path, data_path, config_path) -> None:
        with open(data_path, "r") as f:
            data = json.load(f)

        with open(config_path, "r") as f:
            config = json.load(f)

        self.las_logs = self.import_las_logs(logs_path, data)
        self.seis = self.import_seismic(seis_path)
        self.path = self.import_well_path(path_path, data)
        self.td_table = self.import_time_depth_table(table_path, data, config)

    def import_las_logs(self, file_path, data) -> grid.LogSet:
        file_path = Path(file_path)

        log_data = data["Logs"]

        # Read file
        print(file_path)
        las_logs = lasio.read(file_path)
        las_logs = las_logs.df()

        print(las_logs)

        log_dict = {}


        # Vp
        vp_curve = log_data["VP"]  # "DTCO"
        log_dict['Vp'] = grid.Log(
            las_logs[vp_curve].values, 
            las_logs[vp_curve].index, 
            "md", 
            name="Vp"
        )

        # Vs
        vs_curve = log_data["VS"]  # "DTSM"
        log_dict['Vs'] = grid.Log(
            las_logs[vs_curve].values, 
            las_logs[vs_curve].index, 
            "md", 
            name="Vs"
        )

        # Rho
        rho_curve = log_data["Rho"]  # "RHOB"
        log_dict['Rho'] = grid.Log(
            interpolate_nans(las_logs[rho_curve].values), 
            las_logs[rho_curve].index, 
            "md", 
            name="Rho"
        )

        # GR opcional
        if "GR" in log_data and log_data["GR"] != "":
            gr_curve = log_data["GR"]
            log_dict['GR'] = grid.Log(
                interpolate_nans(las_logs[gr_curve].values), 
                las_logs[gr_curve].index, 
                "md"
            )

        # Caliper opcional
        if "Cali" in log_data and log_data["Cali"] != "":
            cali_curve = log_data["Cali"]
            log_dict['Cali'] = grid.Log(
                las_logs[cali_curve].values, 
                las_logs[cali_curve].index, 
                "md"
            )
            
            
            return grid.LogSet(log_dict)

    def import_seismic(self, seis_path) -> grid.Seismic:
        # only works if not prestack
        file_path = Path(seis_path)

        with segyio.open(file_path, 'r') as f:
            twt = f.samples / 1000
            seis = np.squeeze(segyio.tools.cube(f))
        return grid.Seismic(seis, twt, 'twt')

    def import_well_path(self, path_path, data) -> grid.WellPath:
        file_path = Path(path_path)

        print(data)
        print(json.dumps(data, indent=4))
        print(data["Path"])

        wp = pd.read_csv(file_path, header=1, delimiter=r"\s+", usecols=range(1, len(data["Entire_Path"]) + 1), names=data["Entire_Path"])

        # Find out how to find this value
        kb = 0

        tvd = grid.WellPath.get_tvdkb_from_inclination(
            wp.loc[:, data["Path"][0]].values,
            wp.loc[:, data["Path"][1]].values[:-1]
        )

        tvd = grid.WellPath.tvdkb_to_tvdss(tvd, kb)

        # TODO: Find out how to find depth
        return grid.WellPath(md=wp.loc[:, data["Path"][0]].values, tvdss=tvd, kb=kb)

    def import_time_depth_table(self, table_path, data, config) -> grid.TimeDepthTable:
        file_path = Path(table_path)

        td = pd.read_csv(file_path, header=None, delimiter=r"\r+", skiprows=[0, 1], names=data["Entire_Table"], usecols=range(1, len(data["Entire_Table"])+1))

        if bool(config["isOWT"]):
            twt = td.loc[:, data["Table"][0]].values * 2 # owt to twt
        else:
            twt = td.loc[:, data["Table"][0]].values

        tvdss = td.loc[:, data["Table"][1]].values

        print(data["Table"][1])
        print(td)
        print(tvdss)

        return grid.TimeDepthTable(twt=twt, tvdss=tvdss)

    def import_seismic_trace(self, file_path: str, trace_idx: int = 0) -> grid.Seismic:
        """
        Importa um arquivo SEGY e retorna apenas um traço específico como grid.Seismic.
        TODO: Melhorar a importação para lidar com diferentes formatos de SEGY (1D/2D/3D).
        """
        file_path = Path(file_path)
        with segyio.open(file_path, "r", ignore_geometry=True) as f:
            twt = f.samples / 1000  # ms para s
            trace = f.trace[trace_idx]  # pega o traço desejado

        return grid.Seismic(
            trace, twt, "twt", name=f"{file_path.stem}_trace{trace_idx}"
        )

    def well_tie(self):
        inputs = InputSet(logset_md=self.logs, seismic=self.seis, wellpath=self.path, table=self.td_table)
        model_state_dict = Path("data/tutorial/trained_net_state_dict.pt")
        assert model_state_dict.is_file()

        with open(Path("data/tutorial/network_parameters.yaml"), "r") as yaml_file:
            training_parameters = yaml.load(yaml_file, Loader=yaml.Loader)

        wavelet_extractor = tutorial.load_wavelet_extractor(
            training_parameters, model_state_dict
        )

        modeler = tutorial.get_modeling_tool()

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

        outputs = autotie.tie_v1(
            inputs,
            wavelet_extractor,
            modeler,
            wavelet_scaling_params,
            search_params=search_params,
            search_space=search_space,
            stretch_and_squeeze_params=None,
        )

        best_parameters, values = outputs.ax_client.get_best_parameters()
        means, covariances = values

        outputs.plot_optimization_objective()
        fig, axes = outputs.plot_wavelet(fmax=85, phi_max=15, figsize=(6, 5))
        axes[0].set_xlim((-0.1, 0.1))
        axes[2].set_ylim((-12.5, 12.5))
        fig.tight_layout()
        _scale = 120000
        fig, axes = outputs.plot_tie_window(wiggle_scale=_scale, figsize=(12.0, 7.5))
        fig, ax = viz.plot_td_table(inputs.table, plot_params=dict(label="original"))
        viz.plot_td_table(
            outputs.table, plot_params=dict(label="modified"), fig_axes=(fig, ax)
        )
        ax.legend(loc="best")
        s_and_s_params = dict(window_length=0.060, max_lag=0.010)  # in seconds

        outputs2 = autotie.stretch_and_squeeze(
            inputs,
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

        fig, ax = viz.plot_td_table(inputs.table, plot_params=dict(label="original"))
        viz.plot_td_table(
            outputs2.table, plot_params=dict(label="modified"), fig_axes=(fig, ax)
        )
        ax.legend(loc="best")

        fig, ax = viz.plot_warping(
            outputs.synth_seismic, outputs.seismic, outputs2.dlags
        )
        plt.show()