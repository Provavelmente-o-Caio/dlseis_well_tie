import json
import sys
from pathlib import Path

import lasio
import numpy as np
import pandas as pd
import segyio
import yaml

from wtie import autotie, grid
from wtie.processing.grid import LogSet, Seismic, TimeDepthTable, WellPath
from wtie.processing.logs import interpolate_nans
from wtie.processing.spectral import compute_spectrum
from wtie.utils.datasets import tutorial
from wtie.utils.datasets.utils import InputSet

EXPECTED_ARGUMENTS = 7


class Basic_well_tie:
    def __init__(
        self,
        logs_path: str,
        seis_path: str,
        path_path: str,
        table_path: str,
        data_path: str,
        config_path: str,
        output_path: str,
    ) -> None:
        with open(data_path, "r") as f:
            data = json.load(f)

        with open(config_path, "r") as f:
            config = json.load(f)

        self.las_logs: LogSet = self.import_las_logs(logs_path, data, config)
        self.seis: Seismic = self.import_seismic(seis_path)
        self.path: WellPath = self.import_well_path(path_path, data, config)
        self.td_table: TimeDepthTable = self.import_time_depth_table(
            table_path, data, config
        )
        self.output_path = Path(output_path)

    def import_las_logs(self, file_path, data, config) -> grid.LogSet:
        file_path = Path(file_path)

        log_data = data["Logs"]
        configs = config["Logs"]

        # Read file
        las_logs = lasio.read(file_path)
        las_logs = las_logs.df()

        log_dict = {}

        # loading configs
        start_burn = int(configs["Start_range"])
        end_burn = int(configs["End_range"])
        las_unit = configs["las_unit"]

        # Vp
        vp_curve = log_data["VP"]  # ex: "DTCO"
        sonic = las_logs[vp_curve].values[start_burn:-end_burn]
        sonic = self.convert_to_mps(sonic, las_unit)
        md = las_logs[vp_curve].index.values[start_burn:-end_burn]

        log_dict["Vp"] = grid.Log(
            sonic,
            md,
            "md",
            name="Vp",
        )

        # Vs
        vs_curve = log_data["VS"]  # "DTSM"
        shear = las_logs[vs_curve].values[start_burn:-end_burn]
        shear = self.convert_to_mps(shear, las_unit)
        md = las_logs[vs_curve].index.values[start_burn:-end_burn]

        log_dict["Vs"] = grid.Log(
            shear,
            md,
            "md",
            name="Vs",
        )

        # Rho
        rho_curve = log_data["Rho"]  # "RHOB"
        rho = interpolate_nans(las_logs[rho_curve].values[start_burn:-end_burn])
        md = las_logs[rho_curve].index[start_burn:-end_burn]

        log_dict["Rho"] = grid.Log(
            rho,
            md,
            "md",
            name="Rho",
        )

        # GR opcional
        if "GR" in log_data and log_data["GR"] != "":
            gr_curve = log_data["GR"]
            log_dict["GR"] = grid.Log(
                interpolate_nans(las_logs[gr_curve].values[start_burn:-end_burn]),
                las_logs[gr_curve].index.values[start_burn:-end_burn],
                "md",
            )

        # Caliper opcional
        if "Cali" in log_data and log_data["Cali"] != "":
            cali_curve = log_data["Cali"]
            log_dict["Cali"] = grid.Log(
                las_logs[cali_curve].values[start_burn:-end_burn],
                las_logs[cali_curve].index.values[start_burn:-end_burn],
                "md",
            )

        return grid.LogSet(log_dict)

    def import_seismic(self, seis_path) -> grid.Seismic:
        # only works if not prestack
        file_path = Path(seis_path)

        with segyio.open(file_path, "r") as f:
            twt = f.samples / 1000
            seis = np.squeeze(segyio.tools.cube(f))
        return grid.Seismic(seis, twt, "twt")

    def import_well_path(self, path_path, data, config) -> grid.WellPath:
        file_path = Path(path_path)
        configs = config["Path"]

        wp = pd.read_csv(
            file_path,
            header=1,
            delimiter=r"\s+",
            usecols=range(1, len(data["Entire_Path"]) + 1),
            names=data["Entire_Path"],
            engine="python",
        )

        depth, inclination = data["Path"][0], data["Path"][1]

        # md = np.concatenate((np.zeros((1,)), wp.loc[:, depth].values))
        # dev = np.concatenate((np.zeros((1,)), wp.loc[:, inclination].values[:-1]))

        md = wp.loc[:, depth].values
        dev = wp.loc[:, inclination].values[:-1]

        # Find out how to find this value
        kb = float(configs["datum"])  # meters

        try:
            tvd = grid.WellPath.get_tvdkb_from_inclination(md, dev)
            tvd = grid.WellPath.tvdkb_to_tvdss(tvd, kb)
        except AssertionError:
            md = np.concatenate((np.zeros((1,)), md))
            dev = np.concatenate((np.zeros((1,)), dev))
            tvd = grid.WellPath.get_tvdkb_from_inclination(md, dev)
            tvd = grid.WellPath.tvdkb_to_tvdss(tvd, kb)
        except Exception as e:
            print(f"An error occurred: {e}")

        return grid.WellPath(md=md, tvdss=tvd, kb=kb)

    def import_time_depth_table(self, table_path, data, config) -> grid.TimeDepthTable:
        file_path = Path(table_path)
        configs = config["Table"]

        file_extension = file_path.suffix.lower()

        if file_extension == ".las":
            td = lasio.read(file_path).df()
        else:
            td = pd.read_csv(
                file_path,
                header=None,
                sep=r"\s+",
                skiprows=[0, 1],
                names=data["Entire_Table"],
            )

        if bool(configs["isOWT"]):
            twt = td.loc[:, data["Table"][0]].values * 2  # owt to twt
        else:
            twt = td.loc[:, data["Table"][0]].values
        tvdss = td.loc[:, data["Table"][1]].values
        # remove nans
        good_idx = np.where(~np.isnan(twt))[0]
        twt = twt[good_idx]
        tvdss = tvdss[good_idx]

        if configs["Unit"] == "ms":
            twt /= 1000

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
        inputs = InputSet(
            logset_md=self.las_logs,
            seismic=self.seis,
            wellpath=self.path,
            table=self.td_table,
        )

        model_state_dict = Path("../data/tutorial/trained_net_state_dict.pt")
        assert model_state_dict.is_file()

        with open(Path("../data/tutorial/network_parameters.yaml"), "r") as yaml_file:
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

        self.export_output(
            outputs2, self.output_path, best_parameters, means, covariances
        )

    def export_output(
        self,
        output: autotie.OutputSet,
        output_path: Path,
        best_parameters: dict,
        means: dict,
        covariances: dict,
    ):
        result = {}

        # Wavelet data
        wavelet = output.wavelet
        ff, ampl, _, phase = compute_spectrum(
            wavelet.values, wavelet.sampling_rate, to_degree=True
        )
        ampl /= ampl.max()

        result["wavelet"] = {
            "basis": wavelet.basis.tolist()
            if hasattr(wavelet.basis, "tolist")
            else wavelet.basis,
            "values": wavelet.values.tolist()
            if hasattr(wavelet.values, "tolist")
            else wavelet.values,
            "sampling_rate": float(wavelet.sampling_rate),
            "frequency": ff.tolist() if hasattr(ff, "tolist") else ff,
            "amplitude": ampl.tolist() if hasattr(ampl, "tolist") else ampl,
            "phase": phase.tolist() if hasattr(phase, "tolist") else phase,
        }

        # Tie window data (synthetic seismic vs real seismic)
        result["tie_window"] = {
            "ai": output.logset_twt.AI.tolist()
            if hasattr(output.logset_twt.AI, "tolist")
            else output.logset_twt.AI,
            "r0": output.r.tolist() if hasattr(output.r, "tolist") else output.r,
            "synthetic_seismic": output.synth_seismic.tolist()
            if hasattr(output.synth_seismic, "tolist")
            else output.synth_seismic,
            "real_seismic": output.seismic.tolist()
            if hasattr(output.seismic, "tolist")
            else output.seismic,
            "time_twt": output.twt.tolist()
            if hasattr(output.twt, "tolist")
            else output.twt,
        }

        # Time-Depth Table data
        result["td_table"] = {
            "original": {
                "twt": self.td_table.twt.tolist()
                if hasattr(self.td_table.twt, "tolist")
                else self.td_table.twt,
                "tvdss": self.td_table.tvdss.tolist()
                if hasattr(self.td_table.tvdss, "tolist")
                else self.td_table.tvdss,
            },
            "modified": {
                "twt": output.table.twt.tolist()
                if hasattr(output.table.twt, "tolist")
                else output.table.twt,
                "tvdss": output.table.tvdss.tolist()
                if hasattr(output.table.tvdss, "tolist")
                else output.table.tvdss,
            },
        }

        # Warping Path data
        result["warping_path"] = {
            "dlags": output.dlags.tolist()
            if hasattr(output.dlags, "tolist")
            else output.dlags,
            "time_twt": output.twt.tolist()
            if hasattr(output.twt, "tolist")
            else output.twt,
        }

        # Best parameters from optimization
        result["optimization"] = {
            "best_parameters": {
                k: float(v) if isinstance(v, (int, float, np.number)) else v
                for k, v in best_parameters.items()
            },
            "means": means,
            "covariances": covariances,
        }

        # Export to JSON
        output_file = output_path / "well_tie_results.json"
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, "w") as f:
            json.dump(result, f, indent=2)

        print(f"RESULTS_JSON:{output_file.name}")

    def convert_to_mps(self, values, unit: str):
        """
        Convert LAS log velocity/sonic units to m/s.

        Supported:
          - us/ft
          - us/m
          - ft/s
          - m/s
          - km/s
          - s/m (slowness)
        """

        if values is None:
            return values

        u = unit.strip().lower()

        values = np.asarray(values, dtype=float)

        if u in ["m/s", "mps"]:
            return values

        # ---- slowness ----
        if u in ["us/ft", "µs/ft"]:
            v = 1.0 / values  # ft/us
            v *= 1e6  # ft/s
            v *= 0.3048  # m/s
            return v

        if u in ["us/m", "µs/m"]:
            v = 1.0 / values  # m/us
            v *= 1e6  # m/s
            return v

        if u in ["s/m"]:
            return 1.0 / values

        # ---- velocity ----
        if u in ["ft/s", "fps"]:
            return values * 0.3048

        if u in ["km/s"]:
            return values * 1000.0

        raise ValueError(f"Unsupported unit for velocity conversion: '{unit}'")


if __name__ == "__main__":
    if len(sys.argv) >= EXPECTED_ARGUMENTS + 1:
        (
            _,
            log_path,
            seismic_path,
            wellpath_path,
            td_table_path,
            data_path,
            config_path,
            output_path,
        ) = sys.argv

        wt = Basic_well_tie(
            log_path,
            seismic_path,
            wellpath_path,
            td_table_path,
            data_path,
            config_path,
            output_path,
        )
        wt.well_tie()
    else:
        print("[ERRO] Argumentos insuficientes!")
        print(f"\nEsperado: 7 argumentos, recebido: {len(sys.argv) - 1}")
        print("\nUso:")
        print(
            "  python teste.py LOG_PATH SEISMIC_PATH WELLPATH_PATH "
            "TD_TABLE_PATH DATA_JSON CONFIG_JSON"
        )
        print("\nExemplo:")
        print(
            "  python teste.py data/logs.las data/seismic.sgy data/path.txt "
            "data/table.txt data.json config.json"
        )
        sys.exit(1)
