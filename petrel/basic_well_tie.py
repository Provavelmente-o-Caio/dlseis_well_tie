import numpy as np
import pandas as pd
import segyio
import lasio
import yaml
import sys
import warnings
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


from pathlib import Path

from wtie import grid, viz, autotie
from wtie.processing.logs import interpolate_nans, despike
from wtie.utils.datasets import tutorial
from wtie.utils.datasets.utils import InputSet


class Basic_well_tie:
    def __init__(self, logs_path, seis_path, path_path, table_path) -> None:
        self.logs = self.import_logs_generic(logs_path)
        self.seis = self.import_seismic_trace(seis_path)
        self.path = self.import_well_path_generic(path_path)
        self.td_table = self.import_time_depth_table_generic(table_path)

    def import_logs_generic(self, file_path: str) -> grid.LogSet:
        """
        Importa qualquer arquivo LAS e retorna um grid.LogSet com todos os logs disponíveis,
        mapeando nomes alternativos para 'Vp' e 'Rho' se necessário e interpolando para base regular.
        """
        file_path = Path(file_path)
        las_logs = lasio.read(file_path)
        las_df = las_logs.df()
        logs = {}
        # Define base regular (usando o menor passo encontrado)
        md = las_df.index.values
        step = np.min(np.diff(md))
        md_regular = np.arange(md[0], md[-1] + step, step)
        # Mapeamento de nomes alternativos
        name_map = {
            "Vp": [
                col
                for col in las_df.columns
                if col.startswith("DT") or col.upper() in ["VP", "DTCO", "SONIC"]
            ],
            "Rho": [
                col
                for col in las_df.columns
                if col.upper() in ["RHO", "RHOB", "RHOZ", "DENS"]
            ],
        }
        # Primeiro, tenta mapear os obrigatórios
        for key, aliases in name_map.items():
            for alias in aliases:
                if alias in las_df.columns:
                    values = las_df[alias].values
                    # Interpola para base regular
                    interp_values = np.interp(md_regular, md, interpolate_nans(values))
                    if key == "Vp":
                        if np.nanmean(values) > 100:  # provavelmente DT
                            interp_values = (
                                1 / interp_values * 1e6 / 3.2808
                            )  # ft/us -> m/s
                    logs[key] = grid.Log(interp_values, md_regular, "md", name=key)
                    break
        # Adiciona os demais logs
        for col in las_df.columns:
            if col not in logs:
                try:
                    values = las_df[col].values
                    interp_values = np.interp(md_regular, md, interpolate_nans(values))
                    logs[col] = grid.Log(interp_values, md_regular, "md", name=col)
                except Exception as e:
                    print(f"Não foi possível importar o log {col}: {e}")
        return grid.LogSet(logs=logs)

    def import_well_path_generic(self, file_path: str, kb: float = 0) -> grid.WellPath:
        """
        Importa um arquivo de trajetória de poço.
        """
        file_path = Path(file_path)

        try:
            # Verifica primeiro se é formato Boreas (6 colunas)
            with open(file_path) as f:
                # Pula a primeira linha
                next(f)
                primeira_linha_dados = next(f).strip()

            # Se tem 6 números, é formato Boreas
            if len(primeira_linha_dados.split()) == 11:
                # Lê pulando apenas as duas primeiras linhas
                df = pd.read_csv(
                    file_path, header=None, delimiter=r"\s+", skiprows=[0, 1]
                )
                # Reorganiza os dados das 6 colunas em 3
                data = []
                for _, row in df.iterrows():
                    data.extend([(row[0], row[1]), (row[4], row[6])])
                df_clean = pd.DataFrame(data, columns=["MD", "INC"])
            else:
                # Formato Volve (3 colunas)
                df = pd.read_csv(
                    file_path,
                    header=None,
                    delimiter=r"\s+",
                    comment="#",
                    skip_blank_lines=True,
                )
                df.columns = ["MD", "INC", "AZI"][: len(df.columns)]
                df_clean = df[["MD", "INC"]]

            # Limpa e ordena os dados
            df_clean = df_clean.apply(pd.to_numeric, errors="coerce").dropna()
            df_clean = df_clean.sort_values("MD").drop_duplicates("MD")

            md = df_clean["MD"].values
            inc = df_clean["INC"].values

            # Para o formato Volve, garantimos que inc tem um elemento a menos que md
            if len(inc) == len(md):
                inc = inc[:-1]

            # Garante que começa em MD=0
            if md[0] != 0:
                md = np.concatenate(([0.0], md))
                inc = np.concatenate(([0.0], inc))

            # Garante que os valores são estritamente crescentes
            mask = np.diff(md) > 0
            mask = np.concatenate(([True], mask))
            md = md[mask]
            inc = inc[mask[:-1]]  # ajusta o tamanho de inc

            # Calcula TVD e converte para TVDSS
            tvd = grid.WellPath.get_tvdkb_from_inclination(md, inc)
            tvd = grid.WellPath.tvdkb_to_tvdss(tvd, kb)

            return grid.WellPath(md=md, tvdss=tvd, kb=kb)

        except Exception as e:
            raise ValueError(f"Erro ao ler arquivo {file_path}: {e}")

    def import_time_depth_table_generic(self, file_path: str) -> grid.TimeDepthTable:
        """
        Importa uma tabela tempo/profundidade genérica (espera colunas: TWT, TVDSS).
        Suporta formatos:
        1. Duas colunas simples: TWT TVDSS
        2. Formato Boreas: Depth TVDSS OWT Depth TVDSS OWT
        """
        file_path = Path(file_path)

        # Lê o arquivo ignorando comentários
        df = pd.read_csv(
            file_path, delimiter=r"\s+", header=None, comment="#", skip_blank_lines=True
        )

        # Verifica o formato pelo número de colunas
        if len(df.columns) >= 6:  # Formato Boreas
            # Reorganiza os dados das 6 colunas em 2 (TVDSS, OWT)
            data = []
            for _, row in df.iterrows():
                # Pega apenas TVDSS e OWT de cada par
                data.extend([(row[2], row[1]), (row[5], row[4])])
            df_clean = pd.DataFrame(data, columns=["TWT", "TVDSS"])
        else:  # Formato duas colunas
            df_clean = df.iloc[:, :2]
            df_clean.columns = ["TWT", "TVDSS"]

        # Limpa e ordena os dados
        df_clean = df_clean.apply(pd.to_numeric, errors="coerce").dropna()
        df_clean = df_clean.sort_values("TWT").drop_duplicates("TWT")

        # Extrai os arrays
        twt = df_clean["TWT"].values
        tvdss = df_clean["TVDSS"].values

        # OWT para TWT (se necessário)
        if np.mean(twt) < 1:  # provavelmente OWT
            twt = 2 * twt  # converte para TWT

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