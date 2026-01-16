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