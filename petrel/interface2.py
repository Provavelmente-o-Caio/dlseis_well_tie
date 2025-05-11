import segyio
import lasio
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import sys

def interface():
    print(len(sys.argv))
    if len(sys.argv) == 5:
        _, logs, seis, path, table = sys.argv
        data_path = [logs, seis, path, table]

        for i in range(len(data_path)):
            data_path[i] = Path(data_path[i])
            print(data_path[i])

        las_logs = lasio.read(data_path[0])
        basis = las_logs.LFP_VP.index
        vp = las_logs.LFP_VP.values
        las_logs = las_logs.df()
        print(las_logs.head(4))
        plt.show()
    else:
        return
        

if __name__ == "__main__":
    interface()