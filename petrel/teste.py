from basic_well_tie import Basic_well_tie
import sys

if len(sys.argv) == 5:
    _, log_path, seismic_path, wellpath_path, td_table_path = sys.argv

    wt = Basic_well_tie(log_path, seismic_path, wellpath_path, td_table_path)
    wt.well_tie()
else:
    print("Argumentos insuficientes.")