from basic_well_tie import Basic_well_tie
import sys

print("AAAA")

if len(sys.argv) >= 7:
    _, log_path, seismic_path, wellpath_path, td_table_path, data_path, config_path = sys.argv

    wt = Basic_well_tie(log_path, seismic_path, wellpath_path, td_table_path, data_path, config_path)
    wt.well_tie()
else:
    print("Argumentos insuficientes.")