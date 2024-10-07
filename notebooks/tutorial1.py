import numpy as np
import pandas as pd
import segyio
import lasio

from pathlib import Path
from pprint import pprint

from wtie import grid, viz
from wtie.processing.logs import interpolate_nans, despike

import matplotlib.pyplot as plt

def import_logs(file_path: str) -> grid.LogSet:
    file_path = Path(file_path)
    assert file_path.name == 'volve_159-19A_LFP.las'

    # Read file
    las_logs = lasio.read(file_path)
    las_logs = las_logs.df()

    # Select some logs, there are more, we only load the follwoing
    # must at least contain the keys 'Vp' for acoustic velocity
    # and 'Rho' for the bulk density. 'Vs', for shear velocity, must also
    # be imported if one whishes to perform a prestack well-tie.
    # Other logs are optional.
    log_dict = {}


    log_dict['Vp'] = grid.Log(las_logs.LFP_VP.values, las_logs.LFP_VP.index, 'md', name='Vp')
    log_dict['Vs'] = grid.Log(las_logs.LFP_VS.values, las_logs.LFP_VS.index, 'md', name='Vs')

    # Density contains some NaNs, I fill them with linear interpolation.
    log_dict['Rho'] = grid.Log(interpolate_nans(las_logs.LFP_RHOB.values), las_logs.LFP_RHOB.index, 'md', name='Rho')

    # The gamma ray has an ouylying value, probably best to remove it
    log_dict['GR'] = grid.Log(despike(interpolate_nans(las_logs.LFP_GR.values)), las_logs.LFP_GR.index, 'md', name='GR', unit="API")


    log_dict['Cali'] = grid.Log(las_logs.LFP_CALI.values, las_logs.LFP_CALI.index, 'md', name='Cali')

    return grid.LogSet(log_dict)

def import_seismic(file_path: str) -> grid.Seismic:
    file_path = Path(file_path)
    assert file_path.name == 'volve_15_9_19A_gather.sgy'

    with segyio.open(file_path, 'r') as f:
        _twt = f.samples / 1000 # two-way-time in seconds
        _seis = np.squeeze(segyio.tools.cube(f)) # 2D (angles, samples)
        
    # stacking the first 8 angles
    _seis = np.sum(_seis[:8,:], axis=0)

    return grid.Seismic(_seis, _twt, 'twt', name='Real seismic')


def import_prestack_seismic(file_path: str) -> grid.PreStackSeismic:
    """For simplicity, only angle gathers are allowed."""
    file_path = Path(file_path)
    assert file_path.name == 'volve_15_9_19A_gather.sgy'

    with segyio.open(file_path, 'r') as f:
        _twt = f.samples / 1000
        _seis = np.squeeze(segyio.tools.cube(f))
        _angles = f.offsets

    seismic = []
    for i, theta in enumerate(_angles):
        seismic.append(grid.Seismic(_seis[i,:], _twt, 'twt', theta=theta))

    return grid.PreStackSeismic(seismic, name='Real gather')

def import_well_path(file_path: str) -> grid.WellPath:
    file_path = Path(file_path)
    assert file_path.name == 'volve_path_15_9-19_A.txt'

    _wp = pd.read_csv(file_path, header=None, delimiter=r"\s+",
                      names=('MD (kb) [m]', 'Inclination', 'Azimuth'))

    kb = 25 # meters

    _tvd = grid.WellPath.get_tvdkb_from_inclination(\
                                            _wp.loc[:,'MD (kb) [m]'].values,
                                            _wp.loc[:,'Inclination'].values[:-1]
                                                    )
    _tvd = grid.WellPath.tvdkb_to_tvdss(_tvd, kb)

    return grid.WellPath(md=_wp.loc[:,'MD (kb) [m]'].values, tvdss=_tvd, kb=kb)

def import_time_depth_table(file_path: str) -> grid.TimeDepthTable:
    file_path = Path(file_path)
    assert file_path.name == 'volve_checkshot_15_9_19A.txt'

    _td = pd.read_csv(file_path, header=None, delimiter=r"\s+", skiprows=[0],
                  names=('Curve Name', 'TVDBTDD', 'TVDKB', 'TVDSS', 'TWT'))

    _twt = _td.loc[:,'TWT'].values / 1000 # seconds
    _tvdss = np.abs(_td.loc[:,'TVDSS'].values) # meters

    return grid.TimeDepthTable(twt=_twt, tvdss=_tvdss)

# data path
folder = Path('data/tutorial/Volve')
trajectory_path = folder / 'volve_path_15_9-19_A.txt'
table_path = folder / 'volve_checkshot_15_9_19A.txt'
logs_path = folder / 'volve_159-19A_LFP.las'
seis_path = folder / 'volve_15_9_19A_gather.sgy'

assert folder.exists()

# Data objects
pprint(grid.EXISTING_BASIS_TYPES)

# WELL LOGS

# Lasio
las_logs = lasio.read(logs_path, engine = "normal")
las_logs = las_logs.df()
las_logs.head(4)
print(las_logs.head(4))

# Matplotlib
basis = las_logs.LFP_VP.index
vp = las_logs.LFP_VP.values

fig,ax = plt.subplots()
ax.plot(basis, vp)

Vp = grid.Log(las_logs.LFP_VP.values, las_logs.LFP_VP.index, 'md', name='Vp')
# viz.plot_trace(Vp);

logset_md = import_logs(logs_path) # md is for measured depth

viz.plot_logset(logset_md);

# SEISMIC

# segyo
with segyio.open(seis_path, 'r') as f:
    print(f.samples.size) # number of time samples
    print(f.ilines) 
    print(f.xlines)
    print(f.offsets) # these are actually angles, from 0 to 45 degrees

seismic = import_seismic(seis_path)
gather = import_prestack_seismic(seis_path)
# viz.plot_trace(seismic)
# viz.plot_prestack_trace_as_pixels(gather, figsize=(7,9));

# WELL TRAJECTORY

print(grid.WellPath.__doc__)
print(grid.WellPath.__init__.__doc__)

wellpath = import_well_path(trajectory_path)
# viz.plot_wellpath(wellpath);

print(grid.TimeDepthTable.__doc__)
print(grid.TimeDepthTable.__init__.__doc__)

# Time-Depth relation table
print(grid.TimeDepthTable.__doc__)
print(grid.TimeDepthTable.__init__.__doc__)

td_table = import_time_depth_table(table_path)
# viz.plot_td_table(td_table);

plt.show()