import os
import sys
import numpy as np
from astropy.table import hstack
import astropy.units as u
from astropy.coordinates import SkyCoord, match_coordinates_sky
from sim import make_sim
from afw_utils import run_lsst_pipe_single,run_lsst_pipe_multi, COLUMNS
from utils import sample_position, sample_diff_position
from IPython import get_ipython
from datetime import datetime
import pickle
if get_ipython().__class__.__name__ == "ZMQInteractiveShell":
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm

if __name__ == "__main__":
    folder = sys.argv[1]
    n_jobs = int(sys.argv[2])

    with open(f"{folder}/ECDFS_sim_im.pkl", 'rb') as f:
        afw_dic = pickle.load(f)

    # print('Pipeline Forced',flush=True)
    # lsst_cat = run_lsst_pipe_single(afw_dic, True, n_jobs)

    # with open(f'{folder}/ECDFS_sim_meas_forced.pkl', "wb") as f:
    #     pickle.dump(lsst_cat, f)

    print('Pipeline Scarlet',flush=True)
    cats_f = run_lsst_pipe_multi([b for b in 'ugrizy'], [afw_dic[i]['afw_image'] for i in 'ugrizy'], n_jobs)

    print('Saving outputs', flush=True)

    with open(f'{folder}/ECDFS_sim_meas_forced_s.pkl', "wb") as f:
        pickle.dump(cats_f, f)
