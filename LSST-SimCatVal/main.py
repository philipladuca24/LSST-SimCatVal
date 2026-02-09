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


def SimCatVal(
    skycat_path,
    ra,
    dec,
    img_size,
    buffer,
    config_dic,
    coadd_zp,
    ind,
    forced,
    save_path,
    diff_path=None,
    diff_ra=None,
    diff_dec=None,
    n_jobs=None,
    save=None,

):
    assert os.path.isfile(skycat_path)

    print(datetime.now(),flush=True)
    print('Generating Sims',flush=True) #this should take in info dict which is ouput of sampler, also need to change world origin
    afw_dic, truths, npy_dic = make_sim(skycat_path, ra, dec, img_size, buffer, config_dic, coadd_zp, diff_path, diff_ra, diff_dec, n_jobs)

    # return afw_dic, truths, npy_dic

    print('Running Pipeline',flush=True)
    # matches = []

    if save is not None:
        file_path = f'{save_path}/run_{save}_{ind}'
        print(file_path)
        os.makedirs(file_path, exist_ok=True)
        with open(f'{file_path}/ECDFS_sim_im.pkl', "wb") as f:
            pickle.dump(npy_dic, f)
        with open(f'{file_path}/ECDFS_sim_truth.pkl', "wb") as f:
            pickle.dump(truths, f)

    if forced == 0:
        cats = {}
        for band in tqdm(config_dic.keys()):
            lsst_cat = run_lsst_pipe_single(afw_dic[band])
            cats[band] = lsst_cat

            #matching move to utils?
            # ob_coord = SkyCoord(ra=cat['coord_ra'], dec=cat['coord_dec'])
            # true_coord = SkyCoord(ra=truths[band]['ra']*u.degree, dec=truths[band]['dec']*u.degree)
            # idx, d2d, d3d = match_coordinates_sky(ob_coord, true_coord)
            # max_sep = 0.5 * u.arcsec
            # sep_constraint = d2d < max_sep
            # ob_matches = cat[sep_constraint]
            # truth_matches = truths[band][idx[sep_constraint]]
            # match = hstack([ob_matches,truth_matches])
            # matches.append(match)
        if save is not None:
            with open(f'{file_path}/ECDFS_sim_meas_single.pkl', "wb") as f:
                pickle.dump(cats, f)
    else:
        print('Pipeline Forced',flush=True)
        lsst_cat = run_lsst_pipe_single(afw_dic, forced, n_jobs)

        if save is not None:
            with open(f'{file_path}/ECDFS_sim_meas_forced.pkl', "wb") as f:
                pickle.dump(lsst_cat, f)

        print(datetime.now(),flush=True)
        print('Pipeline Scarlet',flush=True)
        cats_f = run_lsst_pipe_multi([b for b in config_dic.keys()], [afw_dic[i] for i in config_dic.keys()], n_jobs)

        print('Saving outputs', flush=True)
        area = (img_size * 0.2 /60)**2
        if save is not None:
            with open(f'{file_path}/ECDFS_sim_meas_forced_s.pkl', "wb") as f:
                pickle.dump(cats_f, f)
        cats = (lsst_cat, cats_f)
    print("Done!")
    
    return afw_dic, npy_dic, cats, truths, area


if __name__ == "__main__":
    ind = int(sys.argv[1])
    ind_max = int(sys.argv[2])
    skycat_path = sys.argv[3]
    Dp1_sample = sys.argv[4]
    im_size = int(sys.argv[5])
    forced = int(sys.argv[6])
    save_path = sys.argv[7]
    save = sys.argv[8]
    print(sys.argv)
    with open(Dp1_sample, 'rb') as f:
        rsp_sample = pickle.load(f)

    position = sample_position(ind_max, 333)
    ra = position[ind-1][0]
    dec = position[ind-1][1]
    sample = rsp_sample[ind-1].copy()
    sample.pop('ra')
    sample.pop('dec')
    # sample = {'i':sample['i']} ######### temp maybe add this so you dont have to do all bands?

    if len(sys.argv) > 9:
        diff_path = sys.argv[9]
        n_jobs = int(sys.argv[10])
        diff_position = sample_diff_position(ind_max, 333, diff_path)
        diff_ra = diff_position[ind-1][0]
        diff_dec = diff_position[ind-1][1]
        print('Running diff')
        _ = SimCatVal(skycat_path,ra,dec,im_size,50,sample,31.4,ind,forced,save_path,diff_path,diff_ra,diff_dec,n_jobs,save)
    else:    
        print('Running OU')
        _ = SimCatVal(skycat_path,ra,dec,im_size,50,sample,31.4,ind,forced,save_path,save=save)

