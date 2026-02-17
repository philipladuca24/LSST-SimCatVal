import os
import sys
import numpy as np
from astropy.table import hstack
import astropy.units as u
from astropy.coordinates import SkyCoord, match_coordinates_sky
from sim import make_sim
from afw_utils import run_lsst_pipe_single,run_lsst_pipe_multi, COLUMNS, PHOT_COLUMNS
from utils import sample_position, sample_diff_position
from IPython import get_ipython
from datetime import datetime
import pickle
from scipy.spatial import cKDTree
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
    afw_dic, truths, npy_dic = make_sim(skycat_path, ra, dec, img_size, buffer, config_dic, coadd_zp, ind, diff_path, diff_ra, diff_dec, n_jobs)

    for b in config_dic.keys():
        truths[b]['obs_ind'] = ind

    print('Running Pipeline',flush=True)

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

        if save is not None:
            with open(f'{file_path}/ECDFS_sim_meas_single.pkl', "wb") as f:
                pickle.dump(cats, f)
    else:
        if forced == 2:
            print('Pipeline Forced',flush=True)
            lsst_cat = run_lsst_pipe_single(afw_dic, forced, n_jobs)
            lsst_cat = processed_forced(lsst_cat, [b for b in config_dic.keys()], ind)

            if save is not None:
                with open(f'{file_path}/ECDFS_sim_meas_forced.pkl', "wb") as f:
                    pickle.dump(lsst_cat, f)

        print(datetime.now(),flush=True)
        print('Pipeline Scarlet',flush=True)
        cats_f = run_lsst_pipe_multi([b for b in config_dic.keys()], [afw_dic[i] for i in config_dic.keys()], n_jobs)
        cats_f = processed_forced(cats_f, [b for b in config_dic.keys()], ind, img_size)
        match_mask, match_id, t_mcut = match_true(cats_f, truths)
        cats_f['match'] = match_mask
        cats_f['match_id'] = match_id
        cats_f['match_redshfit'] = truths['i'][t_mcut][match_id]['redshift']
        cats_f['match_flux_i'] = truths['i'][t_mcut][match_id]['flux']
        
        print('Saving outputs', flush=True)
        if save is not None:
            with open(f'{file_path}/ECDFS_sim_meas_forced_s.pkl', "wb") as f:
                pickle.dump(cats_f, f)
        cats = (lsst_cat, cats_f)
    print("Done!")
    area = ((img_size - 50) * 0.2 /60)**2
    return afw_dic, npy_dic, cats, truths, area

def processed_forced(cat, bands, ind, img_size):
    cat_to_stack = []
    for b in bands:
        if b == 'i':
            temp = cat[b][COLUMNS]
            temp['obs_ind'] = ind
            temp.rename_columns(PHOT_COLUMNS, [n + f'_{b}' for n in PHOT_COLUMNS])
            if ((temp['base_SdssCentroid_x'] >= 25) & (temp['base_SdssCentroid_x'] <= img_size - 25) & (temp['base_SdssCentroid_y'] >= 25) & (temp['base_SdssCentroid_y'] <= img_size - 25)):
                temp['in_img'].append(True)
            else:
                temp['in_img'].append(False)
        else:
            temp = cat[b][PHOT_COLUMNS]
            temp.rename_columns(PHOT_COLUMNS, [n + f'_{b}' for n in PHOT_COLUMNS])
        cat_to_stack.append(temp)
    return hstack(cat_to_stack)

def match_true(cat, truths, coadd_zp):
    base_cuts = ((cat['deblend_nChild'] == 0) &
        (cat['base_SdssShape_flag'] == False) &
        (cat['modelfit_CModel_instFlux'] >= 0) &
        (cat['modelfit_CModel_flag'] == False) &
        (cat['base_SdssCentroid_flag'] == False))
    mags = -2.5*np.log10(cat['modelfit_CModel_instFlux']) + coadd_zp
    mag_lim = np.percentile(mags[base_cuts], 95) + 1 ## arbitrary, if a mag estimate is off by more than 2 bad, use 1.5?
    t_mcut = (-2.5*np.log10(truths['i']['flux']) + coadd_zp) < mag_lim
    tru = truths['i'][t_mcut]
    ob_pix = np.column_stack([cat['base_SdssCentroid_x'],cat['base_SdssCentroid_y']])
    true_pix = np.column_stack([tru['x'],tru['y']])

    tree = cKDTree(true_pix)
    dist, idx = tree.query(ob_pix, k=1)
    dist = np.array(dist)
    idx = np.array(idx)
    sep_mask = dist < 5

    return sep_mask, idx, t_mcut 


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
        _ = SimCatVal(skycat_path,ra,dec,im_size,25,sample,31.4,ind,forced,save_path,diff_path,diff_ra,diff_dec,n_jobs,save)
    else:    
        print('Running OU')
        _ = SimCatVal(skycat_path,ra,dec,im_size,25,sample,31.4,ind,forced,save_path,save=save)

