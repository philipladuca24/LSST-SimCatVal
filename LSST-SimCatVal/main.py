import os
import numpy as np
from astropy.table import hstack
import astropy.units as u
from astropy.coordinates import SkyCoord, match_coordinates_sky
from sim import make_sim
from afw_utils import run_lsst_pipe_single, COLUMNS
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
    # bands=None,
    # psf_fwhm=None,
    # sigma=None,
    coadd_zp,
    deblend
):
    assert os.path.isfile(skycat_path)
  
    print('Generating Sims') #this should take in info dict which is ouput of sampler, also need to change world origin
    afw_dic, truths, npy_dic = make_sim(skycat_path, ra, dec, img_size, buffer, config_dic, coadd_zp)
    
    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S") #this needs to change
    file_path = f'/hildafs/home/pladuca/main/lsst-sim-package/LSST-SimCatVal/LSST-SimCatVal/outputs/run_{timestamp}'
    os.makedirs(file_path, exist_ok=True)
    with open(f'{file_path}/ECDFS_sim_im.pkl', "wb") as f:
        pickle.dump(npy_dic, f)
    with open(f'{file_path}/ECDFS_sim_truth.pkl', "wb") as f:
        pickle.dump(truths, f)

    print('Running Pipeline')
    matches = []
    cats = {}
    for band in tqdm(config_dic.keys()):
        lsst_cat = run_lsst_pipe_single(afw_dic[band], deblend)

        # primary = ((np.isnan(lsst_cat["modelfit_CModel_instFlux"]) == False) & 
        # (lsst_cat["modelfit_CModel_instFlux"]/lsst_cat["modelfit_CModel_instFluxErr"] > 0) &
        # (lsst_cat["modelfit_CModel_flag"] == 0))
                
        # if (deblend == True):
        #     primary *= lsst_cat["deblend_nChild"] == 0

        # cat = lsst_cat[primary]
        cat = lsst_cat
        cat['mag'] = -2.5*np.log10(cat['modelfit_CModel_instFlux']) + coadd_zp
        cat['snr'] = cat['modelfit_CModel_instFlux'] / cat['modelfit_CModel_instFluxErr']

        cats[band] = cat

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

    area = (img_size * 0.2 /60)**2
    with open(f'{file_path}/ECDFS_sim_meas_single.pkl', "wb") as f:
        pickle.dump(cats, f)
    print("Done!")
    
    return afw_dic, cats, truths, area
