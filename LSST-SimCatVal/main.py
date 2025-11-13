import os
import numpy as np
from astropy.table import hstack
import astropy.units as u
from astropy.coordinates import SkyCoord, match_coordinates_sky
from sim import make_sim
from afw_utils import run_lsst_pipe, COLUMNS
from IPython import get_ipython
if get_ipython().__class__.__name__ == "ZMQInteractiveShell":
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm


def SimCatVal(
    skycat_path,
    img_size,
    bands,
    buffer,
    psf_fwhm,
    sigma,
    coadd_zp,
    deblend
):
    assert os.path.isfile(skycat_path)

    print('Generating Sims')
    afw_img, truths = make_sim(skycat_path, img_size, bands, buffer, psf_fwhm, sigma, coadd_zp)

    print('Running Pipeline')
    matches = []
    cats = []
    for i in tqdm(range(len(bands))):
        lsst_cat = run_lsst_pipe(afw_img[i], deblend)

        primary = ((np.isnan(lsst_cat["modelfit_CModel_instFlux"]) == False) & 
        (lsst_cat["modelfit_CModel_instFlux"]/lsst_cat["modelfit_CModel_instFluxErr"] > 0) &
        (lsst_cat["modelfit_CModel_flag"] == 0))
                
        if (deblend == True):
            primary *= lsst_cat["deblend_nChild"] == 0

        cat = lsst_cat[primary]
        cat['mag'] = -2.5*np.log10(cat['modelfit_CModel_instFlux']) + coadd_zp
        cat['snr'] = cat['modelfit_CModel_instFlux'] / cat['modelfit_CModel_instFluxErr']

        cats.append(cat)
        
        print(cat['coord_ra'])

        ob_coord = SkyCoord(ra=cat['coord_ra'], dec=cat['coord_dec']) #what is happening here
        true_coord = SkyCoord(ra=truths[i]['ra']*u.degree, dec=truths[i]['dec']*u.degree)
        idx, d2d, d3d = match_coordinates_sky(ob_coord, true_coord)
        max_sep = 0.25 * u.arcsec
        sep_constraint = d2d < max_sep
        ob_matches = cat[sep_constraint]
        truth_matches = truths[i][idx[sep_constraint]]
        match = hstack([ob_matches,truth_matches])
        matches.append(match)


    area = (img_size * 0.2 /60)**2
    
    
    # image of simulation with fitted shapes
    # difference between true magnitude and measured magnitude as a function of magnitude
    # true and measured normalized luminosity functions
    # % match between true and measured catalogs

    return afw_img, cats, matches, area
