import os
import numpy as np
from .sim import make_sim
from .afw_utils import run_lsst_pipe, COLUMNS

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

    afw_img = make_sim(skycat_path, img_size, bands, buffer, psf_fwhm, sigma, coadd_zp)

    lsst_cat = run_lsst_pipe(afw_img, deblend)

    primary = ((np.isnan(lsst_cat["slot_ModelFlux_instFlux"]) == False) & 
    (lsst_cat["slot_ModelFlux_instFlux"]/lsst_cat["slot_ModelFlux_instFluxErr"] > 1) &
    (lsst_cat["modelfit_CModel_flag"] == 0))
               
    if (deblend == True):
        primary *= lsst_cat["deblend_nChild"] == 0

    cat = lsst_cat[primary]

    area = (img_size * 0.2 /60)**2
    cat['mag'] = -2.5*np.log10(cat['modelfit_CModel_instFlux']) + coadd_zp
    cat['snr'] = cat['modelfit_CModel_instFlux'] / cat['modelfit_CModel_instFluxErr']

    return cat, area
