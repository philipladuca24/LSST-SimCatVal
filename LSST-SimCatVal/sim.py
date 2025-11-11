from IPython import get_ipython
if get_ipython().__class__.__name__ == "ZMQInteractiveShell":
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm

import numpy as np
import galsim
from .utils import get_wcs, get_psf, get_noise, WORLD_ORIGIN
from .skycat import SkyCat
from .afw_utils import create_afw

def make_sim(
    skycat_path,
    img_size,
    bands,
    buffer,
    psf_fwhm,
    sigma,
    coadd_zp
):
    bands = [b for b in bands]
    wcs = get_wcs(img_size)
    skycat = SkyCat(img_size, buffer, wcs)
    images = []
    for band in tqdm(bands):
        psf = get_psf(psf_fwhm)
        
        noise_img = galsim.Image(img_size, img_size, wcs=wcs)
        rng_galsim = galsim.BaseDeviate(1)
        noise = get_noise(rng_galsim, sigma)
        noise_img.addNoise(noise)

        final_img = galsim.Image(img_size, img_size, wcs=wcs)
        for t in ['galaxy', 'star']:
            for g in tqdm(range(skycat.get_n(t))):
                gal, dx, dy = skycat.get_obj(t, g, band,coadd_zp)
                stamp = get_stamp(gal, psf, dx, dy, rng_galsim, skycat, band, wcs)
                b = stamp.bounds & final_img.bounds
                if b.isDefined():
                    final_img[b] += stamp[b]
        final_img += noise_img
        afw_im = create_afw(final_img, wcs, band, psf_fwhm, sigma, coadd_zp)
        images.append(afw_im)
    return images

def get_stamp(
    gal, 
    psf, 
    dx, 
    dy, 
    rng_galsim, 
    skycat, 
    band, 
    wcs
):
    obj = galsim.Convolve(gal, psf)
    world_pos = WORLD_ORIGIN.deproject(dx * galsim.arcsec, dy * galsim.arcsec,)
    image_pos = wcs.toImage(world_pos)
    stamp_size = 150
    maxN = int(1e6)
    n_photons = galsim.PoissonDeviate(rng_galsim, mean=gal.flux)()

    #for correct poisson noise
    # n_photons = n_photons * 60#num_exposures

    stamp = obj.drawImage(nx=stamp_size, 
                            ny=stamp_size, 
                            bandpass=skycat.bands[band], 
                            wcs=wcs.local(world_pos=world_pos), 
                            method='phot',
                            center=image_pos,
                            n_photons=n_photons,
                            maxN=maxN,
                            rng=rng_galsim)
    return stamp
