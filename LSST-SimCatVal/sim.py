from IPython import get_ipython
if get_ipython().__class__.__name__ == "ZMQInteractiveShell":
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm

from astropy.table import Table
import galsim
from utils import get_wcs, get_psf, get_noise, WORLD_ORIGIN
from skycat import SkyCat
from diffcat import DiffCat
from afw_utils import create_afw
from joblib import Parallel, delayed

def process_object_joblib(idx, band, cat, psf, pointing, wcs, nim, coadd_zp, seed_base):
    rng = galsim.BaseDeviate(seed=seed_base + idx)
    gal, dx, dy, obj_info= cat.get_obj(idx, band, coadd_zp)

    if gal is None:
        return None, None
    stamp, image_pos = get_stamp(gal, psf, pointing, dx, dy, rng, cat, band, wcs, nim, 'diff_gal')
    return stamp, obj_info, image_pos

def make_sim(
    skycat_path,
    ra, 
    dec,
    img_size,
    buffer,
    config_dic,
    coadd_zp,
    diff_path=None,
    diff_ra=None,
    diff_dec=None,
    n_jobs=1
):
    pointing = galsim.CelestialCoord(ra=ra * galsim.degrees,dec=dec * galsim.degrees)
    wcs = get_wcs(img_size, pointing)
    skycat = SkyCat(skycat_path, pointing, img_size, buffer, wcs)

    images_afw = {}
    truths = {}
    images_save = {'ra': ra, 'dec':dec}
    images_save['diff_ra'] = diff_ra
    images_save['diff_dec'] = diff_dec
    for band in tqdm(config_dic.keys(), mininterval=10):
        psf_m2r = config_dic[band]['psf_radius']
        sigma = config_dic[band]['sigma']
        nim = config_dic[band]['n_images']
        e1 = config_dic[band]['psf_e1']
        e2 = config_dic[band]['psf_e2']
        psf = get_psf(psf_m2r, e1, e2)
        
        noise_img = galsim.Image(img_size, img_size, wcs=wcs)
        rng_galsim = galsim.BaseDeviate()
        noise = get_noise(rng_galsim, sigma)
        noise_img.addNoise(noise)

        final_img = galsim.Image(img_size, img_size, wcs=wcs)
        truth = []
        
        if diff_path is not None:
            diff_pointing = galsim.CelestialCoord(ra=diff_ra * galsim.degrees,dec=diff_dec * galsim.degrees)
            diff_wcs = get_wcs(img_size, diff_pointing)
            diffcat = DiffCat(diff_path, diff_pointing, img_size, buffer, diff_wcs)

            # for h in tqdm(range(diffcat.get_n()),mininterval=20, miniters=600):
            #     gal, dx, dy, obj_info= diffcat.get_obj(h, band, coadd_zp)
            #     stamp = get_stamp(gal, psf, pointing, dx, dy, rng_galsim, skycat, band, wcs, nim, 'diff_gal')
            #     b = stamp.bounds & final_img.bounds  
            #     if b.isDefined():
            #         final_img[b] += stamp[b]
            #         truth.append(obj_info)

            results = Parallel(n_jobs=n_jobs)(
                delayed(process_object_joblib)(idx, band, diffcat, psf, pointing, wcs, nim, coadd_zp, 12345
                ) for idx in tqdm(range(diffcat.get_n()),mininterval=20, miniters=600))

            for stamp, obj_info, image_pos in results:
                b = stamp.bounds & final_img.bounds  
                if b.isDefined():
                    final_img[b] += stamp[b]
                    if (image_pos.x >= 0) & (image_pos.x <= img_size) | (image_pos.y >= 0) | (image_pos.y <= img_size):
                        obj_info.append('True')
                    else:
                        obj_info.append('False')
                    truth.append(obj_info)

            ob_types = ['star']
        else:
            ob_types = ['star', 'galaxy']

        for t in ob_types:
            for g in tqdm(range(skycat.get_n(t)),mininterval=20, miniters=600):
                gal, dx, dy, obj_info = skycat.get_obj(t, g, band,coadd_zp)
                if gal is None:
                    continue
                stamp, _ = get_stamp(gal, psf, pointing, dx, dy, rng_galsim, skycat, band, wcs, nim, t)
                b = stamp.bounds & final_img.bounds
                if b.isDefined():
                    final_img[b] += stamp[b]
                    obj_info.append('True')
                    truth.append(obj_info)
        print("making_final",flush=True)
        final_img += noise_img
        print('making psf',flush=True)
        psf_im = psf.drawImage(nx=41,ny=41,scale=0.2, dtype=float)
        print('making afw',flush=True)
        afw_im = create_afw(final_img, wcs, band, psf_im, sigma, coadd_zp)
        images_afw[band] = afw_im
        print('making save',flush=True)
        images_save[band] = {'afw_image':afw_im, 'psf':psf_m2r, 'sigma':sigma, 'n_images':nim}
        print('making truth',flush=True)
        truths[band] = (Table(rows=truth, names=('ra', 'dec','flux','redshift','ob_type','in_img')))
    return images_afw, truths, images_save

def get_stamp(
    gal, 
    psf, 
    pointing,
    dx, 
    dy, 
    rng_galsim, 
    skycat, 
    band, 
    wcs,
    nim,
    star
):
    obj = galsim.Convolve(gal, psf)
    world_pos = pointing.deproject(dx * galsim.arcsec, dy * galsim.arcsec,)
    image_pos = wcs.toImage(world_pos)
    n_photons = galsim.PoissonDeviate(rng_galsim, mean=gal.flux)()

    n_photons = n_photons * nim #for correct poisson noise
    if n_photons >= 5e7:
        n_photons = 5e7

    if star=='star' and n_photons >= 5e5:
        stamp_size = 300
        if n_photons >= 5e7:
            n_photons = 5e7
            stamp_size = 600

        stamp = obj.drawImage(
            nx=stamp_size,
            ny=stamp_size,
            bandpass=skycat.bands[band], 
            wcs=wcs.local(world_pos=world_pos), 
            method='phot',
            n_photons=n_photons,
            maxN=int(1e7),
            rng=rng_galsim)
        stamp.setCenter(image_pos.x, image_pos.y)

    elif star == 'diff_gal':
        stamp = obj.drawImage(
            # nx=stamp_size, # this auto chooses a size
            # ny=stamp_size,
            wcs=wcs.local(world_pos=world_pos), 
            method='phot',
            n_photons=n_photons,
            maxN=int(1e7),
            rng=rng_galsim)
        stamp.setCenter(image_pos.x, image_pos.y)

    else:
        stamp = obj.drawImage(
            # nx=stamp_size, # this auto chooses a size
            # ny=stamp_size,
            bandpass=skycat.bands[band], 
            wcs=wcs.local(world_pos=world_pos), 
            method='phot',
            n_photons=n_photons,
            maxN=int(1e7),
            rng=rng_galsim)
        stamp.setCenter(image_pos.x, image_pos.y)
    return stamp, image_pos
