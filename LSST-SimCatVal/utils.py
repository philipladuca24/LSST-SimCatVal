import galsim

"""Constants"""
PIXEL_SCALE = 0.2
COLLECTING_AREA = 6.423**2 * 10000# value from https://smtn-002.lsst.io/ 
MJD = 61937.090308
WORLD_ORIGIN = galsim.CelestialCoord(
    ra=10.161290322580646 * galsim.degrees,
    dec=-43.40685848593699 * galsim.degrees,
)
LSST_BANDS = 'ugrizy'

def get_sat_vals(zp): # from desc shear package
    return {
    'g': 140000 * 10.0**(0.4*(zp-32.325)),  # from example images
    'r': 140000 * 10.0**(0.4*(zp-32.16)),
    'i': 140000 * 10.0**(0.4*(zp-31.825)),
    'z': 140000 * 10.0**(0.4*(zp-31.50)),
    'y': 140000 * 10.0**(0.4*(zp-31.20)),  # extrapolate from riz
}

def get_wcs(img_size):
    return galsim.TanWCS(
        affine=galsim.AffineTransform(
            PIXEL_SCALE, 0, 0, PIXEL_SCALE,
            origin=galsim.PositionD(img_size / 2, img_size / 2), # I need to know what Im considering the center, 
        ),                                                       # since I use 0,0 as corner when sampling skycat
        world_origin=WORLD_ORIGIN,
        units=galsim.arcsec,
    )

def get_psf(fwhm=2.6*0.2): #fwhm preliminary from DP1 2.6*0.2
    return galsim.Kolmogorov(fwhm=fwhm,scale_unit=galsim.arcsec)


def get_noise(galsim_rng,sigma=0.1):
    return galsim.GaussianNoise(rng=galsim_rng, sigma=sigma)

def get_bandpasses():
    lsst_bandpasses = dict()
    for band in LSST_BANDS:
        # bp_full_path = f'LSST_{band}.dat'
        # bp_full_path = f'/lsst-sim-package/LSST-SimCatVal/LSST-SimCatVal/bandpasses/total_{band}.dat'
        bp_full_path = f'/hildafs/home/pladuca/main/rubin_sim_data/throughputs/baseline/total_{band}.dat'
        bp = galsim.Bandpass(bp_full_path, 'nm')
        bp = bp.truncate(relative_throughput=1.e-3)
        # Remove wavelength values selectively for improved speed but
        # preserve flux integrals.
        bp = bp.thin()
        bp = bp.withZeropoint('AB') # this uses exptime and collecting area to get ~30 zp for real observations
        lsst_bandpasses[band] = bp
    return lsst_bandpasses

def get_flux(obj, band):
    return obj.get_native_attribute(f'lsst_flux_{band}')


def convert_flux(flux, band, coadd_zp):
    zp = band.zeropoint
    return flux * 10 ** (0.4 * (coadd_zp - zp))
