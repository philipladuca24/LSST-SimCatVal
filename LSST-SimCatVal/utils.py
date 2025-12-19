import galsim
import healpy as hp
import numpy as np

"""Constants"""
PIXEL_SCALE = 0.2
COLLECTING_AREA = 6.423**2 * 10000# value from https://smtn-002.lsst.io/ 
MJD = 61937.090308
# WORLD_ORIGIN = galsim.CelestialCoord(
#     ra=10.161290322580646 * galsim.degrees,
#     dec=-43.40685848593699 * galsim.degrees,
# )
WORLD_ORIGIN = galsim.CelestialCoord(
    ra=10.68 * galsim.degrees,
    dec=-43.67 * galsim.degrees,
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

def get_wcs(img_size, point):
    return galsim.TanWCS(
        affine=galsim.AffineTransform(
            PIXEL_SCALE, 0, 0, PIXEL_SCALE,
            origin=galsim.PositionD(img_size // 2, img_size // 2), 
        ),                                                       
        world_origin=point,
        units=galsim.arcsec,
    )

def get_psf(fwhm=2.6): #fwhm preliminary from DP1 2.6*0.2
    fwhm = fwhm * 0.2 #convert pixels to arcsec
    return galsim.Kolmogorov(fwhm=fwhm,scale_unit=galsim.arcsec)


def get_noise(galsim_rng,sigma=0.1):
    return galsim.GaussianNoise(rng=galsim_rng, sigma=sigma)

def get_bandpasses():
    lsst_bandpasses = dict()
    for band in LSST_BANDS:
        # bp_full_path = f'LSST_{band}.dat'
        bp_full_path = f'bandpasses/total_{band}.dat' #v1.7
        # bp_full_path = f'/hildafs/home/pladuca/main/rubin_sim_data/throughputs/baseline/total_{band}.dat' #v1.9
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

def is_point_in_polygon(ra_poly, dec_poly, px, py):
    vertices = np.array(list(zip(ra_poly, dec_poly)))
    n = len(vertices)

    px_flat = px.flatten()
    py_flat = py.flatten()
    inside = np.ones_like(px_flat, dtype=bool)

    for i in range(n):
        A = vertices[i]
        B = vertices[(i+1) % n]
        cross = (B[0] - A[0]) * (py_flat - A[1]) - (B[1] - A[1]) * (px_flat - A[0])
        inside &= (cross >= 0)  # for CCW polygon

    return inside.reshape(px.shape)

def unit(v):
    return v / np.linalg.norm(v)

def offset_polygon_ra_dec(ra, dec, offset):
    vertices = np.array(list(zip(ra, dec)))
    n = len(vertices)
    new_edges = []

    for i in range(n):
        A = vertices[i]
        B = vertices[(i+1) % n]
        edge = B - A
        normal = np.array([-edge[1], edge[0]])
        normal = unit(normal)

        A_off = A + normal * offset
        B_off = B + normal * offset
        new_edges.append((A_off, B_off))

    def intersect(L1, L2):
        p1, p2 = L1
        q1, q2 = L2
        A_mat = np.array([p2 - p1, q1 - q2]).T
        t = np.linalg.solve(A_mat, q1 - p1)[0]
        return p1 + t * (p2 - p1)

    new_vertices = []
    for i in range(n):
        L1 = new_edges[i]
        L2 = new_edges[(i+1) % n]
        new_vertices.append(intersect(L1, L2))

    new_vertices = np.array(new_vertices)
    ra_new = new_vertices[:, 0].tolist()
    dec_new = new_vertices[:, 1].tolist()

    return ra_new, dec_new

def sample_position(count, inp):
    np.random.seed(inp)
    theta, phi = hp.pix2ang(32, 10307)
    ra_cen = np.degrees(phi)
    dec_cen = np.degrees(0.5 * np.pi - theta)
    span_dec = 1.67
    span_ra = span_dec / np.cos(np.deg2rad(dec_cen))
    ra_span = np.linspace(ra_cen-span_ra, ra_cen+span_ra, 67)
    dec_span = np.linspace(dec_cen-span_dec, dec_cen+span_dec, 67)
    x1, y1 = np.meshgrid(ra_span, dec_span)
    
    vertices = hp.boundaries(32, 10307, step=1)
    x, y, z = vertices
    theta = np.arccos(z)   
    phi = np.arctan2(y, x)    
    ra = np.degrees(phi)       
    ra = np.mod(ra, 360)   
    dec = 90 - np.degrees(theta) 
    ra_new, dec_new = offset_polygon_ra_dec(ra, dec, 900*0.2/3600)
    # theta = np.pi / 2.0 - np.radians(y1)
    # phi = np.radians(x1)
    # in_pix = hp.ang2pix(32, theta, phi)
    # yy, xx = np.where(in_pix == 10307)
    # xy_points = np.column_stack((ra_span[xx], dec_span[yy]))
    mask = is_point_in_polygon(ra_new, dec_new, x1, y1)
    xy_points = np.column_stack((x1[mask], y1[mask]))
    rand_perm = np.random.permutation(len(xy_points))
    rand_perm = rand_perm[:min(len(xy_points),count)]
    return xy_points[rand_perm]
