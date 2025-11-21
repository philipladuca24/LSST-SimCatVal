import numpy as np
from lsst.afw.detection import Threshold, FootprintSet, InvalidPsfError
from lsst.daf.butler import Butler
from lsst.geom import Box2I, Point2I, SpherePoint, degrees
from lsst.afw.image import MaskedImageF

""" THIS CAN ONLY RUN ON RSP WITH BUTLER """
DP1_FIELDS = {
    "ECDFS": (53.13, -28.10),  # Extended Chandra Deep Field South
    "EDFS": (59.10, -48.73),  # Euclid Deep Field South
    "Rubin_SV_38_7": (37.86, 6.98),  # Low Ecliptic Latitude Field
    "Rubin_SV_95_-25": (95.00, -25.00),  # Low Galactic Latitude Field
    "47_Tuc": (6.02, -72.08),  # 47 Tuc Globular Cluster
    "Fornax_dSph": (40.00, -34.45),  # Fornax Dwarf Spheroidal Galaxy
    "Seagull": (106.23, -10.51)
}
DP1_BANDS = {
    "ECDFS": 'ugrizy',  # Extended Chandra Deep Field South
    "EDFS": 'ugrizy',  # Euclid Deep Field South
    "Rubin_SV_38_7": 'griz',  # Low Ecliptic Latitude Field
    "Rubin_SV_95_-25": 'ugrizy',  # Low Galactic Latitude Field
    "47_Tuc": 'griy',  # 47 Tuc Globular Cluster
    "Fornax_dSph": 'gri',  # Fornax Dwarf Spheroidal Galaxy
    "Seagull": 'ugrz'
}

SIGMA_TO_FWHM = 2.0*np.sqrt(2.0*np.log(2.0))

class Dp1Sampler:
    def __init__(self,im_field):
        self.butler = Butler("dp1", collections="LSSTComCam/DP1")
        assert self.butler is not None
        self.im_field = im_field
        
        expt_map = self.butler.get('deepCoadd_psf_maglim_consolidated_map_weighted_mean', band='r') # all fields have r band, should I check more / g band
        span_dec = 0.75
        span_ra = span_dec / np.cos(np.deg2rad(DP1_FIELDS[im_field][1]))
        ra = np.linspace(DP1_FIELDS[im_field][0]-span_ra, DP1_FIELDS[im_field][0]+span_ra, 200)#sampled at a resolution of 150 lsst pixels/ meshgrid pixel, 
        dec = np.linspace(DP1_FIELDS[im_field][1]-span_dec, DP1_FIELDS[im_field][1]+span_dec, 200)#this will give some overlap in 200 pixel box background estimates
        x, y = np.meshgrid(ra, dec)
        expt = expt_map.get_values_pos(x, y)
        yy, xx = np.where(expt>0)
        self.sample_points = np.column_stack((ra[xx], dec[yy]))

    def get_zp(self):
        query = f"band.name = '{'r'}' AND patch.region OVERLAPS POINT({DP1_FIELDS[self.im_field][0]}, {DP1_FIELDS[self.im_field][1]})"
        dataset_refs = self.butler.query_datasets("deep_coadd", where=query)
        assert len(dataset_refs) > 0 
        ref = dataset_refs[0]
        deep_coadd = self.butler.get(ref) 
        photcalib = deep_coadd.getPhotoCalib()
        zp = photcalib.instFluxToMagnitude(1.0)
        return zp
    
    def sample(self, num):
        i = 0
        sim_configs = []
        while i < num:
            choice = np.random.randint(len(self.sample_points)) # this is uniform random sampling without replacement
            point = self.sample_points[choice]
            try:
                out = self.measure(point, DP1_BANDS[self.im_field])
                sim_configs.append(out)
                i += 1
            except (AssertionError, InvalidPsfError) as e:
                self.sample_points = np.delete(self.sample_points, choice, axis=0) # can't think of a better way to remove invalid points
        return sim_configs

    def get_image(self, point, band, choice=None):
        query = f"band.name = '{band}' AND patch.region OVERLAPS POINT({point[0]}, {point[1]})"
        dataset_refs = self.butler.query_datasets("deep_coadd", where=query)
        assert len(dataset_refs) > 0 
        if choice == None:
            choice = np.random.randint(len(dataset_refs))
            ref = dataset_refs[choice]
            deep_coadd = self.butler.get(ref)  
            return deep_coadd, choice
        ref = dataset_refs[choice]
        deep_coadd = self.butler.get(ref)  
        return deep_coadd

    def get_measures(self, band, deep_coadd, bbox, point_image,zp):
        mi = deep_coadd.getMaskedImage()
        mi = MaskedImageF(mi, bbox, deep=True)
        variance = deep_coadd.getVariance()
        masked = deep_coadd.getMask()

        threshold_sigma = 10.0
        threshold = Threshold(threshold_sigma)
        footprints = FootprintSet(mi, threshold)
        mask_plane_name = 'DETECTED'
        footprints.setMask(masked, mask_plane_name)
        combo = (masked.array == 0)
        var = variance.array
        var_out = np.sqrt(np.median(var[combo]))

        psf_coadd = deep_coadd.getPsf()
        psf_shape = psf_coadd.computeShape(point_image)
        psf_sigma = psf_shape.getDeterminantRadius()
        psf_out = psf_sigma * SIGMA_TO_FWHM

        #psf image for interpolation
        # psf_out = psf_coadd.computeKernelImage(point_image)

        return {band: {'zp': zp,'psf': psf_out,'var': var_out}}
            
    def measure(self, point, bands):
        out = {}
        bands = bands.replace('r', '')
        deep_coadd, choice = self.get_image(point, 'r')
        
        photcalib = deep_coadd.getPhotoCalib()
        zp = photcalib.instFluxToMagnitude(1.0)
        
        my_spherePoint = SpherePoint(point[0]*degrees, point[1]*degrees)
        wcs_coadd = deep_coadd.getWcs()
        point_image = wcs_coadd.skyToPixel(my_spherePoint)

        mi = deep_coadd.getMaskedImage()
        box_size = 200 
        x0 = int(point_image.getX() - deep_coadd.getX0()) - box_size // 2
        y0 = int(point_image.getY() - deep_coadd.getY0()) - box_size // 2
        height, width = mi.getHeight(), mi.getWidth()
        x0 = max(0, min(x0, width - box_size))
        y0 = max(0, min(y0, height - box_size))
        x1 = x0 + box_size
        y1 = y0 + box_size
        bbox = Box2I(Point2I(x0 + deep_coadd.getX0(), y0 + deep_coadd.getY0()), Point2I(x1 + deep_coadd.getX0(), y1 + deep_coadd.getY0()))

        out = self.get_measures('r', deep_coadd, bbox, point_image,zp)

        for band in bands:
            deep_coadd = self.get_image(point, band, choice)
            temp = self.get_measures(band, deep_coadd, bbox, point_image, zp)
            out = out | temp

        return out
            