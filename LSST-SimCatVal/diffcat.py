import galsim
import numpy as np
from skycatalogs import skyCatalogs
from skycatalogs.utils import PolygonalRegion
from utils import get_bandpasses
from utils import MJD, convert_flux, get_flux
import pickle

class DiffCat:
    def __init__(self,diffsky_path, pointing, img_size, buffer, wcs,):
        self.img_size = img_size
        self.buffer = buffer
        self.wcs = wcs
        self.pointing = pointing

        self._sersic_disk = 1
        self._sersic_bulge = 4

        ## open the precomputed file
        with open(diffsky_path, 'rb') as f:
            diffcat = pickle.load(f)

        corners = (
            (-buffer, -buffer),
            (img_size + buffer, -buffer),
            (img_size + buffer, img_size + buffer),
            (-buffer, img_size + buffer),
        )
        vertices = []
        for x, y in corners:
            sky_coord = self.wcs.toWorld(galsim.PositionD(x, y))
            vertices.append((sky_coord.ra / galsim.degrees, sky_coord.dec / galsim.degrees))
        self.region = PolygonalRegion(vertices)

        #use astropy regions? sample with diff wcs
        ##use astropy to self select an area

        self.galaxies = diffcat # with ob selection
        # self.stars = self.sky_cat.get_object_type_by_region(self.region, object_type='star', mjd=MJD)
        # self.trivial_sed = galsim.SED(
        #     galsim.LookupTable([100, 2600], [1, 1], interpolant="linear"),
        #     wave_type="nm",
        #     flux_type="fphotons",)
        self.ngal = len(self.galaxies)
        # self.nstar = len(self.stars)
        # self.bands = get_bandpasses()

    def get_knot_size(self, z):
        q = -0.5
        if z >= 0.6:
            # Above z=0.6, fractional contribution to post-convolved size
            # is <20% for smallest Roman PSF size, so can treat as point source
            # This also ensures sqrt in formula below has a
            # non-negative argument
            return None
        # Angular diameter scaling approximation in pc
        dA = (3e9/q**2)*(z*q+(q-1)*(np.sqrt(2*q*z+1)-1))/(1+z)**2*(1.4-0.53*z)
        # Using typical knot size 250pc, convert to sigma in arcmin
        return 206264.8*250/dA/2.355
    
    def get_diff_objects(self,row, band, coadd_zp):

        rng = galsim.BaseDeviate(int(row['index']))
        obj_dict = {}

        for component in ['disk','bulge','knots']:
            # knots use the same major/minor axes as the disk component.
            my_cmp = component 
            hlr = np.sqrt(row[f'alpha_{my_cmp}'] * row[f'beta_{my_cmp}'])
            q = row[f'beta_{my_cmp}'] / row[f'alpha_{my_cmp}']
            e1 = (1 - q) / (1 + q) * np.cos(2*row[f'psi_{my_cmp}'])
            e2 = (1 - q) / (1 + q) * np.sin(2*row[f'psi_{my_cmp}'])
            shear = galsim.Shear(g1=e1, g2=e2)

            if component == 'knots':
                ud = galsim.UniformDeviate(int(row['index']))
                sm = row[f'logsm_obs']
                m = (50-3)/(12-6)  # (knot_n range)/(logsm range)
                n_knot_max = m*(sm-6)+3
                n_knot = int(ud()*n_knot_max)  # random n up to n_knot_max
                if n_knot == 0:
                    n_knot += 1  # need at least 1 knot

                npoints = n_knot
                assert npoints > 0
                knot_profile = galsim.Sersic(n=self._sersic_disk,half_light_radius=hlr/2.)
                knot_profile = knot_profile._shear(shear)

                flux = self.get_diff_flux(row[f'lsst_{band}_{my_cmp}'], coadd_zp)

                obj = galsim.RandomKnots(npoints=npoints,profile=knot_profile,flux=flux,rng=rng)
                z = row['redshift']
                size = self.get_knot_size(z)  # get knot size
                if size is not None:
                    obj = galsim.Convolve(obj, galsim.Gaussian(sigma=size))
                obj_dict[component] = obj
            else:
                n = self._sersic_disk if component == 'disk' else self._sersic_bulge
                flux = self.get_diff_flux(row[f'lsst_{band}_{my_cmp}'], coadd_zp)
                obj = galsim.Sersic(n=n, half_light_radius=hlr,flux=flux)
                obj_dict[component] = obj._shear(shear)

        return obj_dict

    def get_diff_flux(self,mag,coadd_zp):
        return 10**(0.4*(coadd_zp - mag))

    def get_n(self):
        return self.ngal
    
    def get_obj(self,index,band,coadd_zp): #need to figure out what to use for lsst bandpasses
        
        diff_info = self.galaxies[index]
        diff_obj = self.get_diff_object(diff_info, band, coadd_zp)
        gs_obj_list = []
        for component in diff_obj:
            gs_obj_list.append(diff_obj[component])

        if not gs_obj_list:
            return None

        if len(gs_obj_list) == 1:
            gs_object = gs_obj_list[0]
        else:
            gs_object = galsim.Add(gs_obj_list)

        gs_object.object_type = "galaxy"

        coord = galsim.CelestialCoord(
            ra=diff_info.ra * galsim.degrees,
            dec=diff_info.dec * galsim.degrees,
        )
        u, v = self.pointing.project(coord)
        dx = u.deg * 3600
        dy = v.deg * 3600

        return gs_object, dx, dy, [diff_info.ra, diff_info.dec, self.get_diff_flux(diff_info[f'lsst_{band}'], coadd_zp), 'diff_galaxy'] #optional items to make a truth catalog

