import galsim
import numpy as np
from skycatalogs import skyCatalogs
from skycatalogs.utils import PolygonalRegion
from .utils import get_wcs, get_bandpasses, get_noise, get_psf, get_sat_vals
from .utils import MJD, WORLD_ORIGIN,convert_flux, get_flux

class SkyCat:
    def __init__(self,skycat_path,img_size,buffer,wcs):
        self.img_size = img_size
        self.buffer = buffer
        self.wcs = wcs
        self.sky_cat = skyCatalogs.open_catalog(skycat_path)

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

        self.galaxies = self.sky_cat.get_object_type_by_region(self.region, object_type='diffsky_galaxy', mjd=MJD)
        self.stars = self.sky_cat.get_object_type_by_region(self.region, object_type='star', mjd=MJD)
        self.trivial_sed = galsim.SED(
            galsim.LookupTable([100, 2600], [1, 1], interpolant="linear"),
            wave_type="nm",
            flux_type="fphotons",)
        self.ngal = len(self.galaxies)
        self.nstar = len(self.stars)
        self.bands = get_bandpasses()

    def get_n(self,ob_type):
        if ob_type == 'galaxy':
            return self.ngal
        else:
            return self.nstar
    
    def get_obj(self,obj_type,index,band,coadd_zp): #need to figure out what to use for lsst bandpasses
        if obj_type == 'star':
            skycat_obj = self.stars[index]
            flux = get_flux(skycat_obj, band)# This directly gets the flux from flux.parquet for both star and gal
            flux = convert_flux(flux, self.bands[band],coadd_zp)
            faint=True
            flux_cap = 3e6
            if flux > flux_cap:
                flux = flux_cap
        else: 
            skycat_obj = self.galaxies[index]
            flux = get_flux(skycat_obj, band)#(skycat_obj.get_LSST_flux(band, mjd=MJD))# * exptime * COLLECTING_AREA) #only needed if not setting ZP in bandpass
            flux = convert_flux(flux, self.bands[band],coadd_zp)
            faint = False # false
        if np.isnan(flux):
            return None
        
        if flux < 40:
            faint = True #True
        if not faint:
            seds = skycat_obj.get_observer_sed_components(mjd=MJD)

        gsobjs = skycat_obj.get_gsobject_components()
        gs_obj_list = []
        for component in gsobjs:
            if faint:
                gsobjs[component] = gsobjs[component].evaluateAtWavelength(self.bands[band])
                gs_obj_list.append(gsobjs[component] * self.trivial_sed)
            else:
                if component in seds:
                    gs_obj_list.append(gsobjs[component] * seds[component])

        if not gs_obj_list:
            return None

        if len(gs_obj_list) == 1:
            gs_object = gs_obj_list[0]
        else:
            gs_object = galsim.Add(gs_obj_list)

        gs_object = gs_object.withFlux(flux, self.bands[band])
        gs_object.flux = flux

        if (skycat_obj.object_type == "diffsky_galaxy"):
            gs_object.object_type = "galaxy"
        if skycat_obj.object_type  == "star":
            gs_object.object_type = "star"

        coord = galsim.CelestialCoord(
            ra=skycat_obj.ra * galsim.degrees,
            dec=skycat_obj.dec * galsim.degrees,
        )
        u, v = WORLD_ORIGIN.project(coord)
        dx = u.deg * 3600
        dy = v.deg * 3600

        return gs_object, dx, dy