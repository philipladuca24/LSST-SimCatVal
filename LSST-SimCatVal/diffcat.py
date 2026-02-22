import galsim
import numpy as np
from utils import get_bandpasses
from utils import MJD, convert_flux, get_flux
import pickle
from astropy.coordinates import Angle, SkyCoord
import astropy.units as u
import dask.dataframe as dd
from astropy.table import Table

class DiffCat:
    def __init__(self, diffsky_path, pointing, img_size, buffer, wcs,):
        self.wcs = wcs
        self.pointing = pointing

        self._sersic_disk = 1
        self._sersic_bulge = 4

        radius = (np.sqrt(2)*(img_size/2 + 2*buffer) * 0.2) * u.arcsec
        radius_deg = radius.to(u.deg)
        ra = pointing.ra.deg*u.degree
        dec = pointing.dec.deg*u.degree
        ra_min = ra - radius_deg / np.cos(np.deg2rad(dec))
        ra_max = ra + radius_deg / np.cos(np.deg2rad(dec))
        dec_min = dec - radius_deg
        dec_max = dec + radius_deg

        dataset = dd.read_parquet(f'{diffsky_path}/diffcat.parquet')

        filt = dataset[((dataset["ra"] > ra_min.value) &
                  (dataset["ra"] < ra_max.value) &
                  (dataset["dec"] > dec_min.value) &
                  (dataset["dec"] < dec_max.value))]

        table = filt.compute()

        self.galaxies = Table.from_pandas(table)
        self.ngal = len(self.galaxies)


    def get_knot_size(self, z):
        q = -0.5
        if z >= 0.6:
            # Above z=0.6, fractional contribution to post-convolved size
            # is <20% for smallest Roman PSF size, so can treat as point source
            # This also ensures sqrt in formula below has a
            # non-negative argument
            return None
        dA = (3e9/q**2)*(z*q+(q-1)*(np.sqrt(2*q*z+1)-1))/(1+z)**2*(1.4-0.53*z)
        return 206264.8*250/dA/2.355
    
    def get_diff_object(self,ind, row, band, coadd_zp):

        rng = galsim.BaseDeviate(int(ind))
        obj_dict = {}

        for component in ['disk','bulge','knots']:
            my_cmp = component 

            if my_cmp == 'knots':
                hlr = row[f'r50_disk_as']
                # q = np.minimum(row[f'beta_disk_as'],row[f'alpha_disk_as']) / np.maximum(row[f'beta_disk_as'],row[f'alpha_disk_as'])
                # q = row[f'alpha_disk_as'] / row[f'beta_disk_as'] ## this is wrong (not for new diffsky!!!!) but the catalog has them backwards
                q = row[f'beta_disk_as'] / row[f'alpha_disk_as']
                e1 = (1 - q) / (1 + q) * np.cos(2*row[f'psi_disk'])
                e2 = (1 - q) / (1 + q) * np.sin(2*row[f'psi_disk'])
                shear = galsim.Shear(g1=e1, g2=e2)
                ud = galsim.UniformDeviate(int(ind))
                sm = row[f'logsm_obs'] #.value
                m = (50-3)/(12-6)  # (knot_n range)/(logsm range)
                n_knot_max = m*(sm-6)+3
                n_knot = int(ud()*n_knot_max)  # random n up to n_knot_max
                if n_knot <= 0:
                    n_knot = 1  # need at least 1 knot
                npoints = n_knot


                # knot_profile = galsim.Sersic(n=self._sersic_disk,half_light_radius=hlr/2.)
                # knot_profile = knot_profile._shear(shear)
                # obj = galsim.RandomKnots(npoints=npoints,profile=knot_profile,rng=rng)
                flux = self.get_diff_flux(row[f'lsst_{band}_{my_cmp}'], coadd_zp)
                obj = galsim.RandomKnots(npoints=npoints,half_light_radius=hlr/2.,flux=flux,rng=rng)
                obj = obj._shear(shear)
                
                z = row['redshift']
                size = self.get_knot_size(z)  # get knot size
                if size is not None:
                    obj = galsim.Convolve(obj, galsim.Gaussian(sigma=size))
                obj_dict[my_cmp] = obj
            else:
                # hlr = np.sqrt(row[f'alpha_{my_cmp}_as'] * row[f'beta_{my_cmp}_as'])
                hlr = row[f'r50_{my_cmp}_as']
                # q = np.minimum(row[f'alpha_{my_cmp}_as'],row[f'beta_{my_cmp}_as']) / np.maximum(row[f'alpha_{my_cmp}_as'],row[f'beta_{my_cmp}_as'])
                # q = row[f'alpha_{my_cmp}_as'] / row[f'beta_{my_cmp}_as']
                q = row[f'beta_{my_cmp}_as'] / row[f'alpha_{my_cmp}_as']
                e1 = (1 - q) / (1 + q) * np.cos(2*row[f'psi_{my_cmp}'])
                e2 = (1 - q) / (1 + q) * np.sin(2*row[f'psi_{my_cmp}'])
                shear = galsim.Shear(g1=e1, g2=e2)
                n = self._sersic_disk if my_cmp == 'disk' else self._sersic_bulge
                flux = self.get_diff_flux(row[f'lsst_{band}_{my_cmp}'], coadd_zp)
                obj = galsim.Sersic(n=n, half_light_radius=hlr,flux=flux)
                obj_dict[my_cmp] = obj._shear(shear)

        return obj_dict

    def get_diff_flux(self,mag,coadd_zp):
        return 10**(0.4*(coadd_zp - mag)) #.value))

    def get_n(self):
        return self.ngal
    
    def get_obj(self,index,band,coadd_zp):
        
        diff_info = self.galaxies[index]
        diff_obj = self.get_diff_object(index, diff_info, band, coadd_zp)
        gs_obj_list = []
        for component in diff_obj:
            gs_obj_list.append(diff_obj[component])

        if not gs_obj_list:
            return None

        if len(gs_obj_list) == 1:
            gs_object = gs_obj_list[0]
        else:
            gs_object = galsim.Add(gs_obj_list)

        gs_object.object_type = "diff_galaxy"

        coord = galsim.CelestialCoord(
            ra=diff_info['ra'] * galsim.degrees,
            dec=diff_info['dec'] * galsim.degrees
        )
        u, v = self.pointing.project(coord)
        dx = u.deg * 3600
        dy = v.deg * 3600
        return gs_object, dx, dy, [index, diff_info['ra'], diff_info['dec'], self.get_diff_flux(diff_info[f'lsst_{band}'], coadd_zp), diff_info['redshift'], 'diff_galaxy']
        #########################  ^ swap for diff_info['idn']
