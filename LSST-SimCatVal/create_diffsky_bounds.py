import sys
import opencosmo as oc
import numpy as np
from glob import glob
import os
import alphashape
from shapely import contains_xy
import astropy.units as u
import pickle
from astropy.table import Table, vstack
from joblib import Parallel, delayed

diffsky_path = sys.argv[1]
diff_num = sys.argv[2]
save_path = sys.argv[3]
image_size = int(sys.argv[4])

def get_poly(dpath, dnum, imsize):
    dataset = oc.open(f'{dpath}/lc_cores-{dnum}.diffsky_gals.hdf5', synth_cores=True)
    cosmology = dataset.cosmology
    dataset = dataset.select(['ra', 'dec'])
    dataset = dataset.collect().get_data()

    points = np.column_stack((dataset['ra'].data, dataset['dec'].data)) 
    alpha_shape = alphashape.alphashape(points, 5)
    inner_poly = alpha_shape.buffer(-imsize*0.2/3600 * (1/np.cos(np.mean(dataset['dec'].data))))
    return inner_poly, cosmology

def get_points(poly, imsize):
    minra, mindec, maxra, maxdec = poly.bounds
    avgdec = (maxdec+mindec)/2
    point_count = 2*(maxdec-mindec)*3600 / (imsize *0.2)
    dec = np.linspace(mindec, maxdec, int(point_count))
    ra = np.linspace(minra, maxra, int(point_count*np.cos(avgdec)))
    x1, y1 = np.meshgrid(ra, dec)
    points = np.column_stack([x1.ravel(), y1.ravel()])
    mask = contains_xy(poly, points[:,0], points[:,1])
    xy_points = points[mask]
    return xy_points

def get_full(dpath, poly):
    bound_ds = oc.open(dpath, synth_cores=True)
    bound_ds = bound_ds.select(['lsst_u', 'lsst_u_bulge','lsst_u_knots','lsst_u_disk',
                            'lsst_g', 'lsst_g_bulge','lsst_g_knots','lsst_g_disk',
                            'lsst_r', 'lsst_r_bulge','lsst_r_knots','lsst_r_disk',
                            'lsst_i', 'lsst_i_bulge','lsst_i_knots','lsst_i_disk',
                            'lsst_z', 'lsst_z_bulge','lsst_z_knots','lsst_z_disk',
                            'lsst_y', 'lsst_y_bulge','lsst_y_knots','lsst_y_disk',
                            'ra','dec','redshift','logsm_obs', 'r50_disk', 'r50_bulge',
                            'beta_bulge','alpha_bulge','ellipticity_bulge','psi_bulge',
                            'ellipticity_disk','beta_disk','alpha_disk','psi_disk'])
    bound_ds = bound_ds.collect().get_data()
    mask = contains_xy(poly, bound_ds['ra'].data, bound_ds['dec'].data)
    return bound_ds[mask].copy()

inner_poly, cosmology = get_poly(diffsky_path,diff_num,image_size)
print("Concave Hull Complete",flush=True)

xy_points = get_points(inner_poly, image_size)
print("Points Generated",flush=True)

diffcat_paths = glob(f'{diffsky_path}/lc_cores*.hdf5')
full_ds = Parallel(n_jobs=6)(delayed(get_full)(p,inner_poly) for p in diffcat_paths)
full_ds = vstack([t for t in full_ds if len(t) > 0])

print('stack created', flush=True)
def kpc_to_arcsec(kpc, dA):  # in Mpc
    theta = (kpc.to(u.Mpc) / dA)* u.rad
    return theta.to(u.arcsec).value

dA = cosmology.angular_diameter_distance(full_ds['redshift'])
for i in ['beta_disk','alpha_disk','beta_bulge','alpha_bulge','r50_bulge','r50_disk']:
    full_ds[f'{i}_as'] = kpc_to_arcsec(full_ds[i],dA)

full_ds['idn'] = np.arange(len(full_ds)) ##new

os.makedirs(save_path, exist_ok=True)
full_ds.write(f'{save_path}/diffcat.parquet', overwrite=True)

with open(f'{save_path}/points.pickle', 'wb') as f:
    pickle.dump(xy_points, f)

print("Complete",flush=True)