import sys
import opencosmo as oc
import numpy as np
from glob import glob
import astropy
import alphashape
from shapely import contains_xy
import astropy.units as u
import pickle

diffsky_path = sys.argv[1]
diff_num = sys.argv[2]
save_path = sys.argv[3]
image_size = sys.argv[4]

dataset = oc.open(f'{diffsky_path}/lc_cores-{diff_num}.diffsky_gals.hdf5', synth_cores=True)
dataset = dataset.select(['ra', 'dec'])
dataset = dataset.collect().get_data()

points = np.column_stack((dataset['ra'].data, dataset['dec'].data)) 
alpha_shape = alphashape.alphashape(points, 5)
inner_poly = alpha_shape.buffer(-image_size*0.2/3600 * (1/np.cos(np.mean(dataset['dec'].data))))
print("Concave Hull Complete")

minra, mindec, maxra, maxdec = inner_poly.bounds
avgdec = (maxdec+mindec)/2
point_count = 2*(maxdec-mindec)*3600 / (image_size *0.2)
dec = np.linspace(mindec, maxdec, int(point_count))
ra = np.linspace(minra, maxra, int(point_count*np.cos(avgdec)))
x1, y1 = np.meshgrid(ra, dec)
points = np.column_stack([x1.ravel(), y1.ravel()])
mask = contains_xy(inner_poly, points[:,0], points[:,1])
xy_points = points[mask]

print("Points Generated")

diffsky_paths = glob(f'{diffsky_path}/lc_cores*.hdf5')
ds_list = []
for diff in diffsky_paths:
    bound_ds = oc.open(diff, synth_cores=True)
    cosmology = bound_ds.cosmology
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

    mask = contains_xy(inner_poly, bound_ds['ra'].data, bound_ds['dec'].data)
    ds_list.append(bound_ds[mask])

full_ds = astropy.table.vstack(ds_list)

def kpc_to_arcsec(kpc, z):
    dA = cosmology.angular_diameter_distance(z)  # in Mpc
    theta = (kpc.to(u.Mpc) / dA)* u.rad
    return theta.to(u.arcsec).value

for i in ['beta_disk','alpha_disk','beta_bulge','alpha_bulge','r50_bulge','r50_disk']:
    full_ds[f'{i}_as'] = kpc_to_arcsec(full_ds[i],full_ds['redshift'])

with open(f'{save_path}/diffcat.pickle', 'wb') as f:
    pickle.dump(f, full_ds)

with open(f'{save_path}/points.pickle', 'wb') as f:
    pickle.dump(f, xy_points)

print("Complete")



