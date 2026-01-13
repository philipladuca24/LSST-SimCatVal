import sys
from main import SimCatVal
from utils import sample_position
import pickle as pkl

ind = int(sys.argv[1])

with open('/hildafs/home/pladuca/main/lsst-sim-package/LSST-SimCatVal/LSST-SimCatVal/ECDFS_1000_psf_e12.pkl', 'rb') as f:
    rsp_sample = pkl.load(f)

position = sample_position(ind, 67)
ra = position[ind-1][0]
dec = position[ind-1][1]
sample = rsp_sample[ind-1].copy()
sample.pop('ra')
sample.pop('dec')
##temp
# simp = {'r':sample['r']}
_ = SimCatVal('/hildafs/home/pladuca/main/skyCatalog.yaml',ra,dec,900,50,sample,31.4,True)
