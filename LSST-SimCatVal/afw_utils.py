import lsst.afw.image as afw_image
from lsst.pipe.tasks.characterizeImage import CharacterizeImageTask
from lsst.meas.algorithms.detection import SourceDetectionTask
from lsst.meas.algorithms.subtractBackground import SubtractBackgroundTask
from lsst.meas.deblender import SourceDeblendTask
from lsst.meas.base import SingleFrameMeasurementTask
from lsst.meas.algorithms import GaussianPsfFactory as psffac
import lsst.meas.extensions.gaap
import lsst.meas.modelfit
import lsst.geom as geom
from lsst.geom import Point2D
from lsst.afw.geom import makeSkyWcs
from lsst.afw.table import SourceTable
import galsim
import coord
import numpy as np

## multi pipe imports
# from lsst.pipe.tasks.multiBand import DetectCoaddSourcesTask, MeasureMergedCoaddSourcesTask, DeblendCoaddSourcesMultiTask
# from lsst.pipe.tasks.mergeDetections import MergeDetectionsTask
# from lsst.pipe.tasks.mergeMeasurements import MergeMeasurementsTask
# from lsst.pipe.tasks.postprocess import WriteObjectTableTask
# from lsst.drp.tasks.forcedPhotCoadd import ForcedPhotCoaddTask
# from lsst.pipe.tasks.coaddBase import SkyInfo
# from lsst.skymap import TractInfo, PatchInfo, BaseSkyMap
# from lsst.meas.algorithms import DeconvolutionTask
# from lsst.meas.base import IdFactory

from utils import get_sat_vals

COLUMNS = [
    'id', 
    'coord_ra', 
    'coord_dec', 
    'base_SdssCentroid_x',
    'base_SdssCentroid_y',
    'base_SdssCentroid_flag',
    'base_SdssShape_xx', 
    'base_SdssShape_yy', 
    'base_SdssShape_xy',
    'base_SdssShape_psf_xx', 
    'base_SdssShape_psf_yy',
    'base_SdssShape_psf_xy',
    'base_SdssShape_flag',
    'modelfit_DoubleShapeletPsfApprox_flag',
    'base_PsfFlux_instFlux',
    'base_PsfFlux_instFluxErr',
    'base_PsfFlux_flag',
    'modelfit_CModel_instFlux', 
    'modelfit_CModel_instFluxErr',
    'modelfit_CModel_flag',
    'ext_gaap_GaapFlux_1_15x_Optimal_instFlux',
    'ext_gaap_GaapFlux_1_15x_Optimal_instFluxErr',
    'base_CircularApertureFlux_4_5_instFlux',
    'base_CircularApertureFlux_4_5_instFluxErr',
    'base_CircularApertureFlux_9_0_instFlux',
    'base_CircularApertureFlux_9_0_instFluxErr',
    'base_CircularApertureFlux_25_0_instFlux',
    'base_CircularApertureFlux_25_0_instFluxErr',
]

def create_afw(img,wcs,band,fwhm,sigma,coadd_zp):

    psf = psffac(defaultFwhm=fwhm, addWing=False)
    psf = psf.apply()

    crpix = wcs.crpix
    # DM uses 0 offset, galsim uses FITS 1 offset
    stack_crpix = Point2D(crpix[0]-1, crpix[1]-1)
    cd_matrix = wcs.cd
    crval = geom.SpherePoint(
        wcs.center.ra/coord.radians,
        wcs.center.dec/coord.radians,
        geom.radians,)
    stack_wcs = makeSkyWcs(
        crpix=stack_crpix,
        crval=crval,
        cdMatrix=cd_matrix,)

    variance = img.copy()
    variance.array[:, :] = sigma**2

    masked_image = afw_image.MaskedImage(len(img.array[0]), len(img.array[0]), dtype=np.float32)
    masked_image.image.array[:, :] = img.array
    masked_image.variance.array[:, :] = variance.array

    shape = img.array.shape
    bmask = np.zeros(shape, dtype=np.int64)
    masked_image.mask.array[:, :] = bmask

    exp = afw_image.Exposure(masked_image, dtype=np.float32)
    zero_flux = 10.0 ** (0.4 * coadd_zp)
    photoCalib = afw_image.makePhotoCalibFromCalibZeroPoint(zero_flux)
    exp.setPhotoCalib(photoCalib)

    filter_label = afw_image.FilterLabel(band=band, physical=band)
    exp.setFilter(filter_label)

    exp.setPsf(psf)

    exp.setWcs(stack_wcs)

    # detector = DetectorWrapper().detector
    # exp.setDetector(detector)
    # band_sats = get_sat_vals(coadd_zp)
    # ny, nx = exp.image.array.shape
    # for row in range(ny):
    #     for col in range(nx):
    #         if (exp.mask.array[row, col] & afw_image.Mask.getPlaneBitMask('SAT')) != 0 or exp.image.array[row, col] > band_sats[band]:
    #             exp.image.array[row, col] = band_sats[band]
    #             exp.mask.array[row, col] |= afw_image.Mask.getPlaneBitMask('SAT')
                # can I change this to np.where? this seems so slow
    return exp

def run_lsst_pipe_single(exp, deblend=True):

    configDetection = SourceDetectionTask.ConfigClass()
    
    if deblend:
        configDeblend = SourceDeblendTask.ConfigClass()
    configMeasurement = SingleFrameMeasurementTask.ConfigClass()
    configMeasurement.plugins.names |= [
        "modelfit_DoubleShapeletPsfApprox",
        "modelfit_CModel",
        "ext_gaap_GaapFlux",
    ]
    configMeasurement.slots.modelFlux = "modelfit_CModel"

    schema = SourceTable.makeMinimalSchema()
    raerr = schema.addField("coord_raErr", type="F")
    decerr = schema.addField("coord_decErr", type="F")

    detect = SourceDetectionTask(schema=schema, config=configDetection)
    if deblend:
        deblender = SourceDeblendTask(schema=schema, config=configDeblend)
    #background = SubtractBackgroundTask() ?
    measure = SingleFrameMeasurementTask(schema=schema, config=configMeasurement)

    table = SourceTable.make(schema)
    detect_result = detect.run(table=table, exposure=exp)
    detected_catalog = detect_result.sources
    if deblend:
        deblender.run(exp, detected_catalog)
    measure.run(measCat=detected_catalog, exposure=exp)
    detected_catalog = detected_catalog.copy(True)

    # if deblend:
    #     return detected_catalog.asAstropy()[COLUMNS+['deblend_nChild']]  
    return detected_catalog.asAstropy() #[COLUMNS]

# def run_lsst_pipe_multi(exp_dict):
#     ref_exp = exp_dict['i']
#     bbox = ref_exp.getBBox()
#     wcs = ref_exp.getWcs()

#     # Minimal synthetic tract+patch
#     tractId = 0
#     patchId = (0,0)
#     tractInfo = TractInfo(id=tractId,wcs=wcs,bbox=bbox,config=BaseSkyMap.ConfigClass())
#     patchInfo = PatchInfo(id=patchId,tract=tractInfo,bbox=bbox)
#     skyInfo = SkyInfo(tractInfo=tractInfo,patchInfo=patchInfo,bbox=bbox,wcs=wcs)
    
#     configDetection = DetectCoaddSourcesTask.ConfigClass() # all from Tae code
#     configDetection.thresholdValue = 3
#     # configDetection.thresholdType = "stdev" 
#     configDetection.doTempLocalBackground = False
#     configDetection.reEstimateBackground = False
#     # configDetection.nSigmaToGrow = 2.0
#     # configDetection.nSigmaForKernel = 10
#     # configDetection.minPixels = 4

#     configMergeDetect = MergeDetectionsTask.ConfigClass()
#     configDeblend = DeblendCoaddSourcesMultiTask.ConfigClass()

#     configMeasurement = MeasureMergedCoaddSourcesTask.ConfigClass()
#     configMeasurement.plugins.names |= [
#         "modelfit_DoubleShapeletPsfApprox",
#         "modelfit_CModel",
#         "ext_gaap_GaapFlux",
#     ]
#     configMeasurement.slots.modelFlux = "modelfit_CModel"
#     configMeasurement.plugins['ext_gaap_GaapFlux'].doMeasure = True

#     configMergeMeasure = MergeMeasurementsTask.ConfigClass()
#     configForcedPhot = ForcedPhotCoaddTask.ConfigClass()

#     configWrite = WriteObjectTableTask.ConfigClass()

#     schema = SourceTable.makeMinimalSchema()
#     raerr = schema.addField("coord_raErr", type="F")
#     decerr = schema.addField("coord_decErr", type="F")

#     detect = SourceDetectionTask(schema=schema, config=configDetection)
#     detect.canMultiprocess = True
#     mergeDetect = MergeDetectionsTask(schema=schema, config=configMergeDetect)

#     deblend = DeblendCoaddSourcesMultiTask(config=configDeblend, schema=schema)

#     measure = MeasureMergedCoaddSourcesTask(schema=schema, config=configMeasurement)
#     measure.canMultiprocess = True

#     mergeMeasure = MergeMeasurementsTask(schema=schema, config=configMergeMeasure)
#     mergeMeasure.priorityList = ['i','r','g','z','y','u']
#     forcedPhot = ForcedPhotCoaddTask(schema=schema, config=configForcedPhot)
#     write = WriteObjectTableTask(schema=schema, config=configWrite)

#     idFactory = IdFactory.makeSimple()
#     dec = DeconvolutionTask()
#     ids = {'u':1, 'g':2,'r':3,'i':4,'z':5,'y':6}
#     sources = {}
#     deconvolvedCoadds = {}
#     bands = []
#     for band, exp in exp_dict.items():
#         deconvolvedCoadds[band] = dec.run(exposure=exp).exposure
#         sources[band] = detect.run(exp, idFactory, ids[band]).sources
#         bands.append(band)
#     mergedDet = mergeDetect.run(sources, skyInfo=skyInfo, idFactory=idFactory,skySeed=67)
#     deblended = deblend.run(coadds=exp_dict, bands=bands, mergedDetections=mergedDet.mergedDetections, deconvolvedCoadds=deconvolvedCoadds, idFactory=idFactory)
#     measurements = {}
#     for band, exp in exp_dict.items():
#         measurements[band] = measure.run(exposure=exp, sources=deblended.sources, skyInfo=skyInfo, exposureId=ids[band])
#     mergeMes = mergeMeasure.run(measurements)
#     forcedResults = {}
#     for band, exp in exp_dict.items():
#         forcedResults[band] = forcedPhot.run(measCat=mergeMes.sources,exposure=exp,refCat=mergeMes.sources,refWcs=exp.getWcs(),exposureId=ids[band])

#     out = write.run(forcedResults, 0, 0)

#     print(out)
#     print(out.columns)

#     return out[COLUMNS+['deblend_nChild']]  
