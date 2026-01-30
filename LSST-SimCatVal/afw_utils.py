import lsst.afw.image as afwImage
import lsst.afw.math as afwMath
from lsst.meas.algorithms import KernelPsf
from lsst.meas.algorithms.detection import SourceDetectionTask
from lsst.meas.deblender import SourceDeblendTask
from lsst.meas.base import SingleFrameMeasurementTask
import lsst.meas.extensions.gaap
import lsst.meas.modelfit
import lsst.geom as geom
from lsst.geom import Point2D
from lsst.afw.geom import makeSkyWcs
from lsst.afw.table import SourceTable
import galsim
import coord
import numpy as np
from lsst.afw.image import MultibandExposure
from lsst.meas.extensions.scarlet import ScarletDeblendTask, updateCatalogFootprints
from lsst.afw.table import SourceCatalog

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

def create_afw(img,wcs,band,psf_im,sigma,coadd_zp):

    # psf = psffac(defaultFwhm=fwhm, addWing=False)
    # psf = psf.apply()
    psf = afwImage.ImageF(np.array(psf_im.array,dtype=np.float32)) # a generated image of the kolmogorov psf
    psf = afwImage.ImageD(psf, deep=True)
    psf = afwMath.FixedKernel(psf)
    psf = KernelPsf(psf)

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

    masked_image = afwImage.MaskedImage(len(img.array[0]), len(img.array[0]), dtype=np.float32)
    masked_image.image.array[:, :] = img.array
    masked_image.variance.array[:, :] = variance.array

    shape = img.array.shape
    bmask = np.zeros(shape, dtype=np.int64)
    masked_image.mask.array[:, :] = bmask

    exp = afwImage.Exposure(masked_image, dtype=np.float32)
    zero_flux = 10.0 ** (0.4 * coadd_zp)
    photoCalib = afwImage.makePhotoCalibFromCalibZeroPoint(zero_flux)
    exp.setPhotoCalib(photoCalib)

    filter_label = afwImage.FilterLabel(band=band, physical=band)
    exp.setFilter(filter_label)

    exp.setPsf(psf)

    exp.setWcs(stack_wcs)

    # detector = DetectorWrapper().detector
    # exp.setDetector(detector)

    # band_sats = get_sat_vals(coadd_zp)
    # exp.mask.array = np.where(exp.image.array > band_sats[band], exp.mask.array | afw_image.Mask.getPlaneBitMask('SAT'), exp.mask.array)
    # exp.image.array = np.where(exp.image.array > band_sats[band], band_sats[band], exp.image.array)

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

def run_lsst_pipe_multi(bands,exp):
    coadds = MultibandExposure.fromExposures(bands, exp)
    for i in range(len(bands)):
        coadds[bands[i]].setWcs(exp[i].getWcs())
        coadds[bands[i]].setPhotoCalib(exp[i].getPhotoCalib())

    schema = SourceCatalog.Table.makeMinimalSchema()
    raerr = schema.addField("coord_raErr", type="F")
    decerr = schema.addField("coord_decErr", type="F")

    detectionTask = SourceDetectionTask(schema=schema)
    config = ScarletDeblendTask.ConfigClass()
    deblendTask = ScarletDeblendTask(schema=schema, config=config)
    
    measureConfig = SingleFrameMeasurementTask.ConfigClass()
    measureConfig.plugins.names |= [
        "modelfit_DoubleShapeletPsfApprox",
        "modelfit_CModel",
        "ext_gaap_GaapFlux",
    ]
    measureConfig.slots.modelFlux = "modelfit_CModel"
    
    measureTask = SingleFrameMeasurementTask(config=measureConfig, schema=schema)

    table = SourceCatalog.Table.make(schema)
    detectionResult = detectionTask.run(table, coadds["i"])
    catalog = detectionResult.sources
    
    deblendedCatalog, scarletModelData  = deblendTask.deblend(coadds, catalog)
    outCatalog = {}
    for band in bands:
        # Update footprints for this band
        updateCatalogFootprints(
            modelData=scarletModelData,
            catalog=deblendedCatalog,
            band=band
        )
        print(f"Measuring band {band}")
        measureTask.run(deblendedCatalog, coadds[band])

        _catalog = SourceCatalog(deblendedCatalog.table.clone())
        _catalog.extend(deblendedCatalog, deep=True)
        outCatalog[band] = _catalog.asAstropy()

    return outCatalog