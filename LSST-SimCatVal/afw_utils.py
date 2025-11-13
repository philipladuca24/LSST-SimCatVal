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
from  lsst.afw.table import SourceTable
import galsim
import coord
import numpy as np

from utils import get_sat_vals

COLUMNS = [
    'id', 
    'coord_ra', 
    'coord_dec', 
    'deblend_nChild',
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
]

def create_afw(img,wcs,band,fwhm,sigma,coadd_zp):

    psf = psffac(defaultFwhm=fwhm/0.2, addWing=False)
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
    band_sats = get_sat_vals(coadd_zp)
    ny, nx = exp.image.array.shape
    for row in range(ny):
        for col in range(nx):
            if (exp.mask.array[row, col] & afw_image.Mask.getPlaneBitMask('SAT')) != 0 or exp.image.array[row, col] > band_sats[band]:
                exp.image.array[row, col] = band_sats[band]
                exp.mask.array[row, col] |= afw_image.Mask.getPlaneBitMask('SAT')
                # can I change this to np.where? this seems so slow
    return exp

def run_lsst_pipe(exp, deblend=True):

    configDetection = SourceDetectionTask.ConfigClass() # all from Tae code
    configDetection.thresholdValue = 10
    configDetection.thresholdType = "stdev" 
    configDetection.doTempLocalBackground = False
    configDetection.nSigmaToGrow = 2.0
    configDetection.nSigmaForKernel = 10
    configDetection.minPixels = 4
    configDetection.reEstimateBackground = False

    if deblend == True:
        configDeblend = SourceDeblendTask.ConfigClass()
    configMeasurement = SingleFrameMeasurementTask.ConfigClass()
    configMeasurement.plugins.names |= [
        "modelfit_DoubleShapeletPsfApprox",
        "modelfit_CModel",
        #"ext_gaap_GaapFlux",
    ]
    configMeasurement.slots.modelFlux = "modelfit_CModel"

    schema = SourceTable.makeMinimalSchema()
    raerr = schema.addField("coord_raErr", type="F")
    decerr = schema.addField("coord_decErr", type="F")

    detect = SourceDetectionTask(schema=schema, config=configDetection)
    detect.canMultiprocess = True
    if deblend == True:
        deblend = SourceDeblendTask(schema=schema, config=configDeblend)
    #background = SubtractBackgroundTask() ?
    measure = SingleFrameMeasurementTask(
        schema=schema, config=configMeasurement
    )
    measure.canMultiprocess = True

    table = SourceTable.make(schema)
    detect_result = detect.run(table=table, exposure=exp)
    detected_catalog = detect_result.sources
    if deblend == True:
        deblend.run(exp, detected_catalog)
    measure.run(measCat=detected_catalog, exposure=exp)
    detected_catalog = detected_catalog.copy(True)

    return detected_catalog.asAstropy()[COLUMNS]
