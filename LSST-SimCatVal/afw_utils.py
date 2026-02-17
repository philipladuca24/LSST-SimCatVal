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
from lsst.afw.table import SourceTable, Schema
import galsim
import coord
import numpy as np
from lsst.afw.image import MultibandExposure
from lsst.meas.extensions.scarlet import ScarletDeblendTask, updateCatalogFootprints
from lsst.afw.table import SourceCatalog
from joblib import Parallel, delayed
from utils import get_sat_vals

COLUMNS = [
    'deblend_nChild',
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
    'ext_gaap_GaapFlux_flag',
    'base_CircularApertureFlux_4_5_instFlux',
    'base_CircularApertureFlux_4_5_instFluxErr',
    'base_CircularApertureFlux_9_0_instFlux',
    'base_CircularApertureFlux_9_0_instFluxErr',
    'base_CircularApertureFlux_25_0_instFlux',
    'base_CircularApertureFlux_25_0_instFluxErr',
    'base_ClassificationSizeExtendedness_flag',
]

PHOT_COLUMNS = [
    'base_PsfFlux_instFlux',
    'base_PsfFlux_instFluxErr',
    'base_PsfFlux_flag',
    'modelfit_CModel_instFlux', 
    'modelfit_CModel_instFluxErr',
    'modelfit_CModel_flag',
    'ext_gaap_GaapFlux_1_15x_Optimal_instFlux',
    'ext_gaap_GaapFlux_1_15x_Optimal_instFluxErr',
    'ext_gaap_GaapFlux_flag',
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

def measure_single_band(band, coadd_band_data, deblend_catalog_data, measureConfig, schema):
    detected_catalog = deblend_catalog_data.copy()
    measureTask = SingleFrameMeasurementTask(config=measureConfig, schema=schema)
    print(f"Measuring band {band}", flush=True)
    measureTask.run(measCat=detected_catalog, exposure=coadd_band_data)
    detected_catalog = detected_catalog.copy(True)
    print(f"Finished band {band}", flush=True)
    return band, detected_catalog.asAstropy()

def run_lsst_pipe_single(exp, forced=0, n_jobs=1):

    configDetection = SourceDetectionTask.ConfigClass()
    
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

    if forced == 0:
        deblender = SourceDeblendTask(schema=schema, config=configDeblend)
        measure = SingleFrameMeasurementTask(schema=schema, config=configMeasurement)

        table = SourceTable.make(schema)
        detect_result = detect.run(table=table, exposure=exp)
        detected_catalog = detect_result.sources

        deblender.run(exp, detected_catalog)
        measure.run(measCat=detected_catalog, exposure=exp)
        detected_catalog = detected_catalog.copy(True)

        return detected_catalog.asAstropy()
    else:
        configDeblend = SourceDeblendTask.ConfigClass()
        deblendTask = SourceDeblendTask(schema=schema, config=configDeblend)
    
        pre_measurement_schema = deblendTask.schema
        base_schema = Schema()
        for item in pre_measurement_schema:
            field = item.field
            base_schema.addField(field)

        measureTask = SingleFrameMeasurementTask(config=configMeasurement, schema=schema)
        table = SourceCatalog.Table.make(schema)
        print('Starting Detections', flush=True)

        detectionResult = detect.run(table, exp["i"])
        catalog = detectionResult.sources
        print('Starting Basic Deblending', flush=True)
        deblendTask.run(exp['i'], catalog)
        print(len(catalog))
        print(f'Starting Measurements (parallel with {n_jobs} jobs)', flush=True)
        job_args = []
        for band in exp.keys():
            job_args.append((
                band, 
                exp[band],
                catalog,
                configMeasurement,
                base_schema))

        results = Parallel(n_jobs=n_jobs, backend='loky')(
            delayed(measure_single_band)(*args) for args in job_args
        )
 
        outCatalog = {}
        for band, catalog in results:
            outCatalog[band] = catalog
        
        print('Measurements complete', flush=True)
        return outCatalog

def measure_single_band_s(band, coadd_band_data, deblend_catalog_data, scarlet_model_data, measureConfig, schema):
    measureTask = SingleFrameMeasurementTask(config=measureConfig, schema=schema)
    print(f"Measuring band {band}", flush=True)
    updateCatalogFootprints(
        modelData=scarlet_model_data,
        catalog=deblend_catalog_data,
        band=band,
        imageForRedistribution=coadd_band_data
    )
    print(f"Starting band {band}", flush=True)
    SIZE_LIMIT = 15000

    filteredCatalog = SourceCatalog(deblend_catalog_data.table.clone())
    for rec in deblend_catalog_data:
        footprint = rec.getFootprint()
        if (footprint is not None and footprint.getArea() <= SIZE_LIMIT) or rec.get('deblend_nChild') > 0:
            filteredCatalog.append(rec)
    
    print(f"Measuring {len(filteredCatalog)}/{len(deblend_catalog_data)} sources "
          f"(removed {len(deblend_catalog_data)-len(filteredCatalog)} large footprints)")

    measureTask.run(filteredCatalog, coadd_band_data)

    _catalog = SourceCatalog(filteredCatalog.table.clone())
    _catalog.extend(filteredCatalog, deep=True)
    print(f"Finished band {band}", flush=True)

    # measureTask.run(deblend_catalog_data, coadd_band_data)
    # _catalog = SourceCatalog(deblend_catalog_data.table.clone())
    # _catalog.extend(deblend_catalog_data, deep=True)
    # print(f"Finished band {band}", flush=True)
    return band, _catalog.asAstropy()

def run_lsst_pipe_multi(bands,exp, n_jobs):
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

    pre_measurement_schema = deblendTask.schema
    base_schema = Schema()
    for item in pre_measurement_schema:
        field = item.field
        base_schema.addField(field)
    
    measureConfig = SingleFrameMeasurementTask.ConfigClass()
    measureConfig.plugins.names |= [
        "modelfit_DoubleShapeletPsfApprox",
        "modelfit_CModel",
        "ext_gaap_GaapFlux",
    ]
    measureConfig.slots.modelFlux = "modelfit_CModel"
    
    measureTask = SingleFrameMeasurementTask(config=measureConfig, schema=schema)

    table = SourceCatalog.Table.make(schema)
    print('Starting Detections', flush=True)

    detectionResult = detectionTask.run(table, coadds["i"])
    catalog = detectionResult.sources

    print('Starting Scarlet Deblend', flush=True)
    deblendedCatalog, scarletModelData  = deblendTask.deblend(coadds, catalog)
    outCatalog = {}
    # print('Starting Measurements', flush=True)
    # for band in bands:
    #     # Update footprints for this band
    #     updateCatalogFootprints(
    #         modelData=scarletModelData,
    #         catalog=deblendedCatalog,
    #         band=band
    #     )
    #     print(f"Measuring band {band}",flush=True)
    #     measureTask.run(deblendedCatalog, coadds[band])

    #     _catalog = SourceCatalog(deblendedCatalog.table.clone())
    #     _catalog.extend(deblendedCatalog, deep=True)
    #     outCatalog[band] = _catalog.asAstropy()

    print(f'Deblended {len(deblendedCatalog)} sources', flush=True)
    print(f'Starting Measurements (parallel with {n_jobs} jobs)', flush=True)
    job_args = []
    for band in bands:
        job_args.append((
            band, 
            coadds[band],
            deblendedCatalog,
            scarletModelData,
            measureConfig,
            base_schema))
    
    results = Parallel(n_jobs=n_jobs, backend='loky')(delayed(measure_single_band_s)(*args) for args in job_args)
    outCatalog = {}
    for band, catalog in results:
        outCatalog[band] = catalog
    print('Measurements complete', flush=True)
    return outCatalog