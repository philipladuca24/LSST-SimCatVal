import lsst
import lsst.afw.image as afw_image
import galsim
import coord
import lsst.geom as geom
from lsst.geom import Point2I, Extent2I, Point2D, Box2I
from lsst.afw.geom import makeSkyWcs

WORLD_ORIGIN = galsim.CelestialCoord(
    ra=10.161290322580646 * galsim.degrees,
    dec=-43.40685848593699 * galsim.degrees,
)
galsim_wcs = galsim.TanWCS(
    affine=galsim.AffineTransform(
        0.2, 0, 0, 0.2,
        origin=galsim.PositionD(100 / 2, 100 / 2),
    ),
    world_origin=WORLD_ORIGIN,
    units=galsim.arcsec,
)
crpix = galsim_wcs.crpix
# DM uses 0 offset, galsim uses FITS 1 offset
stack_crpix = Point2D(crpix[0]-1, crpix[1]-1)
cd_matrix = galsim_wcs.cd
crval = geom.SpherePoint(
    galsim_wcs.center.ra/coord.radians,
    galsim_wcs.center.dec/coord.radians,
    geom.radians,)
stack_wcs = makeSkyWcs(
    crpix=stack_crpix,
    crval=crval,
    cdMatrix=cd_matrix,)

assert stack_wcs != None
print("Passed!")