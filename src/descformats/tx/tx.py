"""
This file contains TXPipe-specific file types, subclassing the more
generic types in base.py
"""
import yaml
import numpy as np

from .base import HDFFile, DataFile, YamlFile
import numpy as np
from .base import FitsFile, HDFFile
from .types import metacalibration_names



def metacalibration_names(names):
    """
    Generate the metacalibrated variants of the inputs names,
    that is, variants with _1p, _1m, _2p, and _2m on the end
    of each name.
    """
    suffices = ["1p", "1m", "2p", "2m"]
    out = []
    for name in names:
        out += [name + "_" + s for s in suffices]
    return out


class MetacalCatalog(FitsFile):
    """
    A metacal output catalog
    """
    # These are columns
    metacal_columns = [
        'mcal_g', 'mcal_g_cov',  'mcal_pars',  'mcal_pars_cov',
        'mcal_T', 'mcal_T_err', 'mcal_T_r', 'mcal_s2n_r',]

    other_columns = ['mcal_flux_cov', 'mcal_weight', 'mcal_flux',
        'mcal_flux_s2n', 'mcal_mag', 'mcal_gpsf', 'mcal_logsb', 'mcal_Tpsf']

    # The parent class will check these columns exist.
    required_columns = ( metacal_columns
                        + metacalibration_names(metacal_columns)
                        + other_columns )

    # Add methods for handling here ...


class ShearCatalog(HDFFile):
    """
    A generic shear catalog
    """

    # These are columns

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._catalog_type = None

    def read_catalog_info(self):
        try:
            group = self.fileObj["shear"]
            info = dict(group.attrs)
        except Exception as err:
            raise ValueError(f"Unable to read shear catalog {str(err)}") from err
        shear_catalog_type = info.get("catalog_type")
        return shear_catalog_type

    @property
    def catalog_type(self):
        if self._catalog_type is not None:
            return self._catalog_type

        if "catalog_type" in self.fileObj["shear"].attrs:
            t = self.fileObj["shear"].attrs["catalog_type"]
        elif "mcal_g1" in self.fileObj["shear"].keys():
            t = "metacal"
        elif "1p" in self.fileObj["shear"].keys():
            t = "metadetect"
        elif "c1" in self.fileObj["shear"].keys():
            t = "lensfit"
        else:
            raise ValueError("Could not figure out catalog format")

        self._catalog_type = t
        return t

    def get_size(self):
        if self.catalog_type == "metadetect":
            return self.fileObj["shear/00/ra"].size
        return self.fileObj["shear/ra"].size

    def get_primary_catalog_names(self, true_shear=False):
        if true_shear:
            shear_cols = ["true_g1", "true_g2", "ra", "dec", "weight"]
            rename = {"true_g1": "g1", "true_g2": "g2"}
        elif self.catalog_type == "metacal":
            shear_cols = ["mcal_g1", "mcal_g2", "ra", "dec", "weight"]
            rename = {"mcal_g1": "g1", "mcal_g2": "g2"}
        elif self.catalog_type == "hsc":
            shear_cols = ["g1", "g2", "c1", "c2", "ra", "dec", "weight"]
            rename = {}
        elif self.catalog_type == "metadetect":
            shear_cols = ["00/g1", "00/g2", "00/ra", "00/dec", "00/weight"]
            rename = {c: c[3:] for c in shear_cols}
        else:
            shear_cols = ["g1", "g2", "ra", "dec", "weight"]
            rename = {}

        return shear_cols, rename


class TomographyCatalog(HDFFile):
    required_datasets = []

    def read_zbins(self, bin_type):
        """
        Read saved redshift bin edges from attributes
        """
        d = dict(self.fileObj["tomography"].attrs)
        nbin = d[f"nbin_{bin_type}"]
        zbins = [
            (d[f"{bin_type}_zmin_{i}"], d[f"{bin_type}_zmax_{i}"]) for i in range(nbin)
        ]
        return zbins

    def read_nbin(self, bin_type):
        d = dict(self.fileObj["tomography"].attrs)
        return d[f"nbin_{bin_type}"]


class RandomsCatalog(HDFFile):
    required_datasets = ["randoms/ra", "randoms/dec"]


class MapsFile(HDFFile):
    required_datasets = []

    def list_maps(self):
        import h5py  # pylint: disable=import-outside-toplevel

        maps = []

        # h5py uses this visititems method to walk through
        # a file, looking at everything underneath a path.
        # We use it here to search through everything in the
        # "maps" section of a maps file looking for any groups
        # that seem to be a map.  You have to pass a function
        # like this to visititems.
        def visit(name, obj):
            if isinstance(obj, h5py.Group):
                keys = obj.keys()
                # we save maps with these two data sets,
                # so if they are both there then this will
                # be a map
                if "pixel" in keys and "value" in keys:
                    maps.append(name)

        # Now actually run this
        self.fileObj["maps"].visititems(visit)

        # return the accumulated list
        return maps

    def read_healpix(self, map_name, return_all=False):
        import healpy  # pylint: disable=import-outside-toplevel

        group = self.fileObj[f"maps/{map_name}"]
        nside = group.attrs["nside"]
        npix = healpy.nside2npix(nside)
        m = np.repeat(healpy.UNSEEN, npix)
        pix = group["pixel"][:]
        val = group["value"][:]
        m[pix] = val
        if return_all:
            return m, pix, nside
        return m

    def read_map_info(self, map_name):
        group = self.fileObj[f"maps/{map_name}"]
        info = dict(group.attrs)
        if "pixelization" not in info:
            raise ValueError(
                f"Map '{map_name}' not found, "
                f"or not saved properly in file {self.path}"
            )
        return info

    def read_map(self, map_name):
        info = self.read_map_info(map_name)
        pixelization = info["pixelization"]
        if pixelization == "gnomonic":
            m = self.read_gnomonic(map_name)
        elif pixelization == "healpix":
            m = self.read_healpix(map_name)
        else:
            raise ValueError(f"Unknown map pixelization type {pixelization}")
        return m

    def read_mask(self):
        mask = self.read_map("mask")
        mask[mask < 0] = 0
        return mask

    def write_map(self, map_name, pixel, value, metadata):
        """
        Save an output map to an HDF5 subgroup.

        The pixel numbering and the metadata are also saved.

        Parameters
        ----------

        group: H5Group
            The h5py Group object in which to store maps
        name: str
            The name of this map, used as the name of a subgroup in the group where the data is stored.
        pixel: array
            Array of indices of observed pixels
        value: array
            Array of values of observed pixels
        metadata: mapping
            Dict or other mapping of metadata to store along with the map
        """
        if "maps" not in self.fileObj:
            self.fileObj.create_group("maps")
        if "pixelization" not in metadata:
            raise ValueError("Map metadata should include pixelization")
        if not pixel.shape == value.shape:
            raise ValueError(
                f"Map pixels and values should be same shape "
                f"but are {pixel.shape} vs {value.shape}"
            )
        subgroup = self.fileObj["maps"].create_group(map_name)
        subgroup.attrs.update(metadata)
        subgroup.create_dataset("pixel", data=pixel)
        subgroup.create_dataset("value", data=value)

    def plot_healpix(self, map_name, view="cart", **kwargs):
        import healpy  # pylint: disable=import-outside-toplevel

        m, pix, nside = self.read_healpix(map_name, return_all=True)
        lon, lat = healpy.pix2ang(nside, pix, lonlat=True)
        # npix = healpy.nside2npix(nside)
        if len(pix) == 0:
            print(f"Empty map {map_name}")
            return
        if len(pix) == len(m):
            w = np.where((m != healpy.UNSEEN) & (m != 0))
        else:
            w = None
        lon_range = [lon[w].min() - 0.1, lon[w].max() + 0.1]
        lat_range = [lat[w].min() - 0.1, lat[w].max() + 0.1]
        lat_range = np.clip(lat_range, -90, 90)
        m[m == 0] = healpy.UNSEEN
        title = kwargs.pop("title", map_name)
        if view == "cart":
            healpy.cartview(
                m, lonra=lon_range, latra=lat_range, title=title, hold=True, **kwargs
            )
        elif view == "moll":
            healpy.mollview(m, title=title, hold=True, **kwargs)
        else:
            raise ValueError(f"Unknown Healpix view mode {view}")

    def read_gnomonic(self, map_name):

        group = self.fileObj[f"maps/{map_name}"]
        info = dict(group.attrs)
        nx = info["nx"]
        ny = info["ny"]
        m = np.zeros((ny, nx))
        m[:, :] = np.nan

        pix = group["pixel"][:]
        val = group["value"][:]
        w = np.where(pix != -9999)
        pix = pix[w]
        val = val[w]
        x = pix % nx
        y = pix // nx
        m[y, x] = val
        return m

    def plot_gnomonic(self, map_name, **kwargs):
        import matplotlib.pyplot as plt  # pylint: disable=import-outside-toplevel

        info = self.read_map_info(map_name)
        ra_min, ra_max = info["ra_min"], info["ra_max"]
        if ra_min > 180 and ra_max < 180:
            ra_min -= 360
        ra_range = (ra_max, ra_min)
        dec_range = (info["dec_min"], info["dec_max"])

        # the view arg is needed for healpix but not gnomonic
        kwargs.pop("view")
        m = self.read_gnomonic(map_name)
        extent = list(ra_range) + list(dec_range)
        title = kwargs.pop("title", map_name)
        plt.imshow(m, aspect="equal", extent=extent, **kwargs)
        plt.title(title)
        plt.colorbar()

    def plot(self, map_name, **kwargs):
        info = self.read_map_info(map_name)
        pixelization = info["pixelization"]
        if pixelization == "gnomonic":
            return self.plot_gnomonic(map_name, **kwargs)
        if pixelization == "healpix":
            return self.plot_healpix(map_name, **kwargs)
        raise ValueError(f"Unknown map pixelization type {pixelization}")


class LensingNoiseMaps(MapsFile):
    required_datasets = []

    def read_rotation(self, realization_index, bin_index):
        g1_name = f"rotation_{realization_index}/g1_{bin_index}"
        g2_name = f"rotation_{realization_index}/g2_{bin_index}"

        g1 = self.read_map(g1_name)
        g2 = self.read_map(g2_name)

        return g1, g2

    def number_of_realizations(self):
        info = self.fileObj["maps"].attrs
        lensing_realizations = info["lensing_realizations"]
        return lensing_realizations


class ClusteringNoiseMaps(MapsFile):
    def read_density_split(self, realization_index, bin_index):
        rho1_name = f"split_{realization_index}/rho1_{bin_index}"
        rho2_name = f"split_{realization_index}/rho2_{bin_index}"
        rho1 = self.read_map(rho1_name)
        rho2 = self.read_map(rho2_name)
        return rho1, rho2

    def number_of_realizations(self):
        info = self.fileObj["maps"].attrs
        clustering_realizations = info["clustering_realizations"]
        return clustering_realizations


class PhotozPDFFile(HDFFile):
    required_datasets = []


class CSVFile(DataFile):
    suffix = "csv"

    def save_file(self, name, dataframe):
        dataframe.to_csv(name)


class SACCFile(DataFile):
    suffix = "sacc"

    @classmethod
    def _open(cls, path, mode, **kwargs):
        import sacc  # pylint: disable=import-outside-toplevel
        if mode == "w":
            raise ValueError(
                "Do not use the open_output method to write sacc files.  Use sacc.write_fits"
            )
        return sacc.Sacc.load_fits(path)

    def read_provenance(self):
        meta = self.fileObj.metadata
        provenance = {
            "uuid": meta.get("provenance/uuid", "UNKNOWN"),
            "creation": meta.get("provenance/creation", "UNKNOWN"),
            "domain": meta.get("provenance/domain", "UNKNOWN"),
            "username": meta.get("provenance/username", "UNKNOWN"),
        }

        return provenance

    @classmethod
    def _close(cls, fileObj):
        pass


class NOfZFile(HDFFile):

    # Must have at least one bin in
    required_datasets = []

    def get_nbin(self, kind):
        return self.fileObj["n_of_z"][kind].attrs["nbin"]

    def get_n_of_z(self, kind, bin_index):
        group = self.fileObj["n_of_z"][kind]
        z = group["z"][:]
        nz = group[f"bin_{bin_index}"][:]
        return (z, nz)

    def get_n_of_z_spline(self, bin_index, kind="cubic", **kwargs):
        import scipy.interpolate  # pylint: disable=import-outside-toplevel

        z, nz = self.get_n_of_z(kind, bin_index)
        spline = scipy.interpolate.interp1d(z, nz, kind=kind, **kwargs)
        return spline

    def save_plot(self, filename, **fig_kw):
        import matplotlib.pyplot as plt  # pylint: disable=import-outside-toplevel

        plt.figure(**fig_kw)
        self.plot()
        plt.legend()
        plt.savefig(filename)
        plt.close()

    def plot(self, kind="cubic"):
        import matplotlib.pyplot as plt  # pylint: disable=import-outside-toplevel

        for b in range(self.get_nbin(kind)):
            z, nz = self.get_n_of_z(kind, b)
            plt.plot(z, nz, label=f"Bin {b}")


class FiducialCosmology(YamlFile):

    # TODO replace when CCL has more complete serialization tools.
    def to_ccl(self, **kwargs):
        import pyccl as ccl  # pylint: disable=import-outside-toplevel

        with open(self.path, "r") as fp:
            params = yaml.load(fp, Loader=yaml.Loader)

        # Now we assemble an init for the object since the CCL YAML has
        # extra info we don't need and different formatting.
        inits = dict(
            Omega_c=params["Omega_c"],
            Omega_b=params["Omega_b"],
            h=params["h"],
            n_s=params["n_s"],
            sigma8=None if params["sigma8"] == "nan" else params["sigma8"],
            A_s=None if params["A_s"] == "nan" else params["A_s"],
            Omega_k=params["Omega_k"],
            Neff=params["Neff"],
            w0=params["w0"],
            wa=params["wa"],
            bcm_log10Mc=params["bcm_log10Mc"],
            bcm_etab=params["bcm_etab"],
            bcm_ks=params["bcm_ks"],
            mu_0=params["mu_0"],
            sigma_0=params["sigma_0"],
        )

        if "z_mg" in params:
            inits["z_mg"] = params["z_mg"]
            inits["df_mg"] = params["df_mg"]

        if "m_nu" in params:
            inits["m_nu"] = params["m_nu"]
            inits["m_nu_type"] = "list"

        inits.update(kwargs)

        return ccl.Cosmology(**inits)

class QPFile(DataFile):
    # TODO: Flesh this out
    suffix = "hdf5"


class DiagnosticMaps(HDFFile):
    required_datasets = [
        'maps/depth/value',
        'maps/depth/pixel',
        ]

    def read_healpix(self, map_name, return_all=False):
        import healpy  # pylint: disable=import-outside-toplevel
        group = self.fileObj[f'maps/{map_name}']
        nside = group.attrs['nside']
        npix = healpy.nside2npix(nside)
        m = np.repeat(healpy.UNSEEN, npix)
        pix = group['pixel'][:]
        val = group['value'][:]
        m[pix] = val
        if return_all:
            return m, pix, nside
        return m

    def display_healpix(self, map_name, **kwargs):
        import healpy  # pylint: disable=import-outside-toplevel
        m, pix, nside = self.read_healpix(map_name, return_all=True)
        lon,lat=healpy.pix2ang(nside,pix,lonlat=True)
        lon_range = [lon.min()-0.1, lon.max()+0.1]
        lat_range = [lat.min()-0.1, lat.max()+0.1]
        title = kwargs.pop('title', map_name)
        healpy.cartview(m,lonra=lon_range, latra=lat_range, title=title, **kwargs)

    def read_tangential(self, map_name):
        group = self.fileObj[f'maps/{map_name}']
        info = dict(group.attrs)
        nx = info['nx']
        ny = info['ny']
        print(ny,nx)
        m = np.zeros((ny,nx))
        m[:,:] = np.nan

        pix = group['pixel'][:]
        val = group['value'][:]
        w = np.where(pix!=-9999)
        pix = pix[w]
        val = val[w]
        x = pix % nx
        y = pix // nx
        m[y,x] = val
        ra_min, ra_max = info['ra_min'], info['ra_max']
        print(ra_min, ra_max)
        if ra_min > 180 and ra_max < 180:
            ra_min -= 360
        ra_range = (ra_min, ra_max)
        dec_range = (info['dec_min'],info['dec_max'])
        return m, ra_range, dec_range

    def display_tangential(self, map_name, **kwargs):
        import pylab  # pylint: disable=import-outside-toplevel
        m, ra_range, dec_range = self.read_tangential(map_name)
        extent = list(ra_range) + list(dec_range)
        title = kwargs.pop('title', map_name)
        pylab.imshow(m, aspect='equal', extent=extent, **kwargs)
        pylab.title(title)
        pylab.colorbar()
        pylab.show()
