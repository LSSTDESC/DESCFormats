import pathlib
import shutil
import warnings
import os
from io import UnsupportedOperation
import tables_io
import pickle
import qp
import yaml
try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:  # pragma: no cover
    from yaml import Loader, Dumper

from .handle import DataHandle
from .data_file import DataFile
from .provenance.provenance import Provenance


class FileValidationError(Exception):
    pass


class TableHandle(DataHandle):
    """DataHandle for single tables of data
    """
    suffix = "hdf5"

    @classmethod
    def _open(cls, path, mode, **kwargs):
        """Open and return the associated file

        Notes
        -----
        This will simply open the file and return a file-like object to the caller.
        It will not read or cache the data
        """
        return tables_io.io.io_open(path, **kwargs)  #pylint: disable=no-member

    @classmethod
    def _read(cls, path, **kwargs):
        """Read and return the data from the associated file """
        return tables_io.read(path, **kwargs)

    @classmethod
    def _write(cls, data, path, **kwargs):
        """Write the data to the associatied file """
        return tables_io.write(data, path, **kwargs)

    @classmethod
    def _size(cls, path, **kwargs):
        return tables_io.io.getInputDataLengthHdf5(path, **kwargs)

    @classmethod
    def _iterator(cls, path, **kwargs):
        """Iterate over the data"""
        return tables_io.iteratorNative(path, **kwargs)


class Hdf5Handle(TableHandle):
    """A data file in the HDF5 format.
    Using these files requires the h5py package, which in turn
    requires an HDF5 library installation.
    """
    suffix = 'hdf5'
    format = "http://edamontology.org/format_3590"    
    required_datasets = []

    @classmethod
    def _open(cls, path, mode, **kwargs):
        # Suppress a warning that h5py always displays
        # on import
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            import h5py  # pylint: disable=import-outside-toplevel
        # Return an open h5py File
        return h5py.File(path, mode, **kwargs)

    def validate(self):
        if self.data is not None:
            return
        if self.fileObj is None:
            self.open(mode='r')
        missing = [name for name in self.required_datasets if name not in self.fileObj]
        if missing:
            text = "\n".join(missing)
            raise FileValidationError(f"These data sets are missing from HDF file {self.path}:\n{text}")

    @classmethod
    def _write_chunk(cls, data, fileObj, groups, start, end, **kwargs):
        if groups is None:
            tables_io.io.writeDictToHdf5ChunkSingle(fileObj, data, start, end, **kwargs)
        else:  # pragma: no cover
            tables_io.io.writeDictToHdf5Chunk(groups, data, start, end, **kwargs)


class FitsHandle(TableHandle):
    """DataHandle for a table written to fits"""
    suffix = 'fits'
    format = "http://edamontology.org/format_2333"    
    required_columns = []

    @classmethod
    def _open(cls, path, mode, **kwargs):
        import fitsio   # pylint: disable=import-outside-toplevel
        # Fitsio doesn't have pure 'w' modes, just 'rw'.
        # Maybe we should check if the file already exists here?
        if mode == 'w':
            mode = 'rw'
        return fitsio.FITS(path, mode=mode, **kwargs)

    def missing_columns(self, columns, hdu=1):
        """
        Check that all supplied columns exist
        and are in the chosen HDU
        """
        if self.data is not None:
            ext = self.data
            found_cols = list(ext.keys())
        elif self.fileObj is None:
            self.open(mode='r')
            ext = self.fileObj[hdu]
            found_cols = ext.get_colnames()
        missing_columns = [col for col in columns if col not in found_cols]
        return missing_columns

    def validate(self):
        """Check that the catalog has all the required columns and complain otherwise"""
        # Find any columns that do not exist in the file

        missing = self.missing_columns(self.required_columns)

        # If there are any, raise an exception that lists them explicitly
        if missing:
            text = "\n".join(missing)
            raise FileValidationError(f"These columns are missing from FITS file {self.path}:\n{text}")


class PqHandle(TableHandle):
    """DataHandle for a parquet table"""
    suffix = 'pq'

    @classmethod
    def _close(cls, fileObj, **kwargs):
        pass


class QPHandle(DataHandle):
    """DataHandle for qp ensembles
    """
    suffix = 'hdf5'

    @classmethod
    def _open(cls, path, mode, **kwargs):
        """Open and return the associated file
        Notes
        -----
        This will simply open the file and return a file-like object to the caller.
        It will not read or cache the data
        """
        return tables_io.io.io_open(path, mode=mode, **kwargs)  #pylint: disable=no-member

    @classmethod
    def _read(cls, path, **kwargs):
        """Read and return the data from the associated file """
        return qp.read(path)

    @classmethod
    def _write(cls, data, path, **kwargs):
        """Write the data to the associatied file """
        return data.write_to(path)

    @classmethod
    def _initialize_write(cls, data, path, data_lenght, **kwargs):
        comm = kwargs.get('communicator', None)
        return data.initializeHdf5Write(path, data_lenght, comm)

    @classmethod
    def _write_chunk(cls, data, fileObj, groups, start, end, **kwargs):
        return data.writeHdf5Chunk(fileObj, start, end)

    @classmethod
    def _finalize_write(cls, data, fileObj, **kwargs):
        return data.finalizeHdf5Write(fileObj)


class HDFFile(DataFile):
    supports_parallel_write = True
    """
    A data file in the HDF5 format.
    Using these files requires the h5py package, which in turn
    requires an HDF5 library installation.

    """
    suffix = "hdf5"
    required_datasets = []

    @classmethod
    def _open(cls, path, mode, **kwargs):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            import h5py
        # Return an open h5py File
        fileObj = h5py.File(path, mode, **kwargs)
        return fileObj

    def validate(self):
        missing = [name for name in self.required_datasets if name not in self.fileObj]
        if missing:
            text = "\n".join(missing)
            raise FileValidationError(
                f"These data sets are missing from HDF file {self.path}:\n{text}"
            )

    def close(self):
        self.fileObj.close()


class FitsFile(DataFile):
    """
    A data file in the FITS format.
    Using these files requires the fitsio package.
    """

    suffix = "fits"
    required_columns = []

    @classmethod
    def _open(cls, path, mode, **kwargs):
        import fitsio

        # Fitsio doesn't have pure 'w' modes, just 'rw'.
        # Maybe we should check if the file already exists here?
        if mode == "w":
            mode = "rw"
        fileObj = fitsio.FITS(path, mode=mode, **kwargs)
        return fileObj
        
    def missing_columns(self, columns, hdu=1):
        """
        Check that all supplied columns exist
        and are in the chosen HDU
        """
        ext = self.fileObj[hdu]
        found_cols = ext.get_colnames()
        missing_columns = [col for col in columns if col not in found_cols]
        return missing_columns

    def validate(self):
        """Check that the catalog has all the required columns and complain otherwise"""
        # Find any columns that do not exist in the file
        missing = self.missing_columns(self.required_columns)

        # If there are any, raise an exception that lists them explicitly
        if missing:
            text = "\n".join(missing)
            raise FileValidationError(
                f"These columns are missing from FITS file {self.path}:\n{text}"
            )

    def close(self):
        self.fileObj.close()


class TextFile(DataFile):
    """
    A data file in plain text format.
    """
    suffix = 'txt'
    

class YamlFile(DataFile):
    """
    A data file in yaml format.
    """
    suffix = 'yml'

    def __init__(self, tag, path=None, mode='r', extra_provenance=None, validate=True, **kwargs):
        if mode=='w':
            mode='a'
        DataFile.__init__(self, tag, path=path, mode=mode, extra_provenance=extra_provenance, validate=validate, **kwargs)

    def open(self, **kwargs):
        DataFile.open(self, **kwargs)
        if self.mode == 'r':
            load_mode = kwargs.get('load_mode', 'full')
            if load_mode == "safe":
                self.content = yaml.safe_load(self.fileObj)
            elif load_mode == "full":
                self.content = yaml.full_load(self.fileObj)
            elif load_mode == "unsafe":
                self.content = yaml.unsafe_load(self.fileObj)
            else:
                raise ValueError(
                    f"Unknown value {load_mode} of load_mode. "
                    "Should be 'safe', 'full', or 'unsafe'"
                )
            self.content.pop('provenance', None)
            
    def read(self, key):
        return self.content[key]

    def write(self, d):
        if not isinstance(d, dict):
            raise ValueError("Only dicts should be passed to YamlFile.write")
        yaml.dump(d, self.fileObj)


class Directory(DataFile):
    suffix = ''

    @classmethod
    def _open(cls, path, mode, **kwargs):
        p = pathlib.Path(path)

        if mode == "w":
            if p.exists():
                shutil.rmtree(p)
            p.mkdir(parents=True)
        else:
            if not p.is_dir():
                raise ValueError(f"Directory input {path} does not exist")
        return p

    @classmethod
    def _close(cls, fileObj, **kwargs):
        pass


    def write_provenance(self):
        """
        Write provenance information to a new group,
        called 'provenance'
        """
        self.provenance.write(os.path.join(self.path, "provenance.yaml"))

    def read_provenance(self):
        self.provenance = Provenance()
        self.provenance.read(os.path.join(self.path, "provenance.yaml"))
        return self.provenance


class FileCollection(Directory):
    """
    Represents a grouped bundle of files, for cases where you don't
    know the exact list in advance.
    """

    suffix = ""

    def write_listing(self, filenames):
        """
        Write a listing file in the directory recording
        (presumably) the filenames put in it.
        """
        fn = self.path_for_file("txpipe_listing.txt")
        with open(fn, "w") as f:
            yaml.dump(filenames, f)

    def read_listing(self):
        """
        Read a listing file from the directory.
        """
        fn = self.path_for_file("txpipe_listing.txt")
        with open(fn, "r") as f:
            filenames = yaml.safe_load(f)
        return filenames

    def path_for_file(self, filename):
        """
        Get the path for a file inside the collection.
        Does not check if the file exists or anything like
        that.
        """
        return str(self.fileObj / filename)


class PNGFile(DataFile):
    suffix = "png"

    @classmethod
    def _open(cls, path, mode, **kwargs):
        import matplotlib
        import matplotlib.pyplot as plt

        if mode != "w":
            raise ValueError("Reading existing PNG files is not supported")
        return plt.figure(**kwargs)

    def close(self, **kwargs):
        import matplotlib
        import matplotlib.pyplot as plt        
        self.fileObj.savefig(self.path, metadata=self.provenance.to_string_dict())
        plt.close(self.fileObj)

    def write_provenance(self):
        # provenance is written on closing the file
        pass

    def read_provenance(self):
        raise ValueError("Reading existing PNG files is not supported")


class PickleFile(DataFile):
    suffix = "pkl"

    @classmethod
    def _open(cls, path, mode, **kwargs):
        return open(path, mode + "b")

    def write(self, obj):
        if self.mode != "w":
            raise UnsupportedOperation(
                "Cannot write to pickle file opened in " f"read-only ({self.mode})"
            )
        pickle.dump(obj, self.fileObj)

    def read(self):
        if self.mode != "r":
            raise UnsupportedOperation(
                "Cannot read from pickle file opened in " f"write-only ({self.mode})"
            )
        return pickle.load(self.fileObj)


class ParquetFile(DataFile):
    suffiz = "pq"

    @classmethod
    def _open(cls, path, mode, **kwargs):
        import pyarrow.parquet
        if mode != "r":
            raise NotImplementedError("Not implemented writing to Parquet")
        return pyarrow.parquet.ParquetFile(path)

    @classmethod
    def _close(cls, fileObj, **kwargs):
        pass
