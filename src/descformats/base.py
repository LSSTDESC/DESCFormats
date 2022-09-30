import pathlib
import shutil
import warnings

from ceci.handle import DataHandle
import tables_io


class FileValidationError(Exception):
    pass


class DataFile(DataHandle):
    """
    A class representing a DataFile to be made by pipeline stages
    and passed on to subsequent ones.

    DataFile itself should not be instantiated - instead subclasses
    should be defined for different file types.

    These subclasses are used in the definition of pipeline stages
    to indicate what kind of file is expected.  The "suffix" attribute,
    which must be defined on subclasses, indicates the file suffix.

    The open method, which can optionally be overridden, is used by the
    machinery of the PipelineStage class to open an input our output
    named by a tag.

    """

    @classmethod
    def _open(cls, path, mode, **kwargs):
        """
        Open a data file.  The base implementation of this function just
        opens and returns a standard python file object.

        Subclasses can override to either open files using different openers
        (like fitsio.FITS), or, for more specific data types, return an
        instance of the class itself to use as an intermediary for the file.

        """
        return open(path, mode)

    @classmethod
    def _close(cls, fileObj):
        fileObj.close()



class TableHandle(DataHandle):
    """DataHandle for single tables of data
    """
    suffix = None

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


HDFFile = Hdf5Handle


FitsFile = FitsHandle


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
    def _close(cls, fileObj):
        pass
