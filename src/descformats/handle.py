import os
import copy

from .base_data import BaseData

class DataHandle(BaseData):
    """Class to act as a handle for a bit of data.  Associating it with a file and
    providing tools to read & write it to that file.  

    These subclasses are used in the definition of pipeline stages
    to indicate what kind of file is expected.  The "suffix" attribute,
    which must be defined on subclasses, indicates the file suffix.

    The open(), read(), iterator() methods can be used to access data in slight different
    ways.

    The write(), initialize_write(), write_chunk() and finalize_write() methods
    can be used to write data in slightly different ways.

    Sub-classes should override the associated protected classmethods such as
    _open(), _read(), _write() etc..

    Parameters
    ----------
    tag : str
        The tag under which this data handle can be found in the store
    data : any or None
        The associated data
    path : str or None
        The path to the associated file
    """
    suffix = ''
    format = "Not Specified"

    def __init__(self, tag, data=None, path=None, extra_provenance=None, **kwargs):
        """Constructor """
        BaseData.__init__(self, tag, path=path)
        self.data = data
        self.extra_provenance = extra_provenance
        self.fileObj = None
        self.groups = None
        self.partial = False

    def open(self, **kwargs):
        """Open and return the associated file

        Notes
        -----
        This will simply open the file and return a file-like object to the caller.
        It will not read or cache the data
        """
        if self.path is None:
            raise ValueError("DataHandle.open() called but path has not been specified")
        self.fileObj = self._open(self.path, kwargs.pop('mode', None), **kwargs)
        return self.fileObj

    @classmethod
    def _open(cls, path, mode, **kwargs):
        raise NotImplementedError("DataHandle._open")  #pragma: no cover

    def close(self, **kwargs):  #pylint: disable=unused-argument
        """Close """
        self._close(self.fileObj, **kwargs)
        self.fileObj = None

    @classmethod
    def _close(cls, fileObj, **kwargs):
        pass

    def read(self, force=False, **kwargs):
        """Read and return the data from the associated file """
        if self.data is not None and not force:
            return self.data
        self.set_data(self._read(self.path, **kwargs))
        return self.data

    @classmethod
    def _read(cls, path, **kwargs):
        raise NotImplementedError("DataHandle._read")  # pragma: no cover

    def write(self, **kwargs):
        """Write the data to the associatied file """
        if self.path is None:
            raise ValueError("DataHandle.write() called but path has not been specified")
        if self.data is None:
            raise ValueError(f"DataHandle.write() called for path {self.path} with no data")
        outdir = os.path.dirname(os.path.abspath(self.path))
        if not os.path.exists(outdir):  # pragma: no cover
            os.makedirs(outdir, exist_ok=True)
        ret_val = self._write(self.data, self.path, **kwargs)
        self.provenance = self.generate_provenance(self.extra_provenance)
        self.write_provenance()
        return ret_val

    def write_provenance(self):
        self.provenance.write(self.path)
    
    @classmethod
    def _write(cls, data, path, **kwargs):
        raise NotImplementedError("DataHandle._write")  #pragma: no cover

    def initialize_write(self, data_lenght, **kwargs):
        """Initialize file to be written by chunks"""
        if self.path is None:  # pragma: no cover
            raise ValueError("DataHandle.write() called but path has not been specified")
        self.groups, self.fileObj = self._initialize_write(self.data, self.path, data_lenght, **kwargs)

    @classmethod
    def _initialize_write(cls, data, path, data_lenght, **kwargs):
        raise NotImplementedError("DataHandle._initialize_write") #pragma: no cover

    def write_chunk(self, start, end, **kwargs):
        """Write the data to the associatied file """
        if self.data is None:
            raise ValueError(f"DataHandle.write_chunk() called for path {self.path} with no data")
        if self.fileObj is None:
            raise ValueError(f"DataHandle.write_chunk() called before open for {self.tag} : {self.path}")
        return self._write_chunk(self.data, self.fileObj, self.groups, start, end, **kwargs)

    @classmethod
    def _write_chunk(cls, data, fileObj, groups, start, end, **kwargs):
        raise NotImplementedError("DataHandle._write_chunk")  #pragma: no cover

    def finalize_write(self, **kwargs):
        """Finalize and close file written by chunks"""
        if self.fileObj is None:  #pragma: no cover
            raise ValueError(f"DataHandle.finalize_wite() called before open for {self.tag} : {self.path}")
        self._finalize_write(self.data, self.fileObj, **kwargs)

    @classmethod
    def _finalize_write(cls, data, fileObj, **kwargs):
        raise NotImplementedError("DataHandle._finalize_write")  #pragma: no cover

    def iterator(self, **kwargs):
        """Iterator over the data"""
        #if self.data is not None:
        #    for i in range(1):
        #        yield i, -1, self.data
        return self._iterator(self.path, **kwargs)

    def set_data(self, data, partial=False):
        """Set the data for a chunk, and set the partial flag to true"""
        self.data = data
        self.partial = partial

    def size(self, **kwargs):
        """Return the size of the data associated to this handle"""
        return self._size(self.path, **kwargs)

    @classmethod
    def _size(cls, path, **kwargs):
        raise NotImplementedError("DataHandle._size")  #pragma: no cover

    @classmethod
    def _iterator(cls, path, **kwargs):
        raise NotImplementedError("DataHandle._iterator")  #pragma: no cover

    @property
    def has_data(self):
        """Return true if the data for this handle are loaded """
        return self.data is not None

    def __str__(self):
        s = f"{type(self)} "
        if self.has_path:
            s += f"{self.path}, ("
        else:
            s += "None, ("
        if self.is_written:
            s += "w"
        if self.has_data:
            s += "d"
        s += ")"
        return s
