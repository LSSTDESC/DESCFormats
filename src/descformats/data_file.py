
from .base_data import BaseData
from .provenance.provenance import Provenance

class DataFile(BaseData):
    """
    This inherits from BaseData and provides the only the 
    open() method which acts as a context manager.
    """

    def __init__(self, tag, path=None, mode='r', extra_provenance=None, validate=True, **kwargs):
        """Constructor """
        BaseData.__init__(self, tag, path=path)
        self.fileObj = None
        self.mode = mode
        self.open(**kwargs)
        
        if validate and mode == "r":
            self.validate()

        if mode in ["w", "rw"]:
            self.provenance = self.generate_provenance(extra_provenance)
            self.write_provenance()
        else:
            try:
                self.provenance = self.read_provenance()
            except FileNotFoundError:
                self.provenance = Provenance()

    def open(self, **kwargs):
        """Open and return the associated file

        Notes
        -----
        This will simply open the file and return a file-like object to the caller.
        It will not read or cache the data
        """
        if self.path is None:
            raise ValueError("DataHandle.open() called but path has not been specified")
        self.fileObj = self._open(self.path, kwargs.pop('mode', self.mode), **kwargs)
        return self.fileObj    

    @classmethod
    def _open(cls, path, mode, **kwargs):
        """
        Open a data file.  The base implementation of this function just
        opens and returns a standard python file object.

        Subclasses can override to either open files using different openers
        (like fitsio.FITS), or, for more specific data types, return an
        instance of the class itself to use as an intermediary for the file.

        """
        return open(path, mode, encoding='utf-8')

    def close(self, **kwargs):  #pylint: disable=unused-argument
        """Close """
        self._close(self.fileObj, **kwargs)
        self.fileObj = None
    
    @classmethod
    def _close(cls, fileObj, **kwargs):
        fileObj.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


        
