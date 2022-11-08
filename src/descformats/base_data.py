import os
from .provenance.provenance import Provenance

class BaseData:
    """Class to keep track of a bit of data. 

    This could mean associated it to a path and either:

    1) providing a class to act as a connect manager to open() it as a 
    file-like object.

    2) providing a class to provide IO functionality such as read(), write()
    and iterated versions of read() and write()

    This functionality will be defined in the sub-classes that are used 
    definition of pipeline stages to indicate what kind of file is expected. 

    The "suffix" attribute, which must be defined on subclasses, indicates the file suffix.

    Parameters
    ----------
    tag : str
        The tag under which this data handle can be found in the store
    path : str or None
        The path to the associated file
    creator : str or None
        The name of the stage that created this data handle
    provenance : dict or None
        The provenance information
    """
    suffix = ''
    format = "Not Specified"

    def __init__(self, tag, path=None, **kwargs):
        """Constructor """
        self.tag = tag
        self.path = path
        self.provenance = None

    @staticmethod
    def generate_provenance(extra_provenance=None):
        """
        Generate provenance information - a dictionary
        of useful information about the origina
        """
        provenance = Provenance()
        provenance.generate()
        if extra_provenance:
            provenance.update(extra_provenance)
        return provenance

    def write_provenance(self):
        """
        Concrete subclasses (for which it is possible) should override
        this method to save the dictionary self.provenance to the file.
        """
        self.provenance.write(self.path, self.suffix)

    def read_provenance(self):
        """
        Concrete subclasses for which it is possible should override
        this method and read the provenance information from the file.

        Other classes will return this dictionary of UNKNOWNs
        """
        self.provenance = Provenance()
        try:
            self.provenance.read(self.path)
        except:
            pass
        return self.provenance
 
    @property
    def has_path(self):
        """Return true if the path for the associated file is defined """
        return self.path is not None

    @property
    def is_written(self):
        """Return true if the associated file has been written """
        if self.path is None:
            return False
        return os.path.exists(self.path)

    def __str__(self):
        s = f"{type(self)} "
        if self.has_path:
            s += f"{self.path}, ("
        else:
            s += "None, ("
        if self.is_written:
            s += "w"
        s += ")"
        return s

    @classmethod
    def make_name(cls, tag):
        """Construct and return file name for a particular data tag """
        if cls.suffix:
            return f"{tag}.{cls.suffix}"
        return tag  #pragma: no cover

    def validate(self):
        """Make sure that the data are valid"""

