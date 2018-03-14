from ..base import FitsFile, HDFFile, DataFile, YamlFile

def metacalibration_names(names):
    """
    Generate the metacalibrated variants of the inputs names,
    that is, variants with _1p, _1m, _2p, and _2m on the end
    of each name.
    """
    suffices = ['1p', '1m', '2p', '2m']
    out = []
    for name in names:
        out += [name + '_' + s for s in suffices]
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
    required_columns = ( metacal_columns + 
                        + metacalibration_names(metacal_columns) 
                        + other_columns )

    # Add methods for handling here ...


class TomographyCatalog(HDFFile):
    required_datasets = ['tomography/bin']

class RandomsCatalog(HDFFile):
    required_datasets = ['randoms/ra', 'randoms/dec', 'randoms/e1', 'randoms/e2']

class DiagnosticMaps(FitsFile):
    pass

class PhotozPDFFile(HDFFile):
    required_datasets = []


class SACCFile(DataFile):
    suffix = 'sacc'