import os
import pytest
import pickle
import numpy as np
import pandas as pd
from types import GeneratorType

from ceci.data import DataStore, DataHandle, _DATA_STORE
import descformats
from descformats.base import TableHandle, Hdf5Handle, FitsHandle, PqHandle

DATADIR = os.path.abspath(os.path.join(os.path.dirname(descformats.__file__), 'data'))

#def test_data_file():    
#    with pytest.raises(ValueError) as errinfo:
#        df = DataFile('dummy', 'x')
    

def do_data_handle(datapath, handle_class):

    _DATA_STORE.clear()

    th = handle_class('data', path=datapath)

    with pytest.raises(ValueError) as errinfo:
        th.write()

    assert not th.has_data
    with pytest.raises(ValueError) as errinfo:
        th.write_chunk(0, 1)        
    assert th.has_path
    assert th.is_written
    data = th.read()
    data2 = th.read()

    assert data is data2
    assert th.has_data
    assert th.make_name('data') == f'data.{handle_class.suffix}'
    
    th2 = handle_class('data2', data=data)
    assert th2.has_data
    assert not th2.has_path
    assert not th2.is_written
    with pytest.raises(ValueError) as errinfo:
        th2.open(mode='w')
    with pytest.raises(ValueError) as errinfo:
        th2.write()
    with pytest.raises(ValueError) as errinfo:
        th2.write_chunk(0, 1)
        
    assert th2.make_name('data2') == f'data2.{handle_class.suffix}'
    assert str(th)
    assert str(th2)
    return th


def test_pq_handle():
    datapath = os.path.join(DATADIR, 'testdata', 'test_dc2_training_9816.pq')
    handle = do_data_handle(datapath, PqHandle)
    pqfile = handle.open(mode='r')
    assert pqfile
    assert handle.fileObj is not None
    handle.close()
    assert handle.fileObj is None

    
def test_hdf5_handle():
    datapath = os.path.join(DATADIR, 'testdata', 'test_dc2_training_9816.hdf5')
    handle = do_data_handle(datapath, Hdf5Handle)
    with handle.open(mode='r') as f:
        assert f
        assert handle.fileObj is not None
    datapath_chunked = os.path.join(DATADIR, 'testdata', 'test_dc2_training_9816_chunked.hdf5')
    handle_chunked = Hdf5Handle("chunked", handle.data, path=datapath_chunked)
    from tables_io.arrayUtils import getGroupInputDataLength, sliceDict, getInitializationForODict
    num_rows = len(handle.data['photometry']['id'])
    check_num_rows = len(handle()['photometry']['id'])
    assert num_rows == check_num_rows
    chunk_size = 1000
    data = handle.data['photometry']
    init_dict = getInitializationForODict(data)
    with handle_chunked.open(mode='w') as fout:
        for k, v in init_dict.items():
            fout.create_dataset(k, v[0], v[1])
        for i in range(0, num_rows, chunk_size):
            start = i
            end = i+chunk_size
            if end > num_rows:
                end = num_rows
            handle_chunked.set_data(sliceDict(handle.data['photometry'], slice(start, end)), partial=True)
            handle_chunked.write_chunk(start, end)
    write_size = handle_chunked.size()
    assert len(handle_chunked.data) <= 1000
    data_called = handle_chunked()
    assert len(data_called['id']) == write_size
    read_chunked = Hdf5Handle("read_chunked", None, path=datapath_chunked)
    data_check = read_chunked.read()
    assert np.allclose(data['id'], data_check['id'])
    assert np.allclose(data_called['id'], data_check['id'])
    os.remove(datapath_chunked)


def test_fits_handle():
    datapath = os.path.join(DATADIR, 'testdata', 'output_BPZ_lite.fits')
    handle = do_data_handle(datapath, FitsHandle)
    fitsfile = handle.open(mode='r')
    assert fitsfile
    assert handle.fileObj is not None
    handle.close()
    assert handle.fileObj is None

