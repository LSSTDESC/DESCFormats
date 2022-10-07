import os
import tempfile
import pytest
import pickle
import numpy as np
import pandas as pd
from types import GeneratorType

import descformats
from descformats.base import (
    TableHandle,
    Hdf5Handle,
    FitsHandle,
    PqHandle,
    TextFile,
    YamlFile,
    QPHandle,
    FileValidationError,
    Directory,
)
from descformats.data import DataStore, DATA_STORE
from descformats.handle import DataHandle

DATADIR = os.path.abspath(os.path.join(os.path.dirname(descformats.__file__), 'data'))

#def test_data_file():
#    with pytest.raises(ValueError) as errinfo:
#        df = DataFile('dummy', 'x')


def do_data_handle(datapath, handle_class):

    DATA_STORE().clear()

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
    th.validate()

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
    assert str(th)
    assert str(th2)

    with tempfile.TemporaryDirectory() as dirname:

        assert th2.make_name('data2') == f'data2.{handle_class.suffix}'
        th2.path = os.path.join(dirname, th2.make_name('data2'))
        th2.write()

        th3 = handle_class('data3', path=th2.path)
        th3.validate()

        th4 = handle_class('data4', path=th2.path)
        data4 = th4.read()

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


def test_qp_handle():
    datapath = os.path.join(DATADIR, 'testdata', 'output_BPZ_lite.fits')
    handle = do_data_handle(datapath, QPHandle)
    qpfile = handle.open()
    assert qpfile
    assert handle.fileObj is not None
    handle.close()
    assert handle.fileObj is None

    other_handle = QPHandle("qp", path=datapath)
    other_data = other_handle.read()
    chunk_size = 1000

    with tempfile.TemporaryDirectory() as dirname:
        write_handle = QPHandle("qp_write", path=os.path.join(dirname, 'qp.hdf5'))        
        num_rows = other_handle.data.npdf
        for i in range(0, num_rows, chunk_size):
            start = i
            end = i+chunk_size
            if end > num_rows:
                end = num_rows
            write_handle.set_data(other_data[start:end])
            if i == 0:
                write_handle.initialize_write(num_rows)
            write_handle.write_chunk(start, end)
        write_handle.finalize_write()

        check_reader = QPHandle("qp_read", path=os.path.join(dirname, 'qp.hdf5'))    
        check_data = check_reader.read()
        assert check_data.npdf == other_data.npdf


def test_fits_handle():
    datapath = os.path.join(DATADIR, 'testdata', 'output_BPZ_lite.fits')
    handle = do_data_handle(datapath, FitsHandle)
    fitsfile = handle.open(mode='r')
    assert fitsfile
    assert handle.fileObj is not None
    handle.close()
    assert handle.fileObj is None
    
    with tempfile.TemporaryDirectory() as dirname:
        handle_write = FitsHandle("newfits", path=os.path.join(dirname, "newfile.fits"))
        fitsfile = handle_write.open(mode='w')
        assert fitsfile        
        assert handle_write.fileObj is not None
        handle_write.close()
        assert handle_write.fileObj is None



def test_text_file():
    datapath = os.path.join(DATADIR, 'testdata', 'OhYouWill.txt')
    handle = do_data_handle(datapath, TextFile)
    textfile = handle.open(mode='r')
    assert textfile
    assert handle.fileObj is not None
    handle.close()
    assert handle.fileObj is None


def test_yaml_file():
    datapath = os.path.join(DATADIR, 'testdata', 'something.yaml')
    handle = do_data_handle(datapath, YamlFile)
    textfile = handle.open(mode='r')
    assert textfile
    assert handle.fileObj is not None
    handle.close()
    assert handle.fileObj is None


def test_directory():
    DATA_STORE().clear()
    datapath = os.path.join(DATADIR, 'testdata')
    handle = Directory("dir", path=datapath)

    with handle.open(mode="r") as directrory:
        assert directrory

    with tempfile.TemporaryDirectory() as dirname:
        handle = Directory("newdir", path=dirname)
        with handle.open(mode="w") as new_dir:
            assert new_dir
            assert os.path.isdir(dirname)
        other_handle = Directory("newdir2", path=dirname)
        with other_handle.open(mode="w") as new_dir:
            assert new_dir
            assert os.path.isdir(dirname)

    handle.close()
    handle = Directory("dir", path="this/path/better/not/exist")
    with pytest.raises(ValueError):
        handle.open(mode='r')
        
    
def test_data_hdf5_iter():

    DATA_STORE().clear()

    datapath = os.path.join(DATADIR, 'testdata', 'test_dc2_training_9816.hdf5')

    #data = DS.read_file('data', TableHandle, datapath)
    th = Hdf5Handle('data', path=datapath)
    x = th.iterator(groupname='photometry', chunk_size=1000)

    assert isinstance(x, GeneratorType)
    for i, xx in enumerate(x):
        assert xx[0] == i*1000
        assert xx[1] - xx[0] <= 1000


def test_data_store():

    DS = DATA_STORE()

    DS.clear()
    DS.__class__.allow_overwrite = False

    datapath_hdf5 = os.path.join(DATADIR, 'testdata', 'test_dc2_training_9816.hdf5')
    datapath_pq = os.path.join(DATADIR, 'testdata', 'test_dc2_training_9816.pq')
    datapath_hdf5_copy = os.path.join(DATADIR, 'testdata', 'test_dc2_training_9816_copy.hdf5')
    datapath_pq_copy = os.path.join(DATADIR, 'testdata', 'test_dc2_training_9816_copy.pq')

    DS.add_data('hdf5', None, Hdf5Handle, path=datapath_hdf5)
    DS.add_data('pq', None, PqHandle, path=datapath_pq)

    with DS.open('hdf5') as f:
        assert f

    data_pq = DS.read('pq')
    data_hdf5 = DS.read('hdf5')

    data_pq2 = DS.read_file('pq2', PqHandle, datapath_pq)
    assert (data_pq == data_pq2.data).all().all()
    assert DS.pq2 == data_pq2

    with pytest.raises(KeyError):
        print(DS.baby_shark)
    
    DS.add_data('pq_copy', data_pq, PqHandle, path=datapath_pq_copy)
    DS.add_data('hdf5_copy', data_hdf5, Hdf5Handle, path=datapath_hdf5_copy)
    DS.write('pq_copy')
    DS.write('hdf5_copy')

    with pytest.raises(KeyError) as errinfo:
        DS.read('nope')
    with pytest.raises(KeyError) as errinfo:
        DS.open('nope')
    with pytest.raises(KeyError) as errinfo:
        DS.write('nope')

    with pytest.raises(TypeError) as errinfo:
        DS['nope'] = None
    with pytest.raises(ValueError) as errinfo:
        DS['pq'] = DS['pq']
    with pytest.raises(ValueError) as errinfo:
        DS.pq = DS['pq']

    assert repr(DS)

    DS2 = DataStore(pq=DS.pq)
    assert isinstance(DS2.pq, DataHandle)

    # pop the 'pq' data item to avoid overwriting file under git control
    DS.pop('pq')

    DS.write_all()
    DS.write_all(force=True)

    os.remove(datapath_hdf5_copy)
    os.remove(datapath_pq_copy)


def test_failed_validation():

    class BadHdf5Handle(Hdf5Handle):
        required_datasets = ["baby dinosaur"]

    DATA_STORE().clear()
    datapath_hdf5 = os.path.join(DATADIR, 'testdata', 'test_dc2_training_9816.hdf5')

    th = BadHdf5Handle('data', path=datapath_hdf5)
    with pytest.raises(FileValidationError):
        th.validate()
        
    class BadFitsHandle(FitsHandle):
        required_columns = ["baby dinosaur"]

    DATA_STORE().clear()
    datapath_fits = os.path.join(DATADIR, 'testdata', 'output_BPZ_lite.fits')

    th = BadFitsHandle('data', path=datapath_fits)
    with pytest.raises(FileValidationError):
        th.validate()


if __name__ == '__main__':
    test_data_store()
