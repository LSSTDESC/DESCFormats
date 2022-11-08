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
    HDFFile,
    FitsHandle,
    FitsFile,
    PqHandle,
    ParquetFile,
    TextFile,
    YamlFile,
    QPHandle,
    FileValidationError,
    Directory,
    FileCollection,
    PNGFile,
)
from descformats.data_store import DataStore, DATA_STORE
from descformats.handle import DataHandle

DATADIR = os.path.abspath(os.path.join(os.path.dirname(descformats.__file__), 'data'))

#def test_data_file():
#    with pytest.raises(ValueError) as errinfo:
#        df = DataFile('dummy', 'x')


def do_data_file(datapath, file_class):

    DATA_STORE().clear()

    with pytest.raises(ValueError) as errinfo:
        fail = file_class('bad', mode='r')
    tf = file_class('data', path=datapath)
    print(tf)
    assert tf.has_path
    assert tf.is_written
    return tf
    

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


def test_pq_file():
    datapath = os.path.join(DATADIR, 'testdata', 'test_dc2_training_9816.pq')
    pqfile = do_data_file(datapath, ParquetFile)

    with ParquetFile("pqfile", path=datapath, mode='r') as pqf:
        assert pqf
    

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
    #data_called = handle_chunked()
    #assert len(data_called['id']) == write_size
    read_chunked = Hdf5Handle("read_chunked", None, path=datapath_chunked)
    data_check = read_chunked.read()
    assert np.allclose(data['id'], data_check['id'])
    #assert np.allclose(data_called['id'], data_check['id'])
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


def test_fits_file():
    datapath = os.path.join(DATADIR, 'testdata', 'output_BPZ_lite.fits')
    handle = FitsHandle("data", path=datapath)
    fitsfile = handle.open(mode='r')

    class BadFitsFile(FitsFile):
        required_columns = ["missing"]
    
    with tempfile.TemporaryDirectory() as dirname:
        fitsfile_out = FitsFile("fits_out", path=os.path.join(dirname, "fits_out.fits"), mode='w')
        fitsfile_out.fileObj.create_table_hdu(fitsfile[1].read())
        fitsfile_out.close()
        fitsfile_read = FitsFile("fits_read", path=os.path.join(dirname, "fits_out.fits"), mode='r')
        assert np.allclose(fitsfile_read.fileObj[2].read()['xvals'], fitsfile[1].read()['xvals'])
        fitsfile_read.close()
        with pytest.raises(FileValidationError):
            fitsfile_bad = BadFitsFile("fits_bad", path=os.path.join(dirname, "fits_out.fits"), mode='r')

def test_hdf5_file():
    datapath = os.path.join(DATADIR, 'testdata', 'test_dc2_training_9816.hdf5')
    handle = Hdf5Handle("data", path=datapath)
    hdffile = handle.open(mode='r')

    class BadHDFFile(HDFFile):
        required_datasets = ["missing"]
    
    with tempfile.TemporaryDirectory() as dirname:
        hdffile_out = HDFFile("hdf_out", path=os.path.join(dirname, "hdf_out.hdf"), mode='w')
        hdffile_out.close()
        hdffile_read = HDFFile("hdf_read", path=os.path.join(dirname, "hdf_out.hdf"), mode='r')
        #assert np.allclose(hdffile_out.fileObj[2].read()['xvals'], hdffile[1].read()['xvals'])
        hdffile_read.close()
        with pytest.raises(FileValidationError):
            hdffile_bad = BadHDFFile("hdf_bad", path=os.path.join(dirname, "hdf_out.hdf"), mode='r')

def test_text_file():
    datapath = os.path.join(DATADIR, 'testdata', 'OhYouWill.txt')
    textfile = do_data_file(datapath, TextFile)
    assert textfile
    assert textfile.fileObj is not None
    textfile.close()
    assert textfile.fileObj is None


def test_yaml_file():
    datapath = os.path.join(DATADIR, 'testdata', 'something.yaml')
    textfile = do_data_file(datapath, YamlFile)
    assert textfile
    assert textfile.fileObj is not None
    textfile.close()
    assert textfile.fileObj is None

    cc = textfile.content.copy()

    with tempfile.TemporaryDirectory() as dirname:
        tmpyml = os.path.join(dirname, "tempout.yml")
        with YamlFile("ymlout", path=tmpyml, mode='w') as ymlout:
            ymlout.write(cc)
            with pytest.raises(ValueError):
                ymlout.write(232)
        with YamlFile("ymlin", path=tmpyml, mode='r') as ymlin:
            assert ymlin.content == cc
    
    for load_mode in ['full', 'safe', 'unsafe']:
        with YamlFile(f"yaml_{load_mode}", path=datapath, mode='r', load_mode=load_mode) as fin:
            assert fin.content['name'] == 'Upload Python Package'
            assert fin.read('name') == 'Upload Python Package'
            
    with pytest.raises(ValueError):
        will_fail = YamlFile("to_fail", path=datapath, mode='r', load_mode='fail')
        

def test_directory():
    DATA_STORE().clear()
    datapath = os.path.join(DATADIR, 'testdata')
    with Directory("dir", path=datapath) as the_dir:
        assert the_dir
    
    with tempfile.TemporaryDirectory() as dirname:
        with Directory("newdir", path=dirname, mode='w') as newdir:
            assert newdir
            assert os.path.isdir(dirname)
        with Directory("newdir2", path=dirname, mode='r') as newdir:
            assert newdir
            assert os.path.isdir(dirname)

    with pytest.raises(ValueError):
        handle = Directory("dir", path="this/path/better/not/exist")


def test_file_collection():
    DATA_STORE().clear()
    with tempfile.TemporaryDirectory() as dirname:
        with FileCollection("newcoll", path=dirname, mode='w') as newdir:
            assert newdir
            assert os.path.isdir(dirname)
            newdir.write_listing(["a.txt", "b.txt"])

        with FileCollection("newcoll_read", path=dirname, mode='r') as newdir:
            assert newdir
            assert os.path.isdir(dirname)
            listing = newdir.read_listing()

    
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


def test_png_file():
    
    DATA_STORE().clear()

    with tempfile.TemporaryDirectory() as dirname:
        with PNGFile('png', path=os.path.join(dirname, 'test.png'), mode='w') as plt:
            assert plt
            with pytest.raises(ValueError):
                plt.read_provenance()
            
        with pytest.raises(ValueError):
            plt = PNGFile('png', path=os.path.join(dirname, 'test.png'), mode='r')


if __name__ == '__main__':
    test_yaml_file()
