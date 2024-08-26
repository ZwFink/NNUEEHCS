import pytest
from nnueehcs.data_utils import (HDF5Dataset,
                                 ARFFDataSet,
                                 CharacterDelimitedDataset,
                                 read_dataset_from_yaml
                                 )
import torch
import io
import os


class FixtureHDF5File:
    def __init__(self, filename: str, group_name: str,
                 input_dset: str, output_dset: str,
                 shape: tuple,
                 dtype: str):
        self.delete_file(filename)
        self.filename = filename
        self.group_name = group_name
        self.input_dset = input_dset
        self.output_dset = output_dset
        self.shape = shape
        self.dtype = dtype

        self.create_dataset()

    def create_dataset(self):
        import h5py
        import numpy as np
        with h5py.File(self.filename, 'w') as f:
            group = f.create_group(self.group_name)
            ipt = group.create_dataset(self.input_dset,
                                       self.shape,
                                       dtype=self.dtype)
            opt = group.create_dataset(self.output_dset,
                                       self.shape,
                                       dtype=self.dtype)

            ipt[...] = np.random.rand(*self.shape)
            opt[...] = np.random.rand(*self.shape)

            self.ipt_groundtruth = ipt[...].copy()
            self.opt_groundtruth = opt[...].copy()

    def __del__(self):
        self.delete_file(self.filename)

    def delete_file(self, filename):
        if os.path.exists(filename):
            os.remove(filename)


@pytest.fixture()
def hdf5file_fixture():
    return FixtureHDF5File('test.hdf5', 'test', 'input', 'output', (100, 10, 10), 'float32')


@pytest.fixture()
def hdf5dataset_fixture(hdf5file_fixture):
    filename = hdf5file_fixture.filename
    group_name = hdf5file_fixture.group_name
    input_dset = hdf5file_fixture.input_dset
    output_dset = hdf5file_fixture.output_dset

    dset = HDF5Dataset(filename, group_name, input_dset, output_dset)
    dset.ipt_groundtruth = hdf5file_fixture.ipt_groundtruth
    dset.opt_groundtruth = hdf5file_fixture.opt_groundtruth
    return dset


@pytest.fixture()
def datafile_yaml():
    return io.StringIO("""
    test:
      type: hdf5
      file_path: test.hdf5
      group_name: test
      input_dataset: input
      output_dataset: output
    arff_test:
        type: arff
        file_path: test.arff
    delim_test:
        type: character_delimited
        file_path: test.ssv
        delimiter: '\s+'
    failure_test:
        type: unknown
    """)


def test_hdf5datasetreader(datafile_yaml, hdf5dataset_fixture):
    dset = read_dataset_from_yaml(datafile_yaml, 'test')
    gtruth_ipt = hdf5dataset_fixture.ipt_groundtruth
    gtruth_opt = hdf5dataset_fixture.opt_groundtruth

    assert (dset.input == gtruth_ipt).all()
    assert (dset.output == gtruth_opt).all()


def test_hdf5datasetreader_failure(datafile_yaml):
    with pytest.raises(ValueError):
        read_dataset_from_yaml(datafile_yaml, 'failure_test')


def test_hdf5dataset(hdf5dataset_fixture):
    dset = hdf5dataset_fixture
    gtruth_ipt = hdf5dataset_fixture.ipt_groundtruth
    gtruth_opt = hdf5dataset_fixture.opt_groundtruth

    assert (dset.input == gtruth_ipt).all()
    assert (dset.output == gtruth_opt).all()


def test_hdf5dataset_len(hdf5dataset_fixture):
    dset = hdf5dataset_fixture
    assert len(dset) == 100


def test_hdf5dataset_getitem(hdf5dataset_fixture):
    dset = hdf5dataset_fixture
    for i in range(len(dset)):
        ipt, opt = dset[i]
        assert (ipt == torch.tensor(dset.input[i])).all()
        assert (opt == torch.tensor(dset.output[i])).all()


def test_hdf5dataset_input_as_torch_tensor(hdf5dataset_fixture):
    dset = hdf5dataset_fixture
    ipt = dset.input_as_torch_tensor()
    assert (ipt == torch.tensor(dset.input)).all()


def test_hdf5dataset_slice(hdf5dataset_fixture):
    dset = hdf5dataset_fixture
    slc = slice(0, 10, 2)
    dset_slice = dset[slc]
    assert len(dset_slice[0]) == 5
    assert len(dset_slice[1]) == 5
    assert (dset_slice[0] == torch.tensor(dset.input[0:10:2])).all()
    assert (dset_slice[1] == torch.tensor(dset.output[0:10:2])).all()


@pytest.fixture()
def arff_file_fixture():
    val = io.StringIO("""
@relation ailerons
@attribute 'climbRate' real
@attribute 'Sgz' real
@attribute 'p' real
@attribute 'q' real
@data
0.000000,0.000000,0.000000,0.000000
1.111111,1.111111,1.111111,1.111111
2.222222,2.222222,2.222222,2.222222
3.333333,3.333333,3.333333,3.333333
4.444444,4.444444,4.444444,4.444444
5.555555,5.555555,5.555555,5.555555
6.666666,6.666666,6.666666,6.666666
""")
    return val


def test_arffdatasetreader(arff_file_fixture):
    dset = ARFFDataSet(arff_file_fixture)
    gtruth_ipt = torch.tensor([[0.0, 0.0, 0.0],
                               [1.111111, 1.111111, 1.111111],
                               [2.222222, 2.222222, 2.222222],
                               [3.333333, 3.333333, 3.333333],
                               [4.444444, 4.444444, 4.444444],
                               [5.555555, 5.555555, 5.555555],
                               [6.666666, 6.666666, 6.666666]
                               ],
                               dtype=torch.float64)
    gtruth_opt = torch.tensor([0.0, 1.111111, 2.222222, 3.333333, 4.444444, 5.555555, 6.666666],
                              dtype=torch.float64)

    assert (dset.input == gtruth_ipt).all()
    assert (dset.output == gtruth_opt).all()


@pytest.fixture()
def arff_dataset_fixture(arff_file_fixture):
    return ARFFDataSet(arff_file_fixture)

@pytest.fixture()
def arff_ground_truth(arff_dataset_fixture):
    return arff_dataset_fixture.input, arff_dataset_fixture.output


def test_arff_slice(arff_dataset_fixture):
    dset = arff_dataset_fixture
    slc = slice(0, 5, 2)
    dset_slice = dset[slc]
    assert len(dset_slice[0]) == 3
    assert len(dset_slice[1]) == 3
    assert (dset_slice[0] == torch.tensor(dset.input[0:5:2])).all()
    assert (dset_slice[1] == torch.tensor(dset.output[0:5:2])).all()


def test_arff_datasetreader(datafile_yaml, arff_file_fixture, arff_ground_truth):
    # safe the fixture to file
    with open('test.arff', 'w') as f:
        arff_file_fixture.seek(0)
        f.write(arff_file_fixture.read())

    dset = read_dataset_from_yaml(datafile_yaml, 'arff_test')

    gtruth_ipt = arff_ground_truth[0]
    gtruth_opt = arff_ground_truth[1]
    assert (dset.input == gtruth_ipt).all()
    assert (dset.output == gtruth_opt).all()

    os.remove('test.arff')


@pytest.fixture()
def space_delim_file_fixture():
    val = io.StringIO("""0.000000   0.000000   0.000000   0.000000
1.111111   1.111111   1.111111   1.111111
2.222222   2.222222   2.222222   2.222222
3.333333   3.333333   3.333333   3.333333
4.444444   4.444444   4.444444   4.444444
5.555555   5.555555   5.555555   5.555555
6.666666   6.666666   6.666666   6.666666
""")
    return val

@pytest.fixture()
def space_delim_file_fixture_header():
    val = io.StringIO("""h1   h2   h3   h4
0.000000   0.000000   0.000000   0.000000
1.111111   1.111111   1.111111   1.111111
2.222222   2.222222   2.222222   2.222222
3.333333   3.333333   3.333333   3.333333
4.444444   4.444444   4.444444   4.444444
5.555555   5.555555   5.555555   5.555555
6.666666   6.666666   6.666666   6.666666
""")
    return val


@pytest.fixture()
def comma_delim_file_fixture_header():
    val = io.StringIO("""h1,h2,h3,h4
0.000000,0.000000,0.000000,0.000000
1.111111,1.111111,1.111111,1.111111
2.222222,2.222222,2.222222,2.222222
3.333333,3.333333,3.333333,3.333333
4.444444,4.444444,4.444444,4.444444
5.555555,5.555555,5.555555,5.555555
6.666666,6.666666,6.666666,6.666666
""")
    return val


def test_space_delim_dataset(space_delim_file_fixture):
    dset = CharacterDelimitedDataset(space_delim_file_fixture, '\s+')
    space_delim_file_fixture.seek(0)
    assert dset.file_has_header(space_delim_file_fixture, '\s+') == False


def test_space_delim_dataset_header(space_delim_file_fixture_header):
    dset = CharacterDelimitedDataset(space_delim_file_fixture_header, '\s+')
    space_delim_file_fixture_header.seek(0)
    assert dset.file_has_header(space_delim_file_fixture_header, '\s+') == True


def test_comma_delim_dataset_header(comma_delim_file_fixture_header):
    dset = CharacterDelimitedDataset(comma_delim_file_fixture_header, ',')
    comma_delim_file_fixture_header.seek(0)
    assert dset.file_has_header(comma_delim_file_fixture_header, ',') == True