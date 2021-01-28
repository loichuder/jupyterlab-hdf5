import h5py
import os
import numpy as np
from jupyterlab_hdf.tests.utils import ServerTest


class TestMeta(ServerTest):
    def setUp(self):
        super().setUp()

        with h5py.File(os.path.join(self.notebook_dir, 'test_file.h5'), 'w') as h5file:
            # Empty group
            h5file.create_group('empty_group')

            # Group with 2 children
            grp = h5file.create_group('group_with_children')
            # A simple dataset
            grp['dataset_1'] = np.random.random((2, 3, 4))
            # and a group with attributes
            attr_grp = grp.create_group('group_with_attrs')
            attr_grp.attrs['string_attr'] = 'I am a group'
            attr_grp.attrs['number_attr'] = 5676

    def test_empty_group(self):
        response = self.tester.get(['meta', 'test_file.h5'], params={'uri': '/empty_group'})

        assert response.status_code == 200
        payload = response.json()
        assert payload['name'] == 'empty_group'
        assert payload['type'] == 'group'
        assert type(payload['id']) == int and payload['id'] > 0
        assert payload['attributeCount'] == 0
        assert payload['childrenCount'] == 0

    def test_group_with_children(self):
        response = self.tester.get(['meta', 'test_file.h5'], params={'uri': '/group_with_children'})

        assert response.status_code == 200
        payload = response.json()
        assert payload['name'] == 'group_with_children'
        assert payload['type'] == 'group'
        assert type(payload['id']) == int and payload['id'] > 0
        assert payload['attributeCount'] == 0
        assert payload['childrenCount'] == 2

    def test_group_with_attr(self):
        response = self.tester.get(['meta', 'test_file.h5'], params={'uri': '/group_with_children/group_with_attrs'})

        assert response.status_code == 200
        payload = response.json()
        assert payload['name'] == 'group_with_attrs'
        assert payload['type'] == 'group'
        assert type(payload['id']) == int and payload['id'] > 0
        assert payload['attributeCount'] == 2
        assert payload['childrenCount'] == 0

    def test_full_dataset(self):
        response = self.tester.get(['meta', 'test_file.h5'], params={'uri': '/group_with_children/dataset_1'})

        assert response.status_code == 200
        payload = response.json()
        assert payload['name'] == 'dataset_1'
        assert payload['type'] == 'dataset'
        assert type(payload['id']) == int and payload['id'] > 0
        assert payload['attributeCount'] == 0
        assert 'childrenCount' not in payload
        assert payload['labels'] == [{'start': 0, 'stop': 2, 'step': 1}, {'start': 0, 'stop': 3, 'step': 1}, {'start': 0, 'stop': 4, 'step': 1}]
        assert payload['dtype'] == '<f8'
        assert payload['ndim'] == 3
        assert payload['shape'] == [2, 3, 4]
        assert payload['size'] == 24

    def test_sliced_dataset(self):
        response = self.tester.get(['meta', 'test_file.h5'], params={'uri': '/group_with_children/dataset_1', 'ixstr': ":, 1:3, 2"})

        assert response.status_code == 200
        payload = response.json()
        assert payload['labels'] == [{'start': 0, 'stop': 2, 'step': 1}, {'start': 1, 'stop': 3, 'step': 1}]
        assert payload['dtype'] == '<f8'
        assert payload['ndim'] == 2
        assert payload['shape'] == [2, 2]
        assert payload['size'] == 4