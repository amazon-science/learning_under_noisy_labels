#!/usr/bin/python3

import unittest

import numpy as np

from src.iaa_api import InterAnnotatorAgreementAPI


class TestAII(unittest.TestCase):
    def setUp(self):
        self.human_annotations = np.array([[0, 1, 2, 3, 4], [0, 2, 2, 0, 4], [0, 2, 1, 0, 4]]).T
        self.api = InterAnnotatorAgreementAPI(self.human_annotations)

    def test_check_d_matrix_shape(self):
        self.assertEqual(self.api._d_matrix.shape, (self.api.num_classes, self.api.num_classes))

    def test_check_m_matrix_shape(self):
        api = InterAnnotatorAgreementAPI(self.human_annotations)
        self.assertEqual(self.api._m_matrix.shape, (self.api.num_classes, api.num_classes))

    def test_label_distribution_sum_one(self):
        api = InterAnnotatorAgreementAPI(self.human_annotations)
        self.assertEqual(sum(api.label_distribution), 1)

    def test_posterior_probability_sum_up_to_1(self):
        pc = self.api.get_posterior_probability()
        self.assertEqual(round(pc.sum(axis=1).sum().item(), 4), pc.shape[0])

    def test_posterior_probability_shape(self):
        pc = self.api.get_posterior_probability()
        self.assertEqual(pc.shape, (self.api.num_samples, self.api.num_classes))

    def test_average_soft_labels_sum_up_to_1(self):
        avg_soft_labels = self.api.get_average_soft_labels()
        self.assertEqual(avg_soft_labels.sum(axis=1).sum(), avg_soft_labels.shape[0])

    def test_mixed_method(self):
        self.assertEqual(self.api.get_mixed().sum(), self.api.annotations.shape[0])


if __name__ == "__main__":
    unittest.main()
