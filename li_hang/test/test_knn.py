import unittest
import logging
import numpy
from knn import KNN


class MyTestCase(unittest.TestCase):
    def test_something(self):
        logging.basicConfig()

        dataset = numpy.array([
            [[5, 4], 1],
            [[9, 6], 1],
            [[4, 7], 1],
            [[2, 3], -1],
            [[8, 1], -1],
            [[7, 2], -1]
        ])

        knn = KNN(dataset, 1)

        test_point = numpy.array([5, 3])

        self.assertEqual(knn.predict(test_point), 1)


if __name__ == '__main__':
    unittest.main()
