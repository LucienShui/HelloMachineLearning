import unittest
import numpy
import logging
from perceptron import Perceptron, DualPerceptron

logging.basicConfig()


class MyTestCase(unittest.TestCase):

    def __test_perceptron(self, perceptron: Perceptron):
        samples = numpy.array([[3, 3], [4, 3], [1, 1]])
        labels = numpy.array([1, 1, -1])

        perceptron.fit(samples, labels)

        for i in range(samples.shape[0]):
            self.assertEqual(perceptron.predict(samples[i]), labels[i])
            logging.debug("i = {} success".format(i))

    def test_perceptron(self):
        self.__test_perceptron(Perceptron())
        self.__test_perceptron(DualPerceptron())


if __name__ == '__main__':
    unittest.main()
