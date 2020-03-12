import unittest
import numpy
import logging
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
from naive_bayes import NaiveBayes

logging.basicConfig()

# S, M, L = "S", "M", "L"
S, M, L = 1, 2, 3

train_input = numpy.array([
    [1, S], [1, M], [1, M], [1, S], [1, S],
    [2, S], [2, M], [2, M], [2, L], [2, L],
    [3, L], [3, M], [3, M], [3, L], [3, L]
])

train_label = numpy.array([
    -1, -1, 1, 1, -1,
    -1, -1, 1, 1, 1,
    1, 1, 1, 1, -1
])

test_input = numpy.array([[2, S]])


class NaiveBayesTestCase(unittest.TestCase):

    def __test_nb__(self, nb):
        nb.fit(train_input, train_label)
        self.assertEqual(nb.predict(test_input), numpy.array([-1]))

    def test_gnb(self):
        self.__test_nb__(GaussianNB())

    def test_bnb(self):
        self.__test_nb__(BernoulliNB())

    def test_mnb(self):
        self.__test_nb__(MultinomialNB())

    def test_custom(self):
        self.__test_nb__(NaiveBayes())


if __name__ == '__main__':
    unittest.main()
