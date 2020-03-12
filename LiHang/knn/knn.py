import numpy

from knn import BaseKNN


class KNN(BaseKNN):

    @classmethod
    def get_distance(cls, point_a: numpy.ndarray, point_b: numpy.ndarray):
        return numpy.sqrt(numpy.sum(numpy.square(point_a - point_b)))
