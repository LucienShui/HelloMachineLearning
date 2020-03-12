import logging
import numpy


class BaseKNN:

    def __init__(self, dataset: numpy.ndarray, k: int):
        """
        init knn
        :param dataset: [[input_0, label_0], [input_1, label_1]]
        :param k: the top num of nn
        """
        self.__logger__ = logging.getLogger("KNN")
        self.__logger__.setLevel(logging.DEBUG)
        self.__dataset__ = dataset
        self.__k__ = k
        self.__logger__.debug("init finished")

    @classmethod
    def get_distance(cls, point_a: numpy.ndarray, point_b: numpy.ndarray):
        raise NotImplementedError

    def __get_k_nearest_neighbors__(self, point: numpy.ndarray, k):
        buf = []

        for each in self.__dataset__:
            dist = self.get_distance(each[0], point)
            label = each[1]

            buf.append([dist, label])

        self.__logger__.debug("buf = {}".format(buf))

        buf.sort()

        cnt = {}

        for each in buf[:k]:
            cnt[each[1]] = cnt.get(each[1], 0) + 1

        self.__logger__.debug("cnt = {}".format(cnt))

        result = sorted(cnt.items(), key=lambda x: x[1])

        return result[0][1]

    def predict(self, point: numpy.ndarray):
        return self.__get_k_nearest_neighbors__(point, self.__k__)
