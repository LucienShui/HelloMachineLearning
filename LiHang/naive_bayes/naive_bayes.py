import numpy
import logging


class NaiveBayes(object):

    def __init__(self, _lambda: float = 1.):
        self.__logger = logging.getLogger("NaiveBayes")
        self.__logger.setLevel(logging.DEBUG)

        self.__lambda = _lambda

        self.__sample_size: int = 0
        self.__unique_label: numpy.ndarray = numpy.array([])
        self.__label_counter: dict = {}
        self.__input_data: numpy.ndarray = numpy.array([])
        self.__input_label: numpy.ndarray = numpy.array([])

    def __calc_x_y(self, feature_dim_idx: int, feature_value: any, label: any, input_data: numpy.ndarray,
                   input_label: numpy.ndarray):
        """
        calculate P(X|Y) with Laplacian smoothing
        :param feature_dim_idx:
        :param feature_value:
        :param label:
        :param input_data:
        :param input_label:
        :return:
        """

        cnt = 0

        for i in range(input_data.shape[0]):
            if input_data[i][feature_dim_idx] == feature_value:
                if input_label[i] == label:
                    cnt += 1

        unique_feature_value = numpy.unique(input_data[:, feature_dim_idx])

        return (cnt + self.__lambda) / \
               (self.__label_counter[label] + unique_feature_value.shape[0] * self.__lambda)

    def __calc_y(self, label: any):
        """
        calculate P(Y) with Laplacian smoothing
        :param label:
        :return:
        """
        upper = (self.__label_counter.get(label, 0) + self.__lambda)
        lower = (self.__sample_size + self.__unique_label.shape[0] * self.__lambda)
        result = upper / lower
        # self.__logger.debug("{} / {} = {}".format(upper, lower, result))
        return result

    def fit(self, input_data: numpy.ndarray, input_label: numpy.ndarray):
        self.__label_counter = {}
        self.__sample_size = input_label.shape[0]
        self.__unique_label = numpy.unique(input_label)
        self.__input_data = input_data
        self.__input_label = input_label

        for label in input_label:
            self.__label_counter[label] = self.__label_counter.get(label, 0) + 1

    def __predict_one(self, data: numpy.ndarray):

        max_value = -0.1
        result = self.__unique_label[0]

        for label in self.__unique_label:

            buf = self.__calc_y(label)

            self.__logger.debug("P(Y = {}) = {}".format(label, buf))

            for i in range(data.shape[0]):
                tmp = self.__calc_x_y(i, data[i], label, self.__input_data, self.__input_label)
                self.__logger.debug("P(X_{} = {} | Y = {}) = {}".format(i, data[i], label, tmp))

                buf *= tmp

            self.__logger.debug("P(X = {} | Y = {}) = {}".format(data, label, buf))

            if buf > max_value:
                result = label
                max_value = buf

        return result

    def predict(self, test_data: numpy.ndarray) -> numpy.ndarray:
        result_array = []

        for each in test_data:
            result_array.append(self.__predict_one(each))

        return numpy.array(result_array)
