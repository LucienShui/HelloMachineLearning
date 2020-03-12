import numpy
import logging


class Perceptron:

    def __init__(self, dim: int = 2):
        self.__logger__ = logging.getLogger("Perceptron")
        self.__logger__.setLevel(logging.DEBUG)
        self.dim = dim
        self.weight = numpy.zeros([dim])
        self.bias = 0.
        self.__logger__.debug("self.dim = {}, self.weight = {}, self.bias = {}".format(
            self.dim, self.weight, self.bias))

    def __apply__(self, data: numpy.ndarray) -> float:
        """
        将点代入超平面的公式
        :param data: 点
        :return: 代入运算的结果
        """
        return self.weight.dot(data) + self.bias

    def fit(self, samples: numpy.ndarray, labels: numpy.ndarray, learning_rate: float = 0.04):
        flag = True
        epoch_cnt = 0  # 循环次数
        update_cnt = 0  # 误分类的次数

        while flag:

            flag = False
            epoch_cnt += 1

            for i in range(samples.shape[0]):
                if labels[i] * self.__apply__(samples[i]) <= 0.:  # 有误分类的点

                    self.__logger__.debug("wrong at samples[{}] = {}".format(i, samples[i]))

                    update_cnt += 1
                    flag = True

                    self.weight += learning_rate * labels[i] * samples[i]
                    self.bias += learning_rate * labels[i]

                    self.__logger__.debug("weight = {}, bias = {}".format(self.weight, self.bias))

        self.__logger__.debug("epoch cnt = {}, update cnt = {}".format(epoch_cnt, update_cnt))

    def predict(self, data: numpy.ndarray):
        result = self.__apply__(data)
        self.__logger__.debug("result = {}".format(result))
        return 1 if result >= 0 else -1
