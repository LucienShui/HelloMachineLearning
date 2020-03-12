import numpy
from perceptron import Perceptron


class DualPerceptron(Perceptron):
    """
    感知机的对偶形式
    """

    def __init__(self):
        super().__init__()
        self.gram = numpy.array([])
        self.dual_weight = numpy.array([])

    def __apply__(self, idx: int) -> float:

        result = 0.

        for i in range(self.gram.shape[0]):
            result += self.dual_weight[i] * self.gram[i][idx]

        return result + self.bias

    def fit(self, samples: numpy.ndarray, labels: numpy.ndarray, learning_rate: float = 0.04):
        length = len(samples)
        self.gram = numpy.zeros([length, length])
        self.dual_weight = numpy.zeros([length])

        for i in range(length):

            for j in range(length):

                if i == j:
                    self.gram[i][i] = labels[i] * samples[i].dot(samples[i])
                else:
                    self.gram[i][j] = labels[i] * samples[i].dot(samples[j])

        flag = True
        epoch_cnt = 0
        update_cnt = 0

        while flag:

            flag = False
            epoch_cnt += 1

            for i in range(length):
                if labels[i] * self.__apply__(i) <= 0:

                    flag = True
                    update_cnt += 1

                    self.dual_weight[i] += learning_rate
                    self.bias += learning_rate * labels[i]

        self.__logger__.debug("epoch cnt = {}, update cnt = {}".format(epoch_cnt, update_cnt))

        for i in range(length):
            self.weight += self.dual_weight[i] * labels[i] * samples[i]

    def predict(self, data: numpy.ndarray):
        return 1 if super().__apply__(data) >= 0 else -1
