import unittest
import pandas
import logging
from sklearn.tree import DecisionTreeClassifier
from sklearn import preprocessing

logging.basicConfig()
logger = logging.getLogger("TestTree")
logger.setLevel(logging.DEBUG)


class TreeTestCase(unittest.TestCase):
    def test_decision_tree_classifier(self):
        """
        https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
        """

        df = pandas.DataFrame(columns=["年龄", "有工作", "有自己的房子", "信贷情况", "类别"],
                              data=[["青年", "否", "否", "一般", "否"],
                                    ["青年", "否", "否", "好", "否"],
                                    ["青年", "是", "否", "好", "是"],
                                    ["青年", "是", "是", "一般", "是"],
                                    ["青年", "否", "否", "一般", "否"],
                                    ["中年", "否", "否", "一般", "否"],
                                    ["中年", "否", "否", "好", "否"],
                                    ["中年", "是", "是", "好", "是"],
                                    ["中年", "否", "是", "非常好", "是"],
                                    ["中年", "否", "是", "非常好", "是"],
                                    ["老年", "否", "是", "非常好", "是"],
                                    ["老年", "否", "是", "好", "是"],
                                    ["老年", "是", "否", "好", "是"],
                                    ["老年", "是", "否", "非常好", "是"],
                                    ["老年", "否", "否", "一般般", "否"]])

        def get_encoder(data_frame: pandas.DataFrame) -> dict:
            result = {}

            for each in data_frame.columns:
                result[each] = preprocessing.LabelEncoder()
                result[each].fit(data_frame[each])

            return result

        def label_encode(data_frame: pandas.DataFrame, label_encoder_dict: dict) -> pandas.DataFrame:
            result = data_frame.copy()

            for each in result.columns:
                result[each] = label_encoder_dict[each].transform(result[each])

            return result

        label_encoder: dict = get_encoder(df)

        train_input = label_encode(df[["年龄", "有工作", "有自己的房子", "信贷情况"]], label_encoder)
        train_label = label_encode(df[["类别"]], label_encoder)

        test_input = pandas.DataFrame(columns=["年龄", "有工作", "有自己的房子", "信贷情况"],
                                      data=[["青年", "否", "是", "一般"],
                                            ["中年", "是", "否", "好"],
                                            ["老年", "否", "是", "一般"]])

        dtc = DecisionTreeClassifier()
        dtc.fit(train_input, train_label)

        prediction = dtc.predict(label_encode(test_input, label_encoder))

        test_result = test_input.copy()
        test_result["类别"] = label_encoder["类别"].inverse_transform(prediction)

        logger.info("\n{}".format(test_result))

        self.assertEqual(label_encoder["类别"].inverse_transform(prediction).tolist(), ["是", "是", "是"])


if __name__ == '__main__':
    unittest.main()
