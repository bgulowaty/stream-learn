from strlearn.learners.listeners.BaseLearningResultListener import BaseLearningResultListener


class PrintingListener(BaseLearningResultListener):

    def listen(self, result):
        print(result)
