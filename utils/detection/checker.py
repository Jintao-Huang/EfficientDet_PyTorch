# Author: Jintao Huang
# Time: 2020-6-9


class Checker:
    def __init__(self, train_tester, test_tester, saver, logger, check_epoch, ignore_num):
        """

        :param train_tester:
        :param test_tester:
        :param saver:
        :param logger:
        :param check_epoch: int. 每几个epoch check一次
        :param ignore_num: int. 前几次忽略的次数. 不check
        """
        assert train_tester or test_tester
        self.train_tester = train_tester
        self.test_tester = test_tester

        self.saver = saver
        self.logger = logger
        self.check_epoch = check_epoch
        self.ignore_num = ignore_num  # ignore check_epoch

    def step(self, epoch, last=False):
        if last or epoch % self.check_epoch == self.check_epoch - 1:
            if self.ignore_num > 0:
                self.ignore_num -= 1
                return
            if self.train_tester:
                print("----------------------------- Train Dataset")
                train_ap_dict = self.train_tester.test(last)
                train_mean_ap = sum(train_ap_dict.values()) / len(train_ap_dict)
                self.logger.log_mes({"train_mean_ap": train_mean_ap})

                if self.test_tester is None:
                    fname = "model_epoch%d_train%.4f.pth" % (epoch, train_mean_ap)
                    self.saver.save(fname)
            if self.test_tester:
                print("----------------------------- Test Dataset")
                test_ap_dict = self.test_tester.test(last)
                test_mean_ap = sum(test_ap_dict.values()) / len(test_ap_dict)
                self.logger.log_mes({"test_mean_ap": test_mean_ap})
                if self.train_tester:
                    fname = "model_epoch%d_train%.4f_test%.4f.pth" % (epoch, train_mean_ap, test_mean_ap)
                else:
                    fname = "model_epoch%d_test%.4f.pth" % (epoch, test_mean_ap)
                self.saver.save(fname)
            print("-----------------------------")
