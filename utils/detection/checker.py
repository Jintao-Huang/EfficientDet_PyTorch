# Author: Jintao Huang
# Time: 2020-6-9


class Checker:
    def __init__(self, train_tester, test_tester, saver, check_epoch):
        assert train_tester or test_tester
        self.train_tester = train_tester
        self.test_tester = test_tester

        self.saver = saver
        self.check_epoch = check_epoch

    def step(self, epoch, last=False):
        if last or epoch % self.check_epoch == self.check_epoch - 1:
            if self.train_tester:
                print("----------------------------- Train Dataset")
                train_ap_dict = self.train_tester.test(last)
                train_map = sum(train_ap_dict.values()) / len(train_ap_dict)
                if self.test_tester is None:
                    fname = "model_epoch%d_train%.4f.pth" % (epoch, train_map)
                    self.saver.save(fname)
            if self.test_tester:
                print("----------------------------- Test Dataset")
                test_ap_dict = self.test_tester.test(last)
                test_map = sum(test_ap_dict.values()) / len(test_ap_dict)
                if self.train_tester:
                    fname = "model_epoch%d_train%.4f_test%.4f.pth" % (epoch, train_map, test_map)
                else:
                    fname = "model_epoch%d_test%.4f.pth" % (epoch, test_map)
                self.saver.save(fname)
            print("-----------------------------")
