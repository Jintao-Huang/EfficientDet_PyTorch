# Author: Jintao Huang
# Time: 2020-6-6
from torchvision.ops.boxes import box_iou
import torch


class APCounter:
    def __init__(self, labels_map, score_thresh=0.5, iou_thresh=0.5):
        """

        :param labels_map: Dict[int: str]
        :param score_thresh:
        :param iou_thresh:
        """
        self.score_thresh = score_thresh
        self.iou_thresh = iou_thresh
        self.label_list = [labels_map[i] for i in range(len(labels_map))]
        # List[List[tuple(score(float), correct(bool))]]. 每个类一个table
        self.pred_table_list = None
        # List[num(int)]. 每个类分别计算
        self.target_num_list = None

    def init_table(self):
        self.pred_table_list = [[] for _ in range(len(self.label_list))]
        # List[num(int)]. 每个类分别计算
        self.target_num_list = [0 for _ in range(len(self.label_list))]

    def add(self, pred_list, target_list):
        """

        :param pred_list: List[Dict]. "scores" 已按从大到小排序
        :param target_list: List[Dict]
        :return: None
        """
        for pred, target in zip(pred_list, target_list):
            pred_boxes, pred_labels, pred_scores = pred['boxes'], pred['labels'], pred['scores']
            target_boxes, target_labels = target['boxes'], target['labels']
            # 1. target_num_list
            for target_label in target_labels:
                target_label = target_label.item()
                self.target_num_list[target_label] += 1
            # 2. pred_table_list
            have_detected = torch.zeros(target_labels.shape[0], dtype=torch.bool)  # 记录已经被检测过的
            for pred_box, pred_label, pred_score in zip(pred_boxes, pred_labels, pred_scores):
                pred_label = pred_label.item()
                pred_score = pred_score.item()
                if pred_score < self.score_thresh:
                    continue
                # 选择同类型的target_boxes
                matched = torch.nonzero(target_labels == pred_label, as_tuple=False)  # (N)
                correct = self._is_box_correct(pred_box, target_boxes, matched, have_detected, self.iou_thresh)
                self.pred_table_list[pred_label].append((pred_score, correct))

    def get_ap_dict(self):
        ap_list = [0. for _ in range(len(self.label_list))]
        for i, (pred_table, target_num) in enumerate(zip(self.pred_table_list, self.target_num_list)):
            recall_list, prec_list = self._calc_pr(pred_table, target_num)
            ap_list[i] = self._calc_ap(recall_list, prec_list)
        ap_dict = {label: ap for label, ap in zip(self.label_list, ap_list)}
        return ap_dict

    @staticmethod
    def print_ap(ap_dict):
        print("AP: ")
        for label, ap in ap_dict.items():
            print("  %s: %f" % (label, ap))
        print("", end="", flush=True)

    @staticmethod
    def _is_box_correct(pred_box, target_boxes, matched, have_detected, iou_thresh=0.5):
        """

        :param pred_box: Tensor[4]
        :param target_boxes: Tensor[N, 4]. all
        :param matched: Tensor[NUM]
        :param have_detected: Tensor[N]. bool
        :param iou_thresh: int
        :return: bool
        """
        t_boxes = target_boxes[matched]  # (NUM, 4)
        if t_boxes.shape[0] == 0:
            return False
        iou_max, idx = torch.max(box_iou(pred_box[None], t_boxes)[0], dim=0)  # (N) -> ()
        if iou_max < iou_thresh:
            return False
        elif have_detected[matched[idx]]:
            return False
        else:
            have_detected[matched[idx]] = True
            return True

    @staticmethod
    def _calc_pr(pred_table, target_num):
        """calculate precision and recall

        :param pred_table: List[tuple(score(float), correct(bool))]. const
        :param target_num: int. const
        :return: recall_list: List[NUM], prec_list: List[NUM]
        """
        pred_table = sorted(pred_table, key=lambda x: -x[0])
        recall_list, prec_list = [], []
        correct_num = 0
        for i, (_, correct) in enumerate(pred_table):
            pred_num = i + 1  # 预测的次数
            if correct:
                correct_num += 1  # 正确的次数
            recall_list.append(correct_num / target_num)
            prec_list.append(correct_num / pred_num)

        return recall_list, prec_list

    @staticmethod
    def _calc_ap(recall_list, prec_list):
        """recall_list(单调递增), prec_list. (recall, prec)为一个点"""

        # 1. 预处理
        recall_list.insert(0, 0.)
        recall_list.append(1.)
        prec_list.insert(0, 0.)
        prec_list.append(0.)
        for i in reversed(range(len(recall_list) - 1)):
            prec_list[i] = max(prec_list[i], prec_list[i + 1])
        # 2. 每个recall值取一个点(prec最高的点)
        idx_list = [0]
        for i in range(0, len(recall_list) - 1):
            if recall_list[i + 1] != recall_list[i]:
                idx_list.append(i + 1)
        # 3. 计算
        ap = 0.
        for i in range(len(idx_list) - 1):
            start = recall_list[idx_list[i]]
            end = recall_list[idx_list[i + 1]]
            value = prec_list[idx_list[i + 1]]
            ap += (end - start) * value
        return ap
