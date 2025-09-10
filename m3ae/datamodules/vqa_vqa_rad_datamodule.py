from collections import defaultdict

from .base_datamodule import BaseDataModule
from ..datasets import VQAVQARADDataset


class VQAVQARADDataModule(BaseDataModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def dataset_cls(self):
        return VQAVQARADDataset

    @property
    def dataset_name(self):
        return "vqa_vqa_rad"

    def setup(self, stage):
        super().setup(stage)

        train_answers = self.train_dataset.table["answers"].to_pandas().tolist()
        val_answers = self.val_dataset.table["answers"].to_pandas().tolist()
        train_labels = self.train_dataset.table["answer_labels"].to_pandas().tolist()
        val_labels = self.val_dataset.table["answer_labels"].to_pandas().tolist()

        # 直接使用列表中的数组
        # train_answer_types = self.train_dataset.answer_type
        # val_answer_types = self.val_dataset.answer_type
        #
        # open_answers_set = set()  # 用于存储开放问题的答案
        # close_answers_set = set()  # 用于存储闭合问题的答案
        #
        # # 遍历收集答案
        # for answers, answer_types in zip(train_answers + val_answers, train_answer_types + val_answer_types):
        #     for answer, answer_type in zip(answers, answer_types):
        #         if answer_type == 1:  # 假设 'open' 表示开放问题
        #             if answer is not None:
        #                 open_answers_set.add(str(answer))  # 465
        #         elif answer_type == 0:  # 假设 'close' 表示闭合问题
        #             if answer is not None:
        #                 close_answers_set.add(str(answer))  # 63

        all_answers = [c for c in train_answers + val_answers if c is not None]
        all_answers = [l for lll in all_answers for ll in lll for l in ll]
        all_labels = [c for c in train_labels + val_labels if c is not None]
        all_labels = [l for lll in all_labels for ll in lll for l in ll]

        self.answer2id = {k: v for k, v in zip(all_answers, all_labels)}
        sorted_a2i = sorted(self.answer2id.items(), key=lambda x: x[1])
        self.num_class = max(self.answer2id.values()) + 1

        self.id2answer = defaultdict(lambda: "unknown")
        for k, v in sorted_a2i:
            self.id2answer[v] = k
