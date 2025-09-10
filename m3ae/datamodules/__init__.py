from .cls_melinda_datamodule import CLSMELINDADataModule
from .irtr_roco_datamodule import IRTRROCODataModule
from .pretraining_medicat_datamodule import MedicatDataModule
from .pretraining_roco_datamodule import ROCODataModule
from .vqa_medvqa_2019_datamodule import VQAMEDVQA2019DataModule
from .vqa_slack_datamodule import VQASLACKDataModule
from .vqa_vqa_rad_datamodule import VQAVQARADDataModule
from .vqa_ovqa_datamodule import VQAOVQADataModule

_datamodules = {
    "medicat": MedicatDataModule,
    "roco": ROCODataModule,
    "vqa_vqa_rad": VQAVQARADDataModule,
    "vqa_ovqa": VQAOVQADataModule,
    "vqa_slack": VQASLACKDataModule,
    "vqa_medvqa_2019": VQAMEDVQA2019DataModule,
    "cls_melinda": CLSMELINDADataModule,
    "irtr_roco": IRTRROCODataModule
}  # 不同数据集的DataModule映射表
