import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3"

import copy
import os
import resource

import pytorch_lightning as pl
# from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from m3ae.config import ex
from m3ae.datamodules.multitask_datamodule import MTDataModule
from m3ae.modules import M3AETransformerSS

# from m3ae.datasets.ROCO import ROCOFeatureDataset
# from torch.utils.data import DataLoader

rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))


@ex.automain
def main(_config):
    _config = copy.deepcopy(_config)  # 创建_config的深拷贝，以避免在后续操作中对原数据的修改
    pl.seed_everything(_config["seed"])

    # Data modules
    dm = MTDataModule(_config, dist=True)

    # retrieval
    # 创建检索数据集
    # retrieval_function = None
    # if "retrieval" in _config and _config["retrieval"]:
    #     # if "retrieval_dataset" in _config:
    #     retrieval_dataset = ROCOFeatureDataset("train", _config['retrieval_data'], device=f"cuda:{_config['gpu_ids']}")
    #     # retrieval_dataset = ROCOFeatureDataset("train", _config['retrieval_data'], device="cpu")
    #     # retrieval_dataset = load_dataset(_config["datafolder"], _config["retrieval_dataset"], "train", device)
    #     retrieval_loader = DataLoader(retrieval_dataset, _config["retrieval_batch"], shuffle=True, num_workers=2)
    #
    #     if "k" in _config:
    #         k = _config["k"]
    #     else:
    #         k = 15
    #
    #     print(f"Using {k}-nn retrieval from {retrieval_dataset.dataroot} with only training data ...")
    #     retrieval_dataset.create_retrieval_dataset(retrieval_loader, is_training_phase=True, retrieval_k=k)
    #     retrieval_function = retrieval_dataset.retrieve_closest_qa_pairs

    # Module
    model = M3AETransformerSS(_config)  # 网络模块初始化
    # model.set_retrieval_function(retrieval_function)
    # print(model)

    # Loggers
    os.makedirs(_config["log_dir"], exist_ok=True)
    exp_name = f'{_config["exp_name"]}'
    run_name = f'{exp_name}-seed{_config["seed"]}-from_{_config["load_path"].replace("/", "_")}'
    tb_logger = pl.loggers.TensorBoardLogger(_config["log_dir"], name=run_name)
    # wb_logger = pl.loggers.WandbLogger(project="MICCAI-M3AE", name=run_name)
    loggers = [tb_logger]

    # Callback
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        save_top_k=1,
        verbose=True,
        monitor="val/the_metric",
        mode="max",
        save_last=True,
        save_weights_only=True if "finetune" in exp_name else False
    )  # 创建回调函数，保存最佳模型
    lr_callback = pl.callbacks.LearningRateMonitor(logging_interval="step")
    callbacks = [checkpoint_callback, lr_callback]

    # Training Hyper-Parameters
    num_gpus = (_config["num_gpus"] if isinstance(_config["num_gpus"], int) else len(_config["num_gpus"]))
    # print(_config)
    gpu_ids = (_config['gpu_ids'])
    grad_steps = max(_config["batch_size"] // (_config["per_gpu_batchsize"] * num_gpus * _config["num_nodes"]), 1)
    max_steps = _config["max_steps"] if _config["max_steps"] is not None else None
    max_epochs = _config["max_epoch"] if max_steps is None else 1000

    # Trainer
    trainer = pl.Trainer(
        gpus=[gpu_ids],
        # gpus=num_gpus,
        num_nodes=_config["num_nodes"],
        precision=_config["precision"],
        accelerator="ddp",
        benchmark=True,
        deterministic=True,
        max_epochs=max_epochs,
        max_steps=max_steps,
        callbacks=callbacks,
        logger=loggers,
        prepare_data_per_node=False,
        replace_sampler_ddp=False,
        accumulate_grad_batches=grad_steps,
        log_every_n_steps=10,
        flush_logs_every_n_steps=10,
        resume_from_checkpoint=_config["resume_from"],
        weights_summary="top",
        fast_dev_run=_config["fast_dev_run"],
        val_check_interval=_config["val_check_interval"],
        default_root_dir=_config["default_root_dir"]
    )  # 初始化训练器

    if not _config["test_only"]:
        trainer.fit(model, datamodule=dm)
        if "finetune" in exp_name:
            trainer.test(ckpt_path="best", datamodule=dm)
    else:
        trainer.test(model, datamodule=dm)
        # get_cam(model)
