import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import functools

import torch
import torch.nn.functional as F
import tqdm
from einops import rearrange
from torch.utils.data.distributed import DistributedSampler

from .dist_utils import all_gather

import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F


def compute_mlm(pl_module, batch):
    infer = pl_module.infer(batch, mask_text=True, mask_image=False)
    mlm_logits = pl_module.mlm_head(infer["multi_modal_text_feats"])
    mlm_labels = infer["text_labels"]

    mlm_loss = F.cross_entropy(
        mlm_logits.view(-1, pl_module.hparams.config["vocab_size"]),
        mlm_labels.view(-1),
        ignore_index=-100,
    )

    ret = {
        "mlm_loss": mlm_loss,
        "mlm_logits": mlm_logits,
        "mlm_labels": mlm_labels,
        "mlm_ids": infer["text_ids"],
    }

    phase = "train" if pl_module.training else "val"
    loss = getattr(pl_module, f"{phase}_mlm_loss")(ret["mlm_loss"])
    acc = getattr(pl_module, f"{phase}_mlm_accuracy")(ret["mlm_logits"], ret["mlm_labels"])
    pl_module.log(f"mlm/{phase}/loss", loss)
    pl_module.log(f"mlm/{phase}/accuracy", acc)

    return ret


def compute_mim(pl_module, batch):
    infer = pl_module.infer(batch, mask_text=False, mask_image=True)

    if pl_module.hparams.config["mim_layer"] == -1:
        multi_modal_image_feats = infer["multi_modal_image_feats"]
    else:
        layer_idx = pl_module.hparams.config["mim_layer"]
        multi_modal_image_feats = infer[f"multi_modal_image_feats_{layer_idx}"]

    mim_logits = pl_module.mim_head(multi_modal_image_feats, infer["mim_ids_restore"])

    target = infer["patched_images"]
    if pl_module.hparams.config["norm_pix_loss"]:
        mean = target.mean(dim=-1, keepdim=True)
        var = target.var(dim=-1, keepdim=True)
        target = (target - mean) / (var + 1.e-6) ** .5
    mim_labels = target
    mask = infer["mim_masks"]

    mim_loss = (mim_logits - mim_labels) ** 2
    mim_loss = mim_loss.mean(dim=-1)  # [N, L], mean loss per patch
    mim_loss = (mim_loss * mask).sum() / mask.sum()  # mean loss on removed patches

    ret = {
        "mim_loss": mim_loss,
        "mim_logits": mim_logits,
        "mim_labels": mim_labels,
    }

    phase = "train" if pl_module.training else "val"
    loss = getattr(pl_module, f"{phase}_mim_loss")(ret["mim_loss"])
    acc = -loss
    pl_module.log(f"mim/{phase}/loss", loss)
    pl_module.log(f"mim/{phase}/accuracy", acc)

    return ret


def compute_itm(pl_module, batch):
    pos_len = len(batch["text"]) // 2
    neg_len = len(batch["text"]) - pos_len
    itm_labels = torch.cat([torch.ones(pos_len), torch.zeros(neg_len)]).to(pl_module.device)
    itm_labels = itm_labels[torch.randperm(itm_labels.size(0))]

    itm_images = [
        torch.stack(
            [
                ti if itm_labels[i] == 1 else fi
                for i, (ti, fi) in enumerate(zip(bti, bfi))
            ]
        )
        for bti, bfi in zip(batch["image"], batch["false_image_0"])
    ]

    batch = {k: v for k, v in batch.items()}
    batch["image"] = itm_images

    infer = pl_module.infer(batch, mask_text=False, mask_image=False)

    itm_logits = pl_module.itm_head(infer["multi_modal_cls_feats"])
    itm_loss = F.cross_entropy(itm_logits, itm_labels.long())

    ret = {
        "itm_loss": itm_loss,
        "itm_logits": itm_logits,
        "itm_labels": itm_labels,
    }

    phase = "train" if pl_module.training else "val"
    loss = getattr(pl_module, f"{phase}_itm_loss")(ret["itm_loss"])
    acc = getattr(pl_module, f"{phase}_itm_accuracy")(ret["itm_logits"], ret["itm_labels"])
    pl_module.log(f"itm/{phase}/loss", loss)
    pl_module.log(f"itm/{phase}/accuracy", acc)

    return ret


def compute_vqa(pl_module, batch, test=False):
    infer = pl_module.infer(batch, mask_text=False, mask_image=False)  # 使用infer方法通过batch数据进行推理，得到多模态特征推理结果
    vqa_logits = pl_module.vqa_head(infer["multi_modal_cls_feats"])  # 将推理结果中的多模态分类特征传递到vqahead，计算出vqa的logits（未归一化的预测得分）
    vqa_targets = torch.zeros(len(vqa_logits), pl_module.hparams.config["vqa_label_size"]).to(
        pl_module.device)  # 初始化vqa_targets，用于存储目标标签(B，lable_size)

    # 从batch中提取vqa_labels和vqa_scores，答案类型转换为张量
    vqa_labels = batch["vqa_labels"]
    vqa_scores = batch["vqa_scores"]
    vqa_answer_types = torch.tensor(batch["answer_types"]).to(pl_module.device)

    # if test:
    #     visualize_attention(batch["image"][0],infer["attentions"]["text2image_attns"],infer["attentions"]["image2text_attns"])
    # 遍历每一个样本的标签和对应分数，构建vqa_targets
    for i, (_label, _score) in enumerate(zip(vqa_labels, vqa_scores)):
        for l, s in zip(_label, _score):
            vqa_targets[i, l] = s

    # 使用二元交叉熵损失函数计算vqa_loss，并×标签数量调整损失
    vqahead_loss = (F.binary_cross_entropy_with_logits(vqa_logits, vqa_targets) * vqa_targets.shape[1])
    alpha = 0.2
    beta = 0.3
    vqa_loss = vqahead_loss + alpha * infer["itc_loss"] + beta * infer["ag_loss"]

    # 返回结果字典
    ret = {
        "vqa_loss": vqa_loss,
        "itc_loss": infer["itc_loss"],
        "vqahead_loss": vqahead_loss,
        "vqa_logits": vqa_logits,
        "vqa_targets": vqa_targets,
        "vqa_labels": vqa_labels,
        "vqa_scores": vqa_scores,
        "vqa_answer_types": vqa_answer_types,
    }

    if test:
        phase = "test"
    else:
        phase = "train" if pl_module.training else "val"

    loss = getattr(pl_module, f"{phase}_vqa_loss")(ret["vqa_loss"])
    score = getattr(pl_module, f"{phase}_vqa_score")(ret["vqa_logits"], ret["vqa_targets"], ret["vqa_answer_types"])
    pl_module.log(f"vqa/{phase}/loss", loss)
    # pl_module.log(f"vqa/{phase}/vqa_loss", ret["vqahead_loss"])
    # pl_module.log(f"vqa/{phase}/itc_loss", ret["itc_loss"])
    pl_module.log(f"vqa/{phase}/score", score)

    return ret


def compute_cls(pl_module, batch, test=False):
    infer = pl_module.infer(batch, mask_text=False, mask_image=False)

    cls_logits = pl_module.cls_head(infer["multi_modal_cls_feats"])
    cls_labels = batch["cls_labels"]
    cls_loss = F.cross_entropy(cls_logits, cls_labels)

    ret = {
        "cls_loss": cls_loss,
        "cls_logits": cls_logits,
        "cls_labels": cls_labels,
    }

    if test:
        phase = "test"
    else:
        phase = "train" if pl_module.training else "val"

    loss = getattr(pl_module, f"{phase}_cls_loss")(ret["cls_loss"])
    acc = getattr(pl_module, f"{phase}_cls_accuracy")(ret["cls_logits"], ret["cls_labels"])
    pl_module.log(f"cls/{phase}/loss", loss)
    pl_module.log(f"cls/{phase}/accuracy", acc)

    return ret


def compute_irtr(pl_module, batch, test=False):
    is_training_phase = pl_module.training
    _bs, _c, _h, _w = batch["image"][0].shape
    false_len = pl_module.hparams.config["draw_false_text"]
    text_ids = torch.stack([batch[f"false_text_{i}_ids"] for i in range(false_len)], dim=1)
    text_masks = torch.stack([batch[f"false_text_{i}_masks"] for i in range(false_len)], dim=1)
    text_labels = torch.stack([batch[f"false_text_{i}_labels"] for i in range(false_len)], dim=1)

    text_ids = torch.cat([batch["text_ids"].unsqueeze(1), text_ids], dim=1)
    text_masks = torch.cat([batch["text_masks"].unsqueeze(1), text_masks], dim=1)
    text_labels = torch.cat([batch["text_labels"].unsqueeze(1), text_labels], dim=1)
    images = batch["image"][0].unsqueeze(1).expand(_bs, false_len + 1, _c, _h, _w)

    batch_infer = {
        "image": [rearrange(images, "bs fs c h w -> (bs fs) c h w")],
        "text_ids": rearrange(text_ids, "bs fs tl -> (bs fs) tl"),
        "text_masks": rearrange(text_masks, "bs fs tl -> (bs fs) tl"),
        "text_labels": rearrange(text_labels, "bs fs tl -> (bs fs) tl"),
    }

    infer = pl_module.infer(batch_infer)

    score = pl_module.irtr_head(infer["multi_modal_cls_feats"])[:, 0]
    score = rearrange(score, "(bs fs) -> bs fs", bs=_bs, fs=false_len + 1)
    answer = torch.zeros(_bs).to(score).long()
    irtr_loss = F.cross_entropy(score, answer)

    ret = {"irtr_loss": irtr_loss}

    if test:
        phase = "test"
    else:
        phase = "train" if pl_module.training else "val"

    irtr_loss = getattr(pl_module, f"{phase}_irtr_loss")(ret["irtr_loss"])
    pl_module.log(f"irtr/{phase}/irtr_loss", irtr_loss)

    return ret


@torch.no_grad()
def compute_irtr_recall(pl_module):
    text_dset = pl_module.trainer.datamodule.dms[0].make_no_false_val_dset()
    text_dset.tokenizer = pl_module.trainer.datamodule.dms[0].tokenizer
    text_loader = torch.utils.data.DataLoader(
        text_dset,
        batch_size=256,
        num_workers=pl_module.hparams.config["num_workers"],
        pin_memory=True,
        collate_fn=functools.partial(text_dset.collate,
                                     mlm_collator=pl_module.trainer.datamodule.dms[0].mlm_collator, ), )

    image_dset = pl_module.trainer.datamodule.dms[0].make_no_false_val_dset(image_only=True)
    image_dset.tokenizer = pl_module.trainer.datamodule.dms[0].tokenizer
    dist_sampler = DistributedSampler(image_dset, shuffle=False)
    image_loader = torch.utils.data.DataLoader(
        image_dset,
        batch_size=1,
        num_workers=pl_module.hparams.config["num_workers"],
        sampler=dist_sampler,
        pin_memory=True,
        collate_fn=functools.partial(image_dset.collate,
                                     mlm_collator=pl_module.trainer.datamodule.dms[0].mlm_collator, ), )

    # TODO: speed up the process by caching text/image features
    text_preload = list()
    for _b in tqdm.tqdm(text_loader, desc="text prefetch loop"):
        # == Begin: Add New Keys ==
        batch_text_preload = {
            "text_ids": _b["text_ids"].to(pl_module.device),
            "text_masks": _b["text_masks"].to(pl_module.device),
            "text_labels": _b["text_labels"].to(pl_module.device),
            "img_index": _b["img_index"],
        }
        text_preload.append(batch_text_preload)
        # == End  : Add New Keys ==

    tiids = list()
    for pre in text_preload:
        tiids += pre["img_index"]
    tiids = torch.tensor(tiids)

    image_preload = list()
    for _b in tqdm.tqdm(image_loader, desc="image prefetch loop"):
        image_preload.append((_b['image'][0], _b["img_index"][0]))

    rank_scores = list()
    rank_iids = list()

    for img_batch in tqdm.tqdm(image_preload, desc="rank loop"):
        _im, _iid = img_batch

        img_batch_score = list()
        for txt_batch in text_preload:
            fblen = len(txt_batch["text_ids"])
            im = _im.repeat(fblen, 1, 1, 1).to(device=txt_batch['text_ids'].device)

            with torch.cuda.amp.autocast():
                # == Begin: Add New Keys ==
                batch_infer = {
                    "text_ids": txt_batch["text_ids"],
                    "text_masks": txt_batch["text_masks"],
                    "text_labels": txt_batch["text_labels"],
                }
                score = pl_module.irtr_head(pl_module.infer(batch_infer, img=im, )["multi_modal_cls_feats"])[:, 0]
                # == End  : Add New Keys ==

            img_batch_score.append(score)

        img_batch_score = torch.cat(img_batch_score)
        rank_scores.append(img_batch_score.cpu().tolist())
        rank_iids.append(_iid)

    torch.distributed.barrier()
    gather_rank_scores = all_gather(rank_scores)
    gather_rank_iids = all_gather(rank_iids)

    iids = torch.tensor(gather_rank_iids)
    iids = iids.view(-1)
    scores = torch.tensor(gather_rank_scores)
    scores = scores.view(len(iids), -1)

    topk10 = scores.topk(10, dim=1)
    topk5 = scores.topk(5, dim=1)
    topk1 = scores.topk(1, dim=1)
    topk10_iids = tiids[topk10.indices]
    topk5_iids = tiids[topk5.indices]
    topk1_iids = tiids[topk1.indices]

    tr_r10 = (iids.unsqueeze(1) == topk10_iids).float().max(dim=1)[0].mean()
    tr_r5 = (iids.unsqueeze(1) == topk5_iids).float().max(dim=1)[0].mean()
    tr_r1 = (iids.unsqueeze(1) == topk1_iids).float().max(dim=1)[0].mean()

    topk10 = scores.topk(10, dim=0)
    topk5 = scores.topk(5, dim=0)
    topk1 = scores.topk(1, dim=0)
    topk10_iids = iids[topk10.indices]
    topk5_iids = iids[topk5.indices]
    topk1_iids = iids[topk1.indices]

    ir_r10 = (tiids.unsqueeze(0) == topk10_iids).float().max(dim=0)[0].mean()
    ir_r5 = (tiids.unsqueeze(0) == topk5_iids).float().max(dim=0)[0].mean()
    ir_r1 = (tiids.unsqueeze(0) == topk1_iids).float().max(dim=0)[0].mean()

    return (ir_r1, ir_r5, ir_r10, tr_r1, tr_r5, tr_r10)


def visualize_attention(image, text2image_attns, image2text_attns):
    attention_map = text2image_attns[-1][0]  # 假设形状是 (B, H, W)
    # attention_map = image2text_attns[-1].mean(dim=1)  # 平均所有注意力头
    attention_map = attention_map[0, -3, :, :].cpu().detach().numpy()  # 选择第一个样本和第一个注意力头
    # 首先，找到全为0的列
    non_zero_columns = np.any(attention_map != 0, axis=0)

    # 其次，保留非零列
    attention_map = attention_map[:, non_zero_columns]
    attention_map = attention_map[:attention_map.shape[1], :]

    attention_map = torch.from_numpy(attention_map)

    # 检查并调整维度，如果只有 2 维则添加批次和通道维度
    if len(attention_map.shape) == 2:
        attention_map = attention_map.unsqueeze(0).unsqueeze(0)  # 使其形状变为 (1, 1, H, W)
    elif len(attention_map.shape) == 3:  # 如果是 (C, H, W) 则添加批次维度
        attention_map = attention_map.unsqueeze(0)  # 使其形状变为 (1, C, H, W)

    # 调整注意力图的大小以匹配图像
    attn_map = F.interpolate(attention_map, size=image.shape[-2:], mode='bilinear',
                             align_corners=False).squeeze()

    # # 归一化注意力图
    attn_map = attn_map.cpu().detach().numpy()
    attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min())

    image = image[:, 0, :, :]

    # # 将注意力图叠加在图像上
    plt.imshow(image.permute(1, 2, 0).cpu().numpy(), cmap='gray')
    plt.imshow(attn_map, cmap='jet', alpha=0.5)  # 调整 alpha 以控制透明度
    plt.axis('off')
    plt.show()

# Example usage:
# Assuming `image` is a tensor of shape (C, H, W)
# visualize_attention(image, ret["attentions"]["text2image_attns"], ret["attentions"]["image2text_attns"])
