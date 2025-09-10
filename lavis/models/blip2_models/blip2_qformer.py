"""
 Copyright (c) 2023, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""
import logging

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.cuda.amp import autocast as autocast
from torch.nn import functional as F

from lavis.common.registry import registry
from lavis.models.base_model import all_gather_with_grad, concat_all_gather
from lavis.models.blip2_models.blip2 import (
    Blip2Base,
    compute_sim_matrix,
    disabled_train,
)
from lavis.models.blip_models.blip_outputs import BlipOutput, BlipOutputFeatures


@registry.register_model("blip2")
@registry.register_model("blip2_feature_extractor")
class Blip2Qformer(Blip2Base):
    """
    BLIP2 first-stage model with Q-former and ViT.
    Supported model types:
        - pretrained: pretrained model with vit-g
        - pretrain_vitL: pretrained model with vit-large
        - coco: fintuned model on coco
    Usage:
        >>> from lavis.models import load_model
        >>> model = load_model("blip2", "pretrain")
    """

    # 保存了不同模型类型对应的配置文件
    PRETRAINED_MODEL_CONFIG_DICT = {
        "pretrain": "configs/models/blip2/blip2_pretrain.yaml",
        "pretrain_vitL": "configs/models/blip2/blip2_pretrain_vitL.yaml",
        "coco": "configs/models/blip2/blip2_coco.yaml",
    }

    def __init__(
        self,
        vit_model="eva_clip_g",
        img_size=224,
        drop_path_rate=0,
        use_grad_checkpoint=False,
        vit_precision="fp16",
        freeze_vit=True, # 冻结视觉编码器的参数
        num_query_token=32,  # 查询token数
        cross_attention_freq=2,
        embed_dim=256,
        max_txt_len=32,
        qformer_text_input=True,
    ):
        super().__init__()

        # 初始化tokenizer
        self.tokenizer = self.init_tokenizer()

        # 初始化视觉编码器和视觉归一化层
        # self.visual_encoder, self.ln_vision = self.init_vision_encoder(
        #     vit_model, img_size, drop_path_rate, use_grad_checkpoint, vit_precision
        # )

        # 冻结视觉编码器的参数
        # if freeze_vit:
        #     for name, param in self.visual_encoder.named_parameters():
        #         param.requires_grad = False
        #     self.visual_encoder = self.visual_encoder.eval()
        #     self.visual_encoder.train = disabled_train
        #     logging.info("freeze vision encoder")

        # 初始化Q-former和查询token
        self.Qformer, self.query_tokens = self.init_Qformer(
            num_query_token, 768, cross_attention_freq
        )
        # 调整查询token的嵌入大小
        if not qformer_text_input:
            self.Qformer.bert.embeddings.word_embeddings = None
            self.Qformer.bert.embeddings.position_embeddings = None
            for layer in self.Qformer.bert.encoder.layer:
                layer.output = None
                layer.intermediate = None
        else:
            # 初始化tokenizer
            self.tokenizer = self.init_tokenizer()
            self.Qformer.resize_token_embeddings(len(self.tokenizer))

        # 获取当前模型的状态字典
        state_dict = self.Qformer.state_dict()
        # 初始化_query参数
        for name, param in self.Qformer.named_parameters():
            if "_query" in name:
                key_orig = name.replace("_query", "")
                param.data.copy_(state_dict[key_orig])

        # 定义几个线性层用于图像和文本的投影
        self.vision_proj = nn.Linear(self.Qformer.config.hidden_size, embed_dim)
        self.text_proj = nn.Linear(self.Qformer.config.hidden_size, embed_dim)

        # self.itm_head = nn.Linear(self.Qformer.config.hidden_size, 2)

        # 定义一个温度参数用于对比损失计算
        self.temp = nn.Parameter(0.07 * torch.ones([]))

        self.max_txt_len = max_txt_len

    def cross_entropy_with_label_smoothing(self, logits, targets, label_smoothing=0.1):
        num_classes = logits.size(1)
        confidence = 1.0 - label_smoothing
        smoothed_labels = torch.full_like(logits, fill_value=label_smoothing / (num_classes - 1))
        smoothed_labels.scatter_(1, targets.unsqueeze(1), confidence)
        return F.kl_div(F.log_softmax(logits, dim=-1), smoothed_labels, reduction='batchmean')

    def forward(self, text_embeds, image_embeds):
        """
        text: [batch_size, seq_len, dim] 8,32,768
        image:[batch_size, cls+576, dim] 8,577,768
        """
        # 提取图像和文本样本
        # image = samples["image"]
        # text = samples["text_input"]

        # 图像特征提取
        # image_embeds = self.ln_vision(self.visual_encoder(image))
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
            image_embeds.device
        )

        # text_atts = torch.ones(text_embeds.size()[:-1], dtype=torch.long).to(
        #     image_embeds.device
        # )

        # 扩展查询令牌以适应批量的输入图像特征
        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)

        query_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            use_cache=True,
            return_dict=True,
        )

        # 归一化图像特征
        image_feats = F.normalize(
            self.vision_proj(query_output.last_hidden_state), dim=-1
        )

        # 模型基于图像的特征和查询令牌进行联合编码
        # if self.qformer_text_input:
        #     text_Qformer = self.tokenizer(
        #         text_embeds,
        #         padding='longest',
        #         truncation=True,
        #         max_length=self.max_txt_len,
        #         return_tensors="pt",
        #     ).to(image_embeds.device)

        # query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(image_embeds.device)
        # Qformer_atts = torch.cat([query_atts, text_atts], dim=1)
        #
        #
        # query_output = self.Qformer.bert(
        #     text_embeds,
        #     attention_mask=Qformer_atts,
        #     query_embeds=query_tokens,
        #     encoder_hidden_states=image_embeds,
        #     encoder_attention_mask=image_atts,
        #     return_dict=True,
        # )
        # else:
        #     query_output = self.Qformer.bert(
        #         query_embeds=query_tokens,
        #         encoder_hidden_states=image_embeds,
        #         encoder_attention_mask=image_atts,
        #         return_dict=True,
        #     )



        # 处理文本输入
        # text_tokens = self.tokenizer(
        #     text,
        #     padding="max_length",
        #     truncation=True,
        #     max_length=self.max_txt_len,
        #     return_tensors="pt",
        # ).to(image_embeds.device)
        # 文本特征提取
        # text_output = self.Qformer.bert(
        #     text_ids,
        #     attention_mask=text_atts_mask,
        #     return_dict=True,
        # )
        # 文本特征归一化
        text_feat = F.normalize(
            self.text_proj(text_embeds[:, 0, :]), dim=-1
        )

        ###============== Image-text Contrastive ===================###
        # 特征聚合
        image_feats_all = concat_all_gather(
            image_feats
        )  # [batch_size*num_gpu, num_query_tokens, embed_dim]
        text_feat_all = concat_all_gather(text_feat)  # [batch_size*num_gpu, embed_dim]

        # 计算计算图像到文本的相似度
        sim_q2t = torch.matmul(
            image_feats.unsqueeze(1), text_feat_all.unsqueeze(-1)
        ).squeeze()
        # [batch_size, batch_size*num_gpu, num_query_tokens]

        # 计算计算文本到图像的相似度
        # image-text similarity: aggregate across all query tokens
        sim_i2t, _ = sim_q2t.max(-1)
        sim_i2t = sim_i2t / self.temp

        # 计算文本到图像的相似度
        # text-query similarity: [batch_size, batch_size*num_gpu, num_query_tokens]
        sim_t2q = torch.matmul(
            text_feat.unsqueeze(1).unsqueeze(1), image_feats_all.permute(0, 2, 1)
        ).squeeze()

        #
        # text-image similarity: aggregate across all query tokens
        sim_t2i, _ = sim_t2q.max(-1)
        sim_t2i = sim_t2i / self.temp  # [batch_size, batch_size*num_gpu]

        # 生成目标标签
        rank = dist.get_rank()
        bs = image_embeds.size(0)
        targets = torch.linspace(rank * bs, rank * bs + bs - 1, bs, dtype=int).to(
            image_embeds.device
        )

        # if "image_id" in samples.keys(): #coco retrieval finetuning
        #     image_ids = samples["image_id"].view(-1,1)
        #     image_ids_all = concat_all_gather(image_ids)
        #     pos_idx = torch.eq(image_ids, image_ids_all.t()).float()
        #     sim_targets = pos_idx / pos_idx.sum(1,keepdim=True)
        #     sim_targets = 0.9 * sim_targets + 0.1 * torch.ones_like(sim_targets) / sim_targets.size(1)
        #
        #     loss_t2i = -torch.sum(F.log_softmax(sim_t2i, dim=1)*sim_targets,dim=1).mean()
        #     loss_i2t = -torch.sum(F.log_softmax(sim_i2t, dim=1)*sim_targets,dim=1).mean()
        #     loss_itc = (loss_t2i+loss_i2t)/2
        # else:
        # loss_itc = (
        #     F.cross_entropy(sim_i2t, targets, label_smoothing=0.1)
        #     + F.cross_entropy(sim_t2i, targets, label_smoothing=0.1)
        # ) / 2

        loss_itc = (
               self.cross_entropy_with_label_smoothing(sim_i2t, targets, label_smoothing=0.1)
               + self.cross_entropy_with_label_smoothing(sim_t2i, targets, label_smoothing=0.1)
        ) / 2
        # loss_itc = 0
        ###============== Image-text Matching ===================###
        # text_input_ids_world = concat_all_gather(text_tokens.input_ids)
        # text_attention_mask_world = concat_all_gather(text_tokens.attention_mask)
        # image_embeds_world = all_gather_with_grad(image_embeds)
        # with torch.no_grad():
        #     if "image_id" in samples.keys():
        #         mask = torch.eq(image_ids, image_ids_all.t())
        #         sim_t2i.masked_fill_(mask, -10000)
        #         sim_i2t.masked_fill_(mask, -10000)
        #     else:
        #         sim_t2i[:, rank * bs : rank * bs + bs].fill_diagonal_(-10000)
        #         sim_i2t[:, rank * bs : rank * bs + bs].fill_diagonal_(-10000)
        #
        #     weights_t2i = F.softmax(sim_t2i, dim=1)
        #     weights_i2t = F.softmax(sim_i2t, dim=1)
        #
        # # select a negative image for each text
        # image_embeds_neg = []
        # for b in range(bs):
        #     neg_idx = torch.multinomial(weights_t2i[b], 1).item()
        #     image_embeds_neg.append(image_embeds_world[neg_idx])
        # image_embeds_neg = torch.stack(image_embeds_neg, dim=0)
        #
        # # select a negative text for each image
        # text_ids_neg = []
        # text_atts_neg = []
        # for b in range(bs):
        #     neg_idx = torch.multinomial(weights_i2t[b], 1).item()
        #     text_ids_neg.append(text_input_ids_world[neg_idx])
        #     text_atts_neg.append(text_attention_mask_world[neg_idx])
        #
        # text_ids_neg = torch.stack(text_ids_neg, dim=0)
        # text_atts_neg = torch.stack(text_atts_neg, dim=0)
        #
        # text_ids_all = torch.cat(
        #     [text_tokens.input_ids, text_tokens.input_ids, text_ids_neg], dim=0
        # )  # pos, pos, neg
        # text_atts_all = torch.cat(
        #     [text_tokens.attention_mask, text_tokens.attention_mask, text_atts_neg],
        #     dim=0,
        # )
        #
        # query_tokens_itm = self.query_tokens.expand(text_ids_all.shape[0], -1, -1)
        # query_atts_itm = torch.ones(query_tokens_itm.size()[:-1], dtype=torch.long).to(
        #     image.device
        # )
        # attention_mask_all = torch.cat([query_atts_itm, text_atts_all], dim=1)
        #
        # image_embeds_all = torch.cat(
        #     [image_embeds, image_embeds_neg, image_embeds], dim=0
        # )  # pos, neg, pos
        # image_atts_all = torch.ones(image_embeds_all.size()[:-1], dtype=torch.long).to(
        #     image.device
        # )
        #
        # output_itm = self.Qformer.bert(
        #     text_ids_all,
        #     query_embeds=query_tokens_itm,
        #     attention_mask=attention_mask_all,
        #     encoder_hidden_states=image_embeds_all,
        #     encoder_attention_mask=image_atts_all,
        #     return_dict=True,
        # )
        #
        # vl_embeddings = output_itm.last_hidden_state[:, : query_tokens_itm.size(1), :]
        # vl_output = self.itm_head(vl_embeddings)
        # logits = vl_output.mean(dim=1)
        #
        # itm_labels = torch.cat(
        #     [torch.ones(bs, dtype=torch.long), torch.zeros(2 * bs, dtype=torch.long)],
        #     dim=0,
        # ).to(image.device)
        # loss_itm = F.cross_entropy(logits, itm_labels)

        ##================= Image Captioning ========================##
        # decoder_input_ids = text_tokens.input_ids.clone()
        # decoder_input_ids[:, 0] = self.tokenizer.bos_token_id
        # labels = decoder_input_ids.masked_fill(
        #     decoder_input_ids == self.tokenizer.pad_token_id, -100
        # )
        #
        # query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(
        #     image.device
        # )
        # attention_mask = torch.cat([query_atts, text_tokens.attention_mask], dim=1)
        # lm_output = self.Qformer(
        #     decoder_input_ids,
        #     attention_mask=attention_mask,
        #     past_key_values=query_output.past_key_values,
        #     return_dict=True,
        #     labels=labels,
        # )
        #
        # loss_lm = lm_output.loss

        return query_output.last_hidden_state, loss_itc
        # return BlipOutput(
        #     intermediate_output=query_output.last_hidden_state,
        #     loss_itc=loss_itc,
        #     # loss_itm=loss_itm,
        #     # loss_lm=loss_lm,
        # )


    @torch.no_grad()
    def generate(
        self,
        samples,
        use_nucleus_sampling=False,
        num_beams=3,
        max_length=30,
        min_length=10,
        top_p=0.9,
        repetition_penalty=1.0,
    ):
        """
        Args:
            samples (dict): A dictionary containing the following keys:
                - image (torch.Tensor): A tensor of shape (batch_size, 3, H, W)
            use_nucleus_sampling (bool): Whether to use nucleus sampling. If False, use top-k sampling.
            num_beams (int): Number of beams for beam search. 1 means no beam search.
            max_length (int): The maximum length of the sequence to be generated.
            min_length (int): The minimum length of the sequence to be generated.
            top_p (float): The cumulative probability for nucleus sampling.
            repetition_penalty (float): The parameter for repetition penalty. 1.0 means no penalty.
            num_captions (int): Number of captions to be generated for each image.
        Returns:
            captions (list): A list of strings of length batch_size * num_captions.
        """
        image = samples["image"]
        image_embeds = self.ln_vision(self.visual_encoder(image))

        if not use_nucleus_sampling:
            image_embeds = image_embeds.repeat_interleave(num_beams, dim=0)
        else:
            num_beams = 1
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
            image.device
        )

        model_kwargs = {
            "encoder_hidden_states": image_embeds,
            "encoder_attention_mask": image_atts,
        }

        input_ids = (
            torch.LongTensor(image.size(0), 1)
            .fill_(self.tokenizer.bos_token_id)
            .to(image.device)
        )
        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)

        outputs = self.Qformer.generate(
            input_ids=input_ids,
            query_embeds=query_tokens,
            max_length=max_length,
            min_length=min_length,
            num_beams=num_beams,
            do_sample=use_nucleus_sampling,
            top_p=top_p,
            eos_token_id=self.tokenizer.sep_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            **model_kwargs
        )
        captions = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return captions

    def forward_image(self, image):
        image_embeds = self.ln_vision(self.visual_encoder(image))
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
            image.device
        )

        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)

        query_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )
        return query_output.last_hidden_state, image_embeds

    def forward_text(self, text_tokens):
        text_output = self.Qformer.bert(
            text_tokens.input_ids,
            attention_mask=text_tokens.attention_mask,
            return_dict=True,
        )
        return text_output.last_hidden_state[:, 0, :]

    def compute_itm(self, image_inputs, text_ids, text_atts):
        image_atts = torch.ones(image_inputs.size()[:-1], dtype=torch.long).to(
            image_inputs.device
        )
        query_tokens = self.query_tokens.expand(image_inputs.shape[0], -1, -1)
        query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(
            image_inputs.device
        )
        attention_mask = torch.cat([query_atts, text_atts], dim=1)
        output_itm = self.Qformer.bert(
            text_ids,
            query_embeds=query_tokens,
            attention_mask=attention_mask,
            encoder_hidden_states=image_inputs,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )
        vl_embeddings = output_itm.last_hidden_state[:, : query_tokens.size(1), :]
        itm_logit = self.itm_head(vl_embeddings)
        itm_logit = itm_logit[:, :, 1].mean(dim=1)
        return itm_logit

    @torch.no_grad()
    def extract_features(self, samples, mode="multimodal"):
        """
        Extract features for multimodal or unimodal samples.
        Args:
            samples (dict): A dictionary of samples, containing the following keys:
                - image (torch.Tensor): A tensor of shape (B, C, H, W) containing the image.
                    Raw images should be preprocessed before being passed to feature extractor.
                - text_input (list): A list of strings containing the text, length B.
            mode (str): The mode of feature extraction. Can be either "multimodal", "text" or "image".
                If "multimodal", return image features and multimodal features;
                if "text", return text features;
                if "image", return image features.
                Default: "multimodal".
        Returns:
            BlipOutputFeatures: A BlipOutputFeatures object containing the features.
                See lavis/models/blip_models/blip_outputs.py for more details.
        """
        image = samples.get("image")
        caption = samples.get("text_input")

        # assert mode is one of "image", "text", "multimodal"
        assert mode in [
            "image",
            "text",
            "multimodal",
        ], "mode must be one of 'image', 'text', 'multimodal'"

        # initalize output
        image_embeds, text_embeds, multimodal_embeds = None, None, None
        image_features, text_features = None, None

        if mode == "image":
            assert (
                image is not None
            ), "Image is not provided for mode 'image' or 'multimodal'"
            # return query features
            with self.maybe_autocast():
                image_embeds_frozen = self.ln_vision(self.visual_encoder(image))
            image_embeds_frozen = image_embeds_frozen.float()
            image_atts = torch.ones(
                image_embeds_frozen.size()[:-1], dtype=torch.long
            ).to(self.device)
            query_tokens = self.query_tokens.expand(
                image_embeds_frozen.shape[0], -1, -1
            )

            query_output = self.Qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds_frozen,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )
            image_embeds = query_output.last_hidden_state
            image_features = F.normalize(self.vision_proj(image_embeds), dim=-1)

        elif mode == "text":
            assert (
                caption is not None
            ), "text input is None for mode 'text' or 'multimodal'"

            # return text features
            text = self.tokenizer(caption, return_tensors="pt", padding=True).to(
                self.device
            )

            text_output = self.Qformer.bert(
                text.input_ids,
                attention_mask=text.attention_mask,
                return_dict=True,
            )
            text_embeds = text_output.last_hidden_state
            text_features = self.text_proj(text_embeds)
            text_features = F.normalize(text_features, dim=-1)

        elif mode == "multimodal":
            # return multimodel query features
            with self.maybe_autocast():
                image_embeds_frozen = self.ln_vision(self.visual_encoder(image))
            image_embeds_frozen = image_embeds_frozen.float()
            image_atts = torch.ones(
                image_embeds_frozen.size()[:-1], dtype=torch.long
            ).to(self.device)
            query_tokens = self.query_tokens.expand(
                image_embeds_frozen.shape[0], -1, -1
            )
            query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(
                self.device
            )

            text = self.tokenizer(caption, return_tensors="pt", padding=True).to(
                self.device
            )
            attention_mask = torch.cat([query_atts, text.attention_mask], dim=1)

            output = self.Qformer.bert(
                text.input_ids,
                query_embeds=query_tokens,
                attention_mask=attention_mask,
                encoder_hidden_states=image_embeds_frozen,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )

            multimodal_embeds = output.last_hidden_state[:, : query_tokens.size(1), :]

        return BlipOutputFeatures(
            image_embeds=image_embeds,
            image_embeds_proj=image_features,
            text_embeds=text_embeds,
            text_embeds_proj=text_features,
            multimodal_embeds=multimodal_embeds,
        )

    @classmethod
    def from_config(cls, cfg):
        vit_model = cfg.get("vit_model", "eva_clip_g")
        img_size = cfg.get("image_size")
        num_query_token = cfg.get("num_query_token")
        cross_attention_freq = cfg.get("cross_attention_freq", 2)

        drop_path_rate = cfg.get("drop_path_rate", 0)
        use_grad_checkpoint = cfg.get("use_grad_checkpoint", False)
        vit_precision = cfg.get("vit_precision", "fp16")
        freeze_vit = cfg.get("freeze_vit", True)

        max_txt_len = cfg.get("max_txt_len", 32)

        model = cls(
            vit_model=vit_model,
            img_size=img_size,
            drop_path_rate=drop_path_rate,
            use_grad_checkpoint=use_grad_checkpoint,
            vit_precision=vit_precision,
            freeze_vit=freeze_vit,
            num_query_token=num_query_token,
            cross_attention_freq=cross_attention_freq,
            max_txt_len=max_txt_len,
        )
        model.load_checkpoint_from_config(cfg)

        return model

    def compute_sim_matrix(self, data_loader, task_cfg):
        """
        Compute similarity i2t, t2i matrix for the given data loader.
        """
        k_test = task_cfg.k_test

        return compute_sim_matrix(model=self, data_loader=data_loader, k_test=k_test)
