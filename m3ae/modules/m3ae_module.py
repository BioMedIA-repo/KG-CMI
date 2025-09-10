import math
import random

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.utils import clip_grad_norm_

from transformers import RobertaConfig, RobertaModel, RobertaForCausalLM, AutoTokenizer
from transformers.models.bert.modeling_bert import BertConfig, BertModel
from transformers import T5Tokenizer, T5ForConditionalGeneration

from m3ae.modules import objectives, m3ae_utils
from m3ae.modules import prediction_heads
from m3ae.modules.language_encoders.bert_model import BertCrossLayer, GAT_module
from m3ae.modules.m3ae_utils import init_weights
from m3ae.modules.vision_encoders import swin_transformer as swin
from m3ae.modules.vision_encoders.clip_model import build_model, adapt_position_encoding
from m3ae.modules.vision_encoders.swin_helpers import swin_adapt_position_encoding

from lavis.models.blip2_models.blip2_qformer import Blip2Qformer

from transformers import RobertaTokenizerFast, DataCollatorForWholeWordMask, BatchEncoding
from m3ae.modules.mambapy.mamba import Mamba, MambaConfig

class M3AETransformerSS(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()  # 保存超参数
        self.config = config

        # == Begin: 1. Build Models ==
        self.is_clip = ('swin' not in config['vit'])  # 判断是否使用clip模型

        if 'roberta' in config['tokenizer']:
            bert_config = RobertaConfig(
                vocab_size=config["vocab_size"],
                hidden_size=config["hidden_size"],
                num_hidden_layers=config["num_layers"],
                num_attention_heads=config["num_heads"],
                intermediate_size=config["hidden_size"] * config["mlp_ratio"],
                max_position_embeddings=config["max_text_len"],
                hidden_dropout_prob=config["drop_rate"],
                attention_probs_dropout_prob=config["drop_rate"],
            )
        elif 'bert' in config['tokenizer']:
            bert_config = BertConfig(
                vocab_size=config["vocab_size"],
                hidden_size=config["hidden_size"],
                num_hidden_layers=config["num_layers"],
                num_attention_heads=config["num_heads"],
                intermediate_size=config["hidden_size"] * config["mlp_ratio"],
                max_position_embeddings=config["max_text_len"],
                hidden_dropout_prob=config["drop_rate"],
                attention_probs_dropout_prob=config["drop_rate"],
            )
        else:
            raise ValueError

        resolution_after = config['image_size']
        if torch.distributed.is_initialized():  # 检查是否初始化了分布式训练环境，
            if torch.distributed.get_rank() == 0:  # 只在主进程加载模型

                if self.is_clip:  # 加载视觉模型
                    build_model(config['vit'], resolution_after=resolution_after)
                else:
                    getattr(swin, self.hparams.config["vit"])(pretrained=True, config=self.hparams.config)

                if 'roberta' in config['tokenizer']:  # 加载语言模型
                    RobertaModel.from_pretrained(config['tokenizer'])
                else:
                    BertModel.from_pretrained(config['tokenizer'])
            torch.distributed.barrier()

        # 不使用分布式训练
        if self.is_clip:
            self.vision_encoder = build_model(config['vit'], resolution_after=resolution_after)
        else:
            self.vision_encoder = getattr(swin, self.hparams.config["vit"])(pretrained=True, config=self.hparams.config)
            self.vision_pooler = nn.AdaptiveAvgPool1d(1)

        if 'roberta' in config['tokenizer']:
            self.language_encoder = RobertaModel.from_pretrained(config['tokenizer'])
            # self.tokenizer = AutoTokenizer.from_pretrained(config['tokenizer'])
            # self.language_decoder = RobertaForCausalLM.from_pretrained(config['tokenizer'])
        else:
            self.language_encoder = BertModel.from_pretrained(config['tokenizer'])

        self.tokenizer = T5Tokenizer.from_pretrained(config['T5_model'])
        self.language_decoder = T5ForConditionalGeneration.from_pretrained(config['T5_model'])

        self.multi_modal_language_proj = nn.Linear(config['input_text_embed_size'],
                                                   config['hidden_size'])  # 创建一个语言线性层，用于将输入的文本嵌入投影到隐藏层

        # 用于对齐图文信息
        self.language_proj = nn.Linear(config['input_text_embed_size'], config['embed_dim'])

        self.multi_modal_language_proj.apply(init_weights)  # 将初始化权重函数应用到语言线性层，以便对该层的权重进行初始化
        self.multi_modal_vision_proj = nn.Linear(config['input_image_embed_size'], config['hidden_size'])  # 视觉同理

        # 用于对齐图文信息
        self.vision_proj = nn.Linear(config['input_text_embed_size'], config['embed_dim'])

        self.multi_modal_vision_proj.apply(init_weights)

        self.modality_type_embeddings = nn.Embedding(2, config["hidden_size"])  # 创建一个嵌入层，用于表示两种模态（文本和视觉）
        self.modality_type_embeddings.apply(init_weights)

        # == Begin: External Graph ==
        used_device = f"cuda:{config['gpu_ids']}" if torch.cuda.is_available() else "cpu"
        self.latent_prompt_size = config['latent_prompt_size']  # 32
        self.organ_disease_feat_size = 577  # it's fixed
        # self.GAT_feat = self.latent_prompt_size // 8 if self.latent_prompt_size >= 8 else 1
        self.GAT_feat = 32
        self.adj_feat = torch.load(config["adj_feat_path"]).to(used_device)
        self.organ_disease_feat = torch.load(config["organ_disease_feat"]).to(used_device)
        self.GAT_layer = GAT_module(config["hidden_size"])
        self.organ_average = nn.Conv1d(self.organ_disease_feat_size, self.GAT_feat, 1)
        self.organ_average.apply(init_weights)
        # == End  : External Graph ==

        # 初始化qformer
        self.qformer = Blip2Qformer()

        self.multi_modal_vision_pooler = prediction_heads.Pooler(config["hidden_size"])  # 创建一个视觉池层 multi_modal_vision_pooler，用于对视觉特征进行池化，以提取出最终的聚合特征
        self.multi_modal_vision_pooler.apply(init_weights)
        self.multi_modal_language_pooler = prediction_heads.Pooler(config["hidden_size"])
        self.multi_modal_language_pooler.apply(init_weights)
        # == End  : 1. Build Models ==

        # == Begin: Build Momentum Model ==
        # 定义一个温度参数用于对比损失计算
        itc = False
        if itc:
            self.temp = nn.Parameter(torch.ones(1) * config['temp'])  # 创建一个参数 temp，用于控制模型的学习率
            self.queue_size = config['queue_size']  # 队列大小
            self.momentum = config['momentum']  # 动量参数

            self.vision_encoder_m = build_model(config['vit'], resolution_after=resolution_after)
            self.vision_proj_m = nn.Linear(config['input_image_embed_size'], config['embed_dim'])

            self.language_encoder_m = RobertaModel.from_pretrained(config['tokenizer'])
            self.language_proj_m = nn.Linear(config['input_text_embed_size'], config['embed_dim'])

            self.model_pairs = [[self.vision_encoder, self.vision_encoder_m],
                                [self.vision_proj, self.vision_proj_m],
                                [self.language_encoder, self.language_encoder_m],
                                [self.language_proj, self.language_proj_m],
                                ]  # 创建模型对，用于后续模型的对比

            self.copy_params()

            self.register_buffer("image_queue", torch.randn(config['embed_dim'], self.queue_size))  # 注册图像队列存储嵌入特征
            self.register_buffer("text_queue", torch.randn(config['embed_dim'], self.queue_size))  # 注册文本队列
            self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))  # 注册队列指针

            self.image_queue = nn.functional.normalize(self.image_queue, dim=0)  # 归一化图像队列
            self.text_queue = nn.functional.normalize(self.text_queue, dim=0)  # 归一化文本队列
        # == End  : Build Momentum Model ==

        # == Begin: 2. Build Pre-Training Heads ==
        if config["loss_names"]["mlm"] > 0:  # 判断是否开启了掩码语言建模（Masked Language Modeling, MLM）损失的计算
            self.mlm_head = prediction_heads.MLMHead(bert_config)
            self.mlm_head.apply(init_weights)
        if config["loss_names"]["mim"] > 0:  # 判断是否开启了掩码图像建模（Masked Image Modeling, MIM）损失的计算
            self.mim_head = prediction_heads.MIMHead(config)
            self.mim_head.apply(init_weights)
        if config["loss_names"]["itm"] > 0 or self.hparams.config["loss_names"][
            "irtr"] > 0:  # 判断是否开启了图像文本匹配（Image-Text Matching, ITM）或图像检索文本检索（Image Retrieval and Text Retrieval, IRTR）的计算
            self.itm_head = prediction_heads.ITMHead(config["hidden_size"] * 2)
            self.itm_head.apply(init_weights)
        # == End  : 2. Build Pre-Training Heads ==

        # == Begin: 3. Load Models ==
        # 在模型初始化时，从指定的路径加载预训练权重，并根据模型的类型调整位置编码，以确保权重能够正确地应用于当前的模型设置。
        if self.hparams.config["load_path"] != "" and not self.hparams.config["test_only"]:
            ckpt = torch.load(self.hparams.config["load_path"], map_location="cpu")
            state_dict = ckpt["state_dict"]
            if self.is_clip:
                state_dict = adapt_position_encoding(state_dict,
                                                     after=resolution_after,
                                                     patch_size=self.hparams.config['patch_size'])
            else:
                state_dict = swin_adapt_position_encoding(state_dict, after=resolution_after)
            self.load_state_dict(state_dict, strict=False)
        # == End  : 3. Load Models ==

        # == 4. Build Heads For Downstream Tasks ==
        hs = self.hparams.config["hidden_size"]  # 获取配置中设置的隐藏层大小 (hidden_size) 并赋值给变量 hs，该值在后续层的定义中将会使用
        if self.hparams.config["loss_names"]["vqa"] > 0:
            vs = self.hparams.config["vqa_label_size"]  # 从配置中获取 VQA 标签的大小（vqa_label_size），并将其赋值给变量 vs
            self.vqa_head = nn.Sequential(
                nn.Linear(hs * 2, hs * 2),
                nn.LayerNorm(hs * 2),
                nn.GELU(),
                nn.Linear(hs * 2, vs),
            )  # 接受来自视觉模态和语言模态的拼接特征，通过线性变换、层归一化和非线性激活的组合处理后，最终输出 VQA 任务的预测结果（对应不同答案的概率或得分）
            self.vqa_head.apply(init_weights)

        if self.hparams.config["loss_names"]["cls"] > 0:
            ms = self.hparams.config["melinda_label_size"][self.hparams.config["label_column_name"]]
            self.cls_head = nn.Sequential(
                nn.Linear(hs * 2, hs * 2),
                nn.LayerNorm(hs * 2),
                nn.GELU(),
                nn.Linear(hs * 2, ms),
            )
            self.cls_head.apply(init_weights)

        if self.hparams.config["loss_names"]["irtr"] > 0:
            self.irtr_head = nn.Linear(hs * 2, 1)
            self.irtr_head.weight.data = self.itm_head.fc.weight.data[1:, :]
            self.irtr_head.bias.data = self.itm_head.fc.bias.data[1:]
            self.margin = 0.2
            for p in self.itm_head.parameters():
                p.requires_grad = False

        m3ae_utils.set_metrics(self)  # 调用 m3ae_utils 模块中的 set_metrics 函数，用于初始化度量指标，以便在训练和评估过程中进行监控
        self.current_tasks = list()
        # == End:  4. Build Heads For Downstream Tasks ==

        # == Begin: 5. Load Models For Testing ==
        if self.hparams.config["load_path"] != "" and self.hparams.config["test_only"]:
            ckpt = torch.load(self.hparams.config["load_path"], map_location="cpu")
            state_dict = ckpt["state_dict"]
            state_dict = adapt_position_encoding(state_dict, after=resolution_after,
                                                 patch_size=self.hparams.config['patch_size'])
            self.load_state_dict(state_dict, strict=False)
        # == End  : 5. Load Models For Testing ==

    # random_masking 方法用于生成掩码信号，会随机保留输入特征中的一部分，同时通过生成掩码张量标记哪些特征被保留或移除。
    # 该方法通常用于训练中的自监督学习任务，如掩码图像建模（MIM），它能帮助模型学习对输入的不完整部分进行推断。
    def random_masking(self, x, mask_ratio):
        x_ = x[:, :1]  # 将输入 x 的第一个元素（通常是 CLS token）提取出来，并将其赋值给 x_
        x = x[:, 1:]  # 剩下的赋值给x
        pos_embed = self.vision_encoder.visual.positional_embedding.unsqueeze(0).to(x)  # 获取视觉编码器中的位置嵌入（positional_embedding），并在第 0 维增加一个维度（批次维度），使其与后续计算的一致

        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x += pos_embed[:, 1:]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0

        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        # append cls token
        x_ = x_ + pos_embed[:, :1]
        x_masked = torch.cat((x_, x_masked), dim=1)

        return x_masked, mask, ids_restore

    # 将图像切分成小块
    def patchify(self, imgs):
        p = self.hparams.config["patch_size"]
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0
        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p ** 2 * 3))
        return x

    def unpatchify(self, x):
        p = self.hparams.config["patch_size"]
        h = w = int(x.shape[1] ** .5)
        assert h * w == x.shape[1]
        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
        return imgs

    # 进行多模态推理
    def infer(
            self,
            batch,
            # test,
            mask_text=False,
            mask_image=False,
            image_token_type_idx=1,
            img=None,
            output_attentions=True,
            itc=False,
            unimodal=False,
    ):
        ret = dict()

        # == Begin: Fetch the inputs ==
        if img is None:
            if f"image_{image_token_type_idx - 1}" in batch:
                img_key = f"image_{image_token_type_idx - 1}"
            else:
                img_key = "image"
            img = batch[img_key][0]  # 获取batch中的image数据
        device = img.device
        do_mlm = "_mlm" if mask_text else ""
        text_ids = batch[f"text_ids{do_mlm}"].to(device)
        text_labels = batch[f"text_labels{do_mlm}"].to(device)
        text_masks = batch[f"text_masks"].to(device)

        vqa_answers = [item[0] for item in batch["vqa_answer"]]
        answer = self.tokenizer(vqa_answers, padding=True, return_tensors="pt").to(device)
        answer_targets = answer.input_ids.masked_fill(answer.input_ids == self.tokenizer.pad_token_id, -100).to(device)
        # == End  : Fetch the inputs ==

        # == Begin: Text Encoding ==
        # 通过语言编码器对文本进行嵌入，应用注意力机制和多个层的处理，得到适合与其他模态结合的文本特征。
        uni_modal_text_feats = self.language_encoder.embeddings(input_ids=text_ids)
        text_input_shape = text_masks.size()
        extended_text_masks = self.language_encoder.get_extended_attention_mask(text_masks, text_input_shape, device)
        for layer in self.language_encoder.encoder.layer:
            uni_modal_text_feats = layer(uni_modal_text_feats, extended_text_masks)[0]
        # uni_modal_text_feats = self.multi_modal_language_proj(uni_modal_text_feats)

        text_feat = F.normalize(self.language_proj(uni_modal_text_feats[:, 0, :]), dim=-1)

        # == End  : Text Encoding ==

        # == Begin: Image Encoding ==
        if mask_image:
            # == Begin: Image Masking ==
            # Mask: length -> length * mask_ratio
            # Perform position embedding inside the masking function
            uni_modal_image_feats = self.vision_encoder.forward_patch_embed(img)
            uni_modal_image_feats, mim_masks, mim_ids_restore = self.random_masking(uni_modal_image_feats,
                                                                                    self.hparams.config["mim_prob"])
            uni_modal_image_feats = self.vision_encoder.forward_trans(uni_modal_image_feats)
            ret["mim_masks"] = mim_masks
            ret["mim_ids_restore"] = mim_ids_restore
            # == End  : Image Masking ==
        else:
            uni_modal_image_feats = self.vision_encoder(img)
        # uni_modal_image_feats = self.multi_modal_vision_proj(uni_modal_image_feats)

        # image_atts =  torch.ones(uni_modal_image_feats()[:-1], dtype=torch.long).to(device)
        image_feat = F.normalize(self.vision_proj(uni_modal_image_feats[:, 0, :]), dim=-1)

        image_masks = torch.ones((uni_modal_image_feats.size(0), uni_modal_image_feats.size(1)), dtype=torch.long,
                                 device=device)
        extended_image_masks = self.language_encoder.get_extended_attention_mask(image_masks, image_masks.size(),
                                                                                 device)
        # == End  : Image Encoding ==

        # == Begin: Image Text Contrastive ==
        if itc:
            with torch.no_grad():  # 表示在其作用域内，不进行梯度计算，节省内存提高推理速度
                self.temp.clamp_(0.001, 0.5)  # 对模型中的温度参数进行范围限制

            alpha = 0.4

            with torch.no_grad():
                self._momentum_update()  # 更新动量特征，确保当前模型的参数平滑地更新到一个目标模型（动量模型）上

                image_embeds_m = self.vision_encoder_m(img)  # 获得动量图像嵌入表示
                image_feat_m = F.normalize(self.vision_proj_m(image_embeds_m[:, 0, :]),
                                           dim=-1)  # 对动量图像的第一维进行投影、归一化，得到动量图像特征
                image_feat_all = torch.cat([image_feat_m.t(), self.image_queue.clone().detach()],
                                           dim=1)  # 按列拼接动量图像特征和图像队列，得到所有图像特征

                # text_output_m = self.language_encoder_m.bert(text_ids, attention_mask=text_masks,
                #                                          return_dict=True, mode='text')  # 获得动量文本嵌入表示
                text_output_m = self.language_encoder_m.embeddings(input_ids=text_ids)
                text_input_shape = text_masks.size()
                extended_text_masks = self.language_encoder.get_extended_attention_mask(text_masks, text_input_shape,
                                                                                        device)
                for layer in self.language_encoder.encoder.layer:
                    text_output_m = layer(text_output_m, extended_text_masks)[0]
                text_output_m = self.multi_modal_language_proj(text_output_m)
                text_feat_m = F.normalize(self.language_proj_m(text_output_m[:, 0, :]),
                                          dim=-1)  # 对动量文本的第一维进行投影、归一化，得到动量文本特征
                text_feat_all = torch.cat([text_feat_m.t(), self.text_queue.clone().detach()],
                                          dim=1)  # 按列拼接动量文本特征和文本队列，得到所有文本特征

                sim_i2t_m = image_feat_m @ text_feat_all / self.temp  # 计算动量图像和文本队列的相似度，通过点乘并除以温度参数
                sim_t2i_m = text_feat_m @ image_feat_all / self.temp  # 计算动量文本和图像队列的相似度

                sim_targets = torch.zeros(sim_i2t_m.size()).to(img.device)  # 创建一个与sim_i2t_m相同大小的零张量，作为相似度目标
                sim_targets.fill_diagonal_(1)  # 填充对角线元素为1，表示正样本

                sim_i2t_targets = alpha * F.softmax(sim_i2t_m, dim=1) + (
                        1 - alpha) * sim_targets  # 计算目标相似度，将动量图像对文本的相似度与目标相似度结合
                sim_t2i_targets = alpha * F.softmax(sim_t2i_m, dim=1) + (1 - alpha) * sim_targets  # 同样地计算动量文本对图像的目标相似度

            # 单模对比
            sim_i2i = image_feat @ image_feat_all / self.temp
            sim_t2t = text_feat @ text_feat_all / self.temp
            # 多模对比
            sim_i2t = image_feat @ text_feat_all / self.temp  # 计算当前的图像特征和所有文本特征之间的相似度
            sim_t2i = text_feat @ image_feat_all / self.temp  # 计算当前的文本特征和所有图像特征之间的相似度

            loss_i2i = -torch.sum(F.log_softmax(sim_i2i, dim=1) * sim_targets, dim=1).mean()
            loss_t2t = -torch.sum(F.log_softmax(sim_t2t, dim=1) * sim_targets, dim=1).mean()

            loss_i2t = -torch.sum(F.log_softmax(sim_i2t, dim=1) * sim_i2t_targets,
                                  dim=1).mean()  # 计算图像对文本的损失，使用交叉熵损失的形式
            loss_t2i = -torch.sum(F.log_softmax(sim_t2i, dim=1) * sim_t2i_targets, dim=1).mean()  # 计算文本对图像的损失


            loss_itc = (loss_i2t + loss_t2i) / 2  # 计算总损失，将两个损失取平均

            self._dequeue_and_enqueue(image_feat_m, text_feat_m)  # 将动量特征添加到图像和文本队列中，更新队列以保持固定大小
        else:
            loss_itc = 0.0
        # == End  : Image Text Contrastive ==

        # == Begin: Assign Type Embeddings ==
        # 对文本和图像特征进行调整，通过添加相应的模态类型嵌入，让模型能够知道输入数据来自哪个模态
        uni_modal_text_feats, uni_modal_image_feats = (
            uni_modal_text_feats + self.modality_type_embeddings(torch.zeros_like(text_masks)),
            uni_modal_image_feats + self.modality_type_embeddings(torch.full_like(image_masks, image_token_type_idx)),
        )
        # == End  : Assign Type Embeddings ==

        # == Begin: Multi-Modal Fusion ==
        ret["attentions"] = {"text2image_attns": [], "image2text_attns": []} if output_attentions else None
        x, y = uni_modal_text_feats, uni_modal_image_feats
        # == End  : Multi-Modal Fusion ==

        # == Begin: External Graph ==
        adj_feat = self.adj_feat.unsqueeze(0).expand(x.size(0), self.adj_feat.size(0),
                                                     self.adj_feat.size(1))  # 将adj_feat扩展到batch_size维度,[B,577,577]
        organ_disease_feat = self.organ_disease_feat.expand(x.size(0), self.organ_disease_feat.size(
            1))  # 将organ_disease_feat扩展到batch_size维度,[B,577]
        organ_disease_feat = self.language_encoder.embeddings.word_embeddings(
            organ_disease_feat)  # 将organ_disease_feat转换为embedding,[B,577,768]
        GAT_feat = self.GAT_layer(x, organ_disease_feat, adj_feat)  # LP和organ作为节点特征，adj作为邻接矩阵输入到图注意力网络层
        GAT_feat = self.organ_average(GAT_feat)  # 对GAT进行平均池化,[B,4,768]
        # == End  : External Graph ==

        # == Begin: Q-Former Alignment ==
        y, loss_itc = self.qformer(x, y)
        # y_cls_token = y[:, 0:1, :]
        # y = self.avg_pool_proj(y[:, 1:, :])
        # y = torch.cat([y_cls_token, y], dim=1)

        # == Begin: Mamba Multi-Modal Projection ==
        config = MambaConfig(d_model=x.size(-1), n_layers=2)
        MambaModule = Mamba(config).to(device)
        x1, y = MambaModule(x, y)
        # == End  : Mamba Multi-Modal Projection ==

        # == Begin: Text-Graph Mamba Fusion ==
        x2, k = MambaModule(x, GAT_feat)
        # == End  : Text-Graph Mamba Fusion ==

        multi_modal_text_feat_i, multi_modal_text_feat_k, multi_modal_image_feats, knowledge_feat = x1, x2, y, k

        multi_modal_text_feats = multi_modal_text_feat_i * multi_modal_text_feat_k
        multi_modal_text_cls_feats = self.multi_modal_language_pooler(multi_modal_text_feats)

        if self.is_clip:
            multi_modal_image_cls_feats = self.multi_modal_vision_pooler(y)
        else:
            avg_image_feats = self.vision_pooler(multi_modal_image_feats.transpose(1, 2)).view(
                multi_modal_image_feats.size(0), 1, -1)
            multi_modal_image_cls_feats = self.multi_modal_vision_pooler(avg_image_feats)
        # multi_modal_text_cls_feats, multi_modal_image_cls_feats = x, y
        # multi_modal_text_cls_feats = multi_modal_text_cls_feat_i * multi_modal_text_cls_feat_k
        multi_modal_cls_feats = torch.cat([multi_modal_text_cls_feats, multi_modal_image_cls_feats], dim=-1)  # 特征拼接

        # == Begin: Answer Generation Loss ==
        multi_modal_feats = torch.cat([multi_modal_text_feats, multi_modal_image_feats], dim=1)

        # multi_modal_open_feats, open_answer_targets = self.seperate(multi_modal_feats, answer_targets, batch["answer_types"])
        indexs_open = []
        for i in range(len(batch["answer_types"])):  # 分离close、open问题索引
            if batch["answer_types"][i] == 1:
                indexs_open.append(i)

        multi_modal_open_feats = multi_modal_feats[indexs_open, :]
        open_answer_targets = answer_targets[indexs_open, :]

        attention_masks = torch.ones((multi_modal_open_feats.size(0), multi_modal_open_feats.size(1)), device=device)
        # attention_masks = torch.ones((multi_modal_cls_feats.size(0), multi_modal_cls_feats.size(1)), dtype=torch.long,
        #                          device=device)

        if len(indexs_open) == 0:
            ag_loss = 0.0
        else:
            # vocab_size = self.language_decoder.config.vocab_size
            # if (open_answer_targets < 0).any() or (open_answer_targets >= vocab_size).any():
            #     print("open_answer_targets contains out-of-range values")
            #     ag_loss = 0.0
            # 检查输入嵌入的数值
            if torch.isnan(multi_modal_open_feats).any() or torch.isinf(multi_modal_open_feats).any():
                print("multi_modal_open_feats contains NaN or Inf values")
                ag_loss = 0.0
            # 检查目标标签的数值
            if torch.isnan(open_answer_targets).any() or torch.isinf(open_answer_targets).any():
                print("open_answer_targets contains NaN or Inf values")
                ag_loss = 0.0
            # 检查注意力掩码的形状和值
            if attention_masks.shape != (multi_modal_open_feats.size(0), multi_modal_open_feats.size(1)):
                print("attention_masks shape is incorrect")
                ag_loss = 0.0
            if not ((attention_masks == 0) | (attention_masks == 1)).all():
                print("attention_masks contains values other than 0 or 1")
                ag_loss = 0.0
            else:
                multi_modal_open_feats = multi_modal_open_feats.float()
                open_answer_targets = open_answer_targets.long()

                answer_output = self.language_decoder(inputs_embeds=multi_modal_open_feats,
                                                      attention_mask=attention_masks,
                                                      # encoder_hidden_states=uni_modal_text_feats,
                                                      # encoder_attention_mask=extended_text_masks,
                                                      labels=open_answer_targets,
                                                      return_dict=True,
                                                      )
                ag_loss = answer_output.loss

                # 梯度裁剪
                clip_grad_norm_(self.language_decoder.parameters(), max_norm=1.0)
        # ag_loss = ag_loss.sum() / img.size(0)
        # ag_loss = 0.0
        # == End  : Answer Generation Loss ==

        ret.update({
            "images": img,
            "patched_images": self.patchify(img),
            "text_labels": text_labels,
            "text_ids": text_ids,
            "text_masks": text_masks,
            "extended_image_masks": extended_image_masks,
            "extended_text_masks": extended_text_masks,
            "multi_modal_text_feats": multi_modal_text_feats,
            "multi_modal_image_feats": multi_modal_image_feats,
            "multi_modal_cls_feats": multi_modal_cls_feats,
            "itc_loss": loss_itc,
            "ag_loss": ag_loss
        })

        return ret

    # def get_retrieval_text(self, input):
    #     self.retrieval_tokenizer = RobertaTokenizerFast.from_pretrained('downloaded/roberta-base')
    #     encoding = self.retrieval_tokenizer(
    #         input,
    #         padding="max_length",
    #         truncation=True,
    #         max_length=self.config["max_text_len"],
    #         return_special_tokens_mask=True,
    #         return_offsets_mapping=True,
    #     )
    #     return {
    #         "text": [input, encoding],
    #     }
    #
    # def convert_batch_encoding_structure(self, original_batch_encoding):
    #     """
    #     将包含多个样本编码的单个 BatchEncoding 转换为多个独立 BatchEncoding 的列表
    #
    #     参数:
    #         original_batch_encoding: 图一结构的 BatchEncoding，内部字段是二维列表（如 input_ids 是 [[样本1], [样本2], ...]）
    #
    #     返回:
    #         list[BatchEncoding]: 图二结构的列表，每个元素是单个样本的 BatchEncoding
    #     """
    #     # 提取所有字段的键（如 'input_ids', 'attention_mask' 等）
    #     keys = original_batch_encoding.keys()
    #     # 获取样本数量（从 input_ids 的长度判断，假设所有字段样本数量一致）
    #     num_samples = len(original_batch_encoding['input_ids'])
    #
    #     converted = []
    #     for i in range(num_samples):
    #         # 为每个样本构建单独的字典，提取该样本在各字段的值（一维列表）
    #         sample_dict = {}
    #         for key in keys:
    #             sample_dict[key] = original_batch_encoding[key][i]
    #         # 用样本字典创建新的 BatchEncoding
    #         converted_batch_encoding = BatchEncoding(sample_dict)
    #         converted.append(converted_batch_encoding)
    #
    #     return converted
    #
    # def retrieval_text_process(self, dict_batch, batch_size):
    #     txt_keys = "retrieval_text"  # 筛选出文本相关键
    #     retrieval_text_data = dict_batch[txt_keys]
    #     texts = retrieval_text_data[0]  # 文本内容列表
    #     flatten_encodings = retrieval_text_data[1]  # BatchEncoding 类型的编码数据
    #
    #     # Collator for Dataloaders
    #     collator = (
    #         DataCollatorForWholeWordMask
    #     )
    #
    #     mlm_collator = collator(tokenizer=self.retrieval_tokenizer, mlm=True, mlm_probability=0.15)
    #
    #     converted_flatten_encodings = self.convert_batch_encoding_structure(flatten_encodings)
    #
    #     flatten_mlms = mlm_collator(converted_flatten_encodings)  # 应用 MLM 整理器：生成掩码后的输入和标签
    #
    #     for i, txt_key in enumerate([txt_keys]):
    #         mlm_ids, mlm_labels = (
    #             flatten_mlms["input_ids"][batch_size * (i): batch_size * (i + 1)],
    #             flatten_mlms["labels"][batch_size * (i): batch_size * (i + 1)],
    #         )
    #
    #         input_ids = torch.zeros_like(mlm_ids)
    #         attention_mask = torch.zeros_like(mlm_ids)
    #         for _i, encoding in enumerate(converted_flatten_encodings):
    #             _input_ids, _attention_mask = (
    #                 torch.tensor(encoding["input_ids"]),
    #                 torch.tensor(encoding["attention_mask"]),
    #             )
    #             input_ids[_i, : len(_input_ids)] = _input_ids
    #             attention_mask[_i, : len(_attention_mask)] = _attention_mask
    #
    #         dict_batch[txt_key] = texts
    #         dict_batch[f"{txt_key}_ids"] = input_ids
    #         dict_batch[f"{txt_key}_labels"] = torch.full_like(input_ids, -100)
    #         dict_batch[f"{txt_key}_ids_mlm"] = mlm_ids
    #         dict_batch[f"{txt_key}_labels_mlm"] = mlm_labels
    #         dict_batch[f"{txt_key}_masks"] = attention_mask
    #
    #     return dict_batch


    def forward(self, batch, test=False):
        ret = dict()

        if len(self.current_tasks) == 0:
            ret.update(self.infer(batch))
            return ret

        # Pre-Training: Masked Language Modeling
        if "mlm" in self.current_tasks:
            ret.update(objectives.compute_mlm(self, batch))

        # Pre-Training: Masked Image Modeling
        if "mim" in self.current_tasks:
            ret.update(objectives.compute_mim(self, batch))

        # Pre-Training: Image Text Matching
        if "itm" in self.current_tasks:
            ret.update(objectives.compute_itm(self, batch))

        # Fine-Tuning: Visual Question Answering
        if "vqa" in self.current_tasks:
            ret.update(objectives.compute_vqa(self, batch, test=test))

        # Fine-Tuning: Image-Text Classification
        if "cls" in self.current_tasks:
            ret.update(objectives.compute_cls(self, batch, test=test))

        # Fine-Tuning: Image Retrieval and Text Retrieval
        if "irtr" in self.current_tasks:
            ret.update(objectives.compute_irtr(self, batch, test))

        return ret

    def training_step(self, batch, batch_idx):
        m3ae_utils.set_task(self)
        # if self.retrieval_function:
        #     retrieval_info = self.retrieval_function(batch)
        #     input_sentences = [batch['text'][i] + retrieval_info[i] for i in range(len(batch['text']))]
        #     retrieval_text = self.get_retrieval_text(input_sentences)['text']
        #     batch["retrieval_text"] = retrieval_text
        #     batch = self.retrieval_text_process(batch, len(batch['qid']))
        output = self(batch)
        total_loss = sum([v * self.hparams.config["loss_names"][k.replace("_loss", "")]
                          for k, v in output.items() if "loss" in k])

        return total_loss

    def training_epoch_end(self, outs):
        m3ae_utils.epoch_wrapup(self)

    def validation_step(self, batch, batch_idx):
        m3ae_utils.set_task(self)
        # if self.retrieval_function:
        #     retrieval_info = self.retrieval_function(batch)
        #     input_sentences = [batch['text'][i] + retrieval_info[i] for i in range(len(batch['text']))]
        #     retrieval_text = self.get_retrieval_text(input_sentences)['text']
        #     batch["retrieval_text"] = retrieval_text
        #     batch = self.retrieval_text_process(batch, len(batch['qid']))
        output = self(batch)

    def validation_epoch_end(self, outs):
        m3ae_utils.epoch_wrapup(self)

    def test_step(self, batch, batch_idx):
        m3ae_utils.set_task(self)
        # if self.retrieval_function:
        #     retrieval_info = self.retrieval_function(batch)
        #     input_sentences = [batch['text'][i] + retrieval_info[i] for i in range(len(batch['text']))]
        #     retrieval_text = self.get_retrieval_text(input_sentences)['text']
        #     batch["retrieval_text"] = retrieval_text
        #     batch = self.retrieval_text_process(batch, len(batch['qid']))
        output = self(batch, test=True)

    def test_epoch_end(self, outs):
        m3ae_utils.epoch_wrapup(self, test=True)

    def configure_optimizers(self):
        return m3ae_utils.set_schedule(self)

    @torch.no_grad()
    def copy_params(self):
        for model_pair in self.model_pairs:
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data.copy_(param.data)  # initialize
                param_m.requires_grad = False  # not update by gradient

    @torch.no_grad()
    def _momentum_update(self):
        for model_pair in self.model_pairs:
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data = param_m.data * self.momentum + param.data * (1. - self.momentum)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, image_feat, text_feat):
        # gather keys before updating queue
        if is_dist_avail_and_initialized():
            image_feats = concat_all_gather(image_feat)
            text_feats = concat_all_gather(text_feat)
        else:
            image_feats = image_feat
            text_feats = text_feat

        batch_size = image_feats.shape[0]

        ptr = int(self.queue_ptr)
        assert self.queue_size % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.image_queue[:, ptr:ptr + batch_size] = image_feats.T
        self.text_queue[:, ptr:ptr + batch_size] = text_feats.T
        ptr = (ptr + batch_size) % self.queue_size  # move pointer

        self.queue_ptr[0] = ptr

@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
                      for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output

def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True

class GatedMultimodalLayer(nn.Module):
    def __init__(self, size_in1, size_in2, size_out):
        super().__init__()
        self.size_in1, self.size_in2, self.size_out = size_in1, size_in2, size_out

        weights_hidden1 = torch.Tensor(size_out, size_in1)
        self.weights_hidden1 = nn.Parameter(weights_hidden1)

        weights_hidden2 = torch.Tensor(size_out, size_in2)
        self.weights_hidden2 = nn.Parameter(weights_hidden2)

        weight_sigmoid = torch.Tensor(size_out * 2)
        self.weight_sigmoid = nn.Parameter(weight_sigmoid)

        nn.init.kaiming_uniform_(self.weights_hidden1, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.weights_hidden2, a=math.sqrt(5))
        nn.init.normal_(self.weight_sigmoid)

        self.tanh_f = nn.Tanh()
        self.sigmoid_f = nn.Sigmoid()

    def forward(self, x1, x2):
        h1 = self.tanh_f(torch.mm(x1, self.weights_hidden1.t()))
        h2 = self.tanh_f(torch.mm(x2, self.weights_hidden2.t()))
        x = torch.cat((h1, h2), dim=1)
        z = self.sigmoid_f(torch.matmul(x, self.weight_sigmoid.t()))

        return z.view(z.size()[0], 1) * h1 + (1 - z).view(z.size()[0], 1) * h2