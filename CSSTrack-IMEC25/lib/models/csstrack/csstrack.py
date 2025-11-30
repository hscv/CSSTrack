"""
Basic CSSTrack model.
"""
import math
import os
from typing import List

import torch
from torch import nn
from torch.nn.modules.transformer import _get_clones

from lib.models.layers.head import build_box_head
from lib.models.csstrack.hivit import hivit_small, hivit_base
from lib.utils.box_ops import box_xyxy_to_cxcywh

from lib.models.layers.transformer_dec import build_transformer_dec
from lib.models.layers.position_encoding import build_position_encoding
from lib.utils.misc import NestedTensor
import torch.nn.functional as F


import queue

band_number = 25
band_split_group = 4

import torch
import torch.nn as nn



class Band_selection_attn(nn.Module, ):
    def __init__(self, inplanes=32, hide_channel=8, smooth=False):
        super(Band_selection_attn, self).__init__()
        self.mhsa = nn.MultiheadAttention(embed_dim=32, num_heads=8, batch_first=True)
        self.relu = nn.ReLU(inplace=False)
        self.temperature = 0.5
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x_template, x_search): 
        """ Forward pass with input x. """
        x_template = x_template.view(-1, 32, band_number)
        x_template = x_template.permute(0, 2, 1)

        x_search = x_search.view(-1, 32, band_number)
        x_search = x_search.permute(0, 2, 1) 

        attn_output, attn_weights = self.mhsa(
            query=x_search,
            key=x_template,
            value=x_template,
        )
        attn_output = self.relu(attn_output)
        w0 = attn_output
        w0_T = w0.transpose(1, 2)
        cm = torch.matmul(w0, w0_T)
        cm = cm / cm.max()
        for i in range(band_number):
            cm[:, i, i] = 0.0
        y = cm.mean(dim=2)
        orderY = torch.sort(y, dim=-1, descending=True, out=None)
        return orderY, cm

class BandSelection(nn.Module):
    def __init__(self, ):
        super().__init__()
        self.attn = Band_selection_attn(inplanes=32, hide_channel=8)
        self.proj = nn.Conv2d(band_number, 32*band_number, kernel_size=1)
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=1)
        self.maxpool = nn.AdaptiveMaxPool2d(output_size=1)

    def _split_Channel(self,feat_channel,order):
        res = []
        b = feat_channel.size()[0]
        for i in range(band_split_group):
            gg = feat_channel[None,0,order[0,i*3:i*3+3],:,:]
            for k in range(1,b):
                gg = torch.cat((gg,feat_channel[None,k,order[k,i*3:i*3+3],:,:]),dim=0)
            res.append(gg)
        return res

    def forward(self, template_input_features, search_input_features):
        b, c, _, _ = template_input_features.size()
        res_template = self.proj(template_input_features)
        res_search = self.proj(search_input_features)
        res_template = self.avgpool(res_template)
        res_search = self.avgpool(res_search)
        orderY, cm = self.attn(res_template, res_search)
        tx_search = search_input_features.view(b,c,-1)
        wx_search = torch.bmm(cm, tx_search)
        tx_template = template_input_features.view(b,c,-1)
        wx_template = torch.bmm(cm, tx_template)
        cg_res = [tx_template, tx_search, wx_template, wx_search]
        order = orderY[1]
        template_arr = self._split_Channel(template_input_features, order)

        order = orderY[1]
        search_arr = self._split_Channel(search_input_features, order)
        return template_arr, search_arr, orderY, cg_res

class SSTAttentionFusion(nn.Module):
    def __init__(self, dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.dim = dim
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, dropout=dropout)
        self.norm = nn.LayerNorm(dim)
        self.proj = nn.Conv2d(band_split_group * dim, dim, kernel_size=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, spe_spa_features, temporal_features, feat_len_s=256, feat_sz_s=32):
        enc_opt_arr = []
        for k in range(((spe_spa_features.size()[1])//320)):
            enc_opt_arr.append(spe_spa_features[:, 320*(k+1)-feat_len_s:320*(k+1)])

        enc_opt = torch.cat(enc_opt_arr, dim=1).transpose(0,1)
        N_all, B, C = enc_opt.size()
        num_spec = band_split_group
        N = N_all // num_spec
        dec_opt = temporal_features.transpose(0,1)
        dec_opt = dec_opt.view(-1, B, C)

        attn_output, _ = self.attn(query=enc_opt, key=dec_opt, value=dec_opt)
        attn_output = attn_output.permute(1, 0, 2)
        attn_output = attn_output.view(B, num_spec, N, C)
        spec_input = torch.stack(enc_opt_arr, dim=1)
        attn_output = attn_output + spec_input

        attn_output = self.norm(attn_output.view(B, -1, C)).view(B, num_spec, N, C)
        attn_output = attn_output.permute(0, 1, 3, 2)


        attn_output = attn_output.contiguous().view(B, num_spec*512, N)
        attn_output = attn_output.contiguous().view(B, num_spec*512, feat_sz_s, feat_sz_s)

        final_output = self.proj(attn_output)
        final_output = final_output.view(B, C, -1)
        final_output = final_output.unsqueeze(1)
        return final_output



class CSSTrack(nn.Module):
    """ This is the base class for CSSTrack """

    def __init__(self, transformer, box_head, transformer_dec, position_encoding, aux_loss=False, head_type="CORNER"):
        """ Initializes the model.
        Parameters:
            transformer: torch module of the transformer architecture.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.backbone = transformer
        self.box_head = box_head

        self.aux_loss = aux_loss
        self.head_type = head_type
        if head_type == "CORNER" or head_type == "CENTER":
            self.feat_sz_s = int(box_head.feat_sz)
            self.feat_len_s = int(box_head.feat_sz ** 2)

        if self.aux_loss:
            self.box_head = _get_clones(self.box_head, 6)
        
        self.transformer_dec = transformer_dec
        self.position_encoding = position_encoding 
        self.query_embed=nn.Embedding(num_embeddings=1, embedding_dim=512)
        self.query_embed_new=nn.Embedding(num_embeddings=band_split_group, embedding_dim=512)
        self.sstAttnFusion = SSTAttentionFusion(512)
        self.bandSelection = BandSelection()


    def forward(self, template: torch.Tensor,
                search: torch.Tensor,
                return_last_attn=False,
                training=True, #True
                tgt_pre = None,
                ):
        b0, num_search = template[0].shape[0], len(search)
        if training:
            search = torch.cat(search, dim=0)
            template = template[0].repeat(num_search,1,1,1)
        template_arr, search_arr, orderY, cg_res = self.bandSelection(template, search)
        x_arr = []
        for kk in range(band_split_group):
            x, aux_dict = self.backbone(z=template_arr[kk], x=search_arr[kk],
                                        return_last_attn=return_last_attn, )
            x_arr.append(x)
        x = torch.cat((x_arr), dim=1)

        b,n,c = x.shape
        input_dec = x
        batches = [[] for _ in range(b0)]
        for i, input in enumerate(input_dec):
            batches[i % b0].append(input.unsqueeze(0))
        x_decs = []
        query_embed_new = self.query_embed_new.weight
        assert len(query_embed_new.size()) in [2, 3]
        if len(query_embed_new.size()) == 2:
            query_embeding = query_embed_new.unsqueeze(0)
        for i,batch in enumerate(batches):
            if len(batch) ==0:
                continue
            tgt_all = [torch.zeros_like(query_embeding) for _ in range(num_search)]

            for j, input in enumerate(batch):
                pos_embed = self.position_encoding(1)
                tgt_q = tgt_all[j]
                tgt_kv = torch.cat(tgt_all[:j+1], dim=0)
                if not training and len(tgt_pre) != 0:
                    tgt_kv = torch.cat(tgt_pre, dim=0)
                tgt = [tgt_q, tgt_kv]
                tgt_out = self.transformer_dec(input.transpose(0, 1), tgt, self.feat_len_s, pos_embed, query_embeding)
                x_decs.append(tgt_out[0])
                tgt_all[j] = tgt_out[0]
            if not training:
                if len(tgt_pre) < 3: #num_search-1 
                    tgt_pre.append(tgt_out[0])
                else:
                    tgt_pre.pop(0)
                    tgt_pre.append(tgt_out[0])
            
        batch0 =[]
        if not training:
            batch0.append(x_decs[0])
        else:
            batch0 = [x_decs[i + j*num_search]  for i in range(num_search) for j in range(b0)]
        
        x_dec = torch.cat(batch0, dim = 1)

        # Forward head
        feat_last = x
        if isinstance(x, list):
            feat_last = x[-1]
        opt = self.sstAttnFusion(feat_last, x_dec, feat_len_s=self.feat_len_s, feat_sz_s=self.feat_sz_s)
        out = self.forward_head(opt, None) # STM and head
        
        out.update(aux_dict)
        out['tgt'] = tgt_pre 
        out['cg_res'] = cg_res
        out['bs_order'] = orderY
        return out

    # def forward_head(self, cat_feature, out_dec=None, gt_score_map=None):
    def forward_head(self, opt, gt_score_map=None):
        """
        cat_feature: output embeddings of the backbone, it can be (HW1+HW2, B, C) or (HW2, B, C)
        """
        
        bs, Nq, C, HW = opt.size()
        opt_feat = opt.view(-1, C, self.feat_sz_s, self.feat_sz_s)

        #Head
        if self.head_type == "CORNER":
            # run the corner head
            pred_box, score_map = self.box_head(opt_feat, True)
            outputs_coord = box_xyxy_to_cxcywh(pred_box)
            outputs_coord_new = outputs_coord.view(bs, Nq, 4)
            out = {'pred_boxes': outputs_coord_new,
                   'score_map': score_map,
                   }
            return out

        elif self.head_type == "CENTER":
            # run the center head
            score_map_ctr, bbox, size_map, offset_map = self.box_head(opt_feat, gt_score_map)
            # outputs_coord = box_xyxy_to_cxcywh(bbox)
            outputs_coord = bbox
            outputs_coord_new = outputs_coord.view(bs, Nq, 4)
            out = {'pred_boxes': outputs_coord_new,
                   'score_map': score_map_ctr,
                   'size_map': size_map,
                   'offset_map': offset_map}
            return out
        else:
            raise NotImplementedError


def build_csstrack(cfg, training=True):
    current_dir = os.path.dirname(os.path.abspath(__file__))  # This is your Project Root
    pretrained_path = os.path.join(current_dir, '../../../pretrained_models')
    if cfg.MODEL.PRETRAIN_FILE and ('CSSTrack' not in cfg.MODEL.PRETRAIN_FILE) and training:
        pretrained = os.path.join(pretrained_path, cfg.MODEL.PRETRAIN_FILE)
    else:
        pretrained = ''

    print ('cfg.MODEL.BACKBONE.TYPE = ', cfg.MODEL.BACKBONE.TYPE)
    if cfg.MODEL.BACKBONE.TYPE == 'hivit_small':
        backbone = hivit_small(pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE)
        hidden_dim = backbone.embed_dim
        patch_start_index = 1

    elif cfg.MODEL.BACKBONE.TYPE == 'hivit_base':
        backbone = hivit_base(pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE)
        hidden_dim = backbone.embed_dim
        patch_start_index = 1

    else:
        raise NotImplementedError

    backbone.finetune_track(cfg=cfg, patch_start_index=patch_start_index)

    
    transformer_dec = build_transformer_dec(cfg, hidden_dim)
    position_encoding = build_position_encoding(cfg, sz = 1)

    box_head = build_box_head(cfg, hidden_dim)
    model = CSSTrack(
        backbone,
        box_head,
        transformer_dec,
        position_encoding,        
        aux_loss=False,
        head_type=cfg.MODEL.HEAD.TYPE,
    )

    if 'CSSTrack' in cfg.MODEL.PRETRAIN_FILE and training:
        checkpoint = torch.load(cfg.MODEL.PRETRAIN_FILE, map_location="cpu")
        missing_keys, unexpected_keys = model.load_state_dict(checkpoint["net"], strict=False)
        print('Load pretrained model from: ' + cfg.MODEL.PRETRAIN_FILE)

    return model
