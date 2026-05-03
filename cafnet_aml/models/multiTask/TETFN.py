"""
Paper: TETFN: A text enhanced transformer fusion network for multimodal sentiment analysis
该模型是一种文本增强的Transformer融合网络，用于多模态情感分析，结合文本、音频、视觉模态特征，通过文本增强的 Transformer 进行跨模态融合
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
# 导入 Transformer 编码器模块 用于构建模型中的 Transformer 结构
from ..subNets.transformers_encoder.transformer import TransformerEncoder
# 导入 Bert 文本编码器模块 用于处理文本模态特征
from ..subNets import BertTextEncoder

# 对外暴露的类 仅 TETFN 可被外部 import
__all__ = ['TETFN']

class TETFN(nn.Module):
    def __init__(self, args):
        """
        初始化 TETFN 模型
        Args:
            args: 配置参数字典，包含模型各部分的参数设置，如特征维度、dropout率、网络层数等
        """
        super(TETFN, self).__init__()
        # 保存配置参数
        self.args = args
        # 标记是否需要数据对齐
        self.aligned = args.need_data_aligned
        # 初始化文本编码器（基于 Bert） 用于提取文本特征 对应架构图中 “Text” 分支的 Bert 部分
        self.text_model = BertTextEncoder(use_finetune=args.use_finetune, transformers=args.transformers, pretrained=args.pretrained)

        # 音频和视觉子网络相关参数 从配置中获取各模态输入特征维度
        text_in, audio_in, video_in = args.feature_dims

        # 初始化音频子网络 处理音频模态特征 对应架构图中 “Acoustic” 分支的 Conv1D、LSTM 等部分
        self.audio_model = AuViSubNet(
            audio_in, 
            args.a_lstm_hidden_size, 
            args.conv1d_kernel_size_a,
            args.dst_feature_dims,
            num_layers=args.a_lstm_layers, dropout=args.a_lstm_dropout)
        
        # 初始化视觉子网络 处理视觉模态特征 对应架构图中 “Visual” 分支的 Conv1D、LSTM 等部分
        self.video_model = AuViSubNet(
            video_in, 
            args.v_lstm_hidden_size, 
            args.conv1d_kernel_size_a,
            args.dst_feature_dims,
            num_layers=args.v_lstm_layers, dropout=args.v_lstm_dropout)
        
        # 文本特征投影层 将文本特征投影到目标特征维度
        self.proj_l = nn.Conv1d(text_in, args.dst_feature_dims, kernel_size=args.conv1d_kernel_size_l, padding=0, bias=False)
        
        # 初始化各 Transformer 融合网络 用于不同模态间的交叉融合 对应架构图中的 “Cross-modal Layer”
        self.trans_l_with_a = self.get_network(self_type='la')  # 文本与音频交叉的 Transformer
        self.trans_l_with_v = self.get_network(self_type='lv')  # 文本与视觉交叉的 Transformer
        self.trans_a_with_l = self.get_network(self_type='al')  # 音频与文本交叉的 Transformer

        # 文本增强的音频与视觉交叉 Transformer 对应架构中“文本增强转换器(TET)”部分 实现文本对音频-视觉融合的增强
        self.trans_a_with_v = TextEnhancedTransformer(
            embed_dim=args.dst_feature_dims,
            num_heads=args.nheads, 
            layers=2, attn_dropout=args.attn_dropout,relu_dropout=args.relu_dropout,res_dropout=args.res_dropout,embed_dropout=args.embed_dropout)
    
        self.trans_v_with_l = self.get_network(self_type='vl')  # 视觉与文本交叉的 Transformer 对应“Cross-modal Layer”
        
        # 文本增强的视觉与音频交叉 Transformer 同样属于“文本增强转换器(TET)”，增强文本对视觉-音频融合的作用
        self.trans_v_with_a = TextEnhancedTransformer(
            embed_dim=args.dst_feature_dims,
            num_heads=args.nheads, 
            layers=2, attn_dropout=args.attn_dropout,relu_dropout=args.relu_dropout,res_dropout=args.res_dropout,embed_dropout=args.embed_dropout)
        
        # 各模态记忆 Transformer 用于整合模态内信息 对应架构中各模态内部的特征处理部分
        self.trans_l_mem = self.get_network(self_type='l_mem', layers=2)  # 文本记忆Transformer
        self.trans_a_mem = self.get_network(self_type='a_mem', layers=2)  # 音频记忆Transformer
        self.trans_v_mem = self.get_network(self_type='v_mem', layers=2)  # 视觉记忆Transformer

        # 多模态融合后的分类层 对应架构中“Prediction”模块的多模态分支
        self.post_fusion_dropout = nn.Dropout(p=args.post_fusion_dropout)
        self.post_fusion_layer_1 = nn.Linear(6 * args.dst_feature_dims, args.post_fusion_dim)
        self.post_fusion_layer_2 = nn.Linear(args.post_fusion_dim, args.post_fusion_dim)
        self.post_fusion_layer_3 = nn.Linear(args.post_fusion_dim, 1)

        # 文本单模态分类层 对应架构中“Prediction”模块的文本分支 属于单峰标签生成模块(ULGM)的一部分
        self.post_text_dropout = nn.Dropout(p=args.post_text_dropout)
        self.post_text_layer_1 = nn.Linear(args.dst_feature_dims, args.post_text_dim)
        self.post_text_layer_2 = nn.Linear(args.post_text_dim, args.post_text_dim)
        self.post_text_layer_3 = nn.Linear(args.post_text_dim, 1)

        # 音频单模态分类层 对应架构中“Prediction”模块的音频分支 属于单峰标签生成模块(ULGM)的一部分
        self.post_audio_dropout = nn.Dropout(p=args.post_audio_dropout)
        self.post_audio_layer_1 = nn.Linear(args.dst_feature_dims, args.post_audio_dim)
        self.post_audio_layer_2 = nn.Linear(args.post_audio_dim, args.post_audio_dim)
        self.post_audio_layer_3 = nn.Linear(args.post_audio_dim, 1)

        # 视觉单模态分类层 对应架构中“Prediction”模块的视觉分支 属于单峰标签生成模块(ULGM)的一部分
        self.post_video_dropout = nn.Dropout(p=args.post_video_dropout)
        self.post_video_layer_1 = nn.Linear(args.dst_feature_dims, args.post_video_dim)
        self.post_video_layer_2 = nn.Linear(args.post_video_dim, args.post_video_dim)
        self.post_video_layer_3 = nn.Linear(args.post_video_dim, 1)

    def get_network(self, self_type='l', layers=-1):
        """
        获取不同类型的 Transformer 网络 根据模态类型设置不同的嵌入维度和注意力 dropout 等参数
        Args:
            self_type: 网络类型，区分不同模态组合
            layers: 网络层数
        Returns:
            构建好的 TransformerEncoder 实例
        """
        if self_type in ['l', 'al', 'vl']:
            embed_dim, attn_dropout = self.args.dst_feature_dims, self.args.attn_dropout
        elif self_type in ['a', 'la', 'va']:
            embed_dim, attn_dropout = self.args.dst_feature_dims, self.args.attn_dropout_a
        elif self_type in ['v', 'lv', 'av']:
            embed_dim, attn_dropout = self.args.dst_feature_dims, self.args.attn_dropout_v
        elif self_type == 'l_mem':
            embed_dim, attn_dropout = 2*self.args.dst_feature_dims, self.args.attn_dropout
        elif self_type == 'a_mem':
            embed_dim, attn_dropout = 2*self.args.dst_feature_dims, self.args.attn_dropout
        elif self_type == 'v_mem':
            embed_dim, attn_dropout = 2*self.args.dst_feature_dims, self.args.attn_dropout
        else:
            raise ValueError("Unknown network type")
        
        return TransformerEncoder(embed_dim=embed_dim,
                                  num_heads=self.args.nheads,
                                  layers=2,
                                  attn_dropout=attn_dropout,
                                  relu_dropout=self.args.relu_dropout,
                                  res_dropout=self.args.res_dropout,
                                  embed_dropout=self.args.embed_dropout,
                                  attn_mask=True)

    def forward(self, text, audio, video):
        """
        模型前向传播过程
        Args:
            text: 文本模态输入
            audio: 音频模态输入
            video: 视觉模态输入
        Returns:
            res: 包含多模态及各单模态预测结果、中间特征的字典
        """
        # 处理音频和视觉的长度信息
        audio, audio_lengths = audio
        video, video_lengths = video

        # 计算文本有效长度（基于 Bert 的 mask）
        mask_len = torch.sum(text[:,1,:], dim=1, keepdim=True)
        text_lengths = mask_len.squeeze(1).int().detach().cpu()

        # 文本特征提取 对应架构中 “Text” 分支的 Bert 等处理部分
        text = self.text_model(text)

        # 音频和视觉特征提取 根据是否需要数据对齐 使用不同的长度信息 对应架构中“Acoustic”和“Visual”分支的 Conv1D、LSTM 等处理
        if self.aligned:
            audio = self.audio_model(audio, text_lengths)
            video = self.video_model(video, text_lengths)
        else:
            audio = self.audio_model(audio, audio_lengths)
            video = self.video_model(video, video_lengths)
        
        # 文本特征投影 统一到目标特征维度
        text = self.proj_l(text.transpose(1,2))
        # 调整各模态特征的维度顺序 适应 Transformer 输入
        proj_x_a = audio.permute(2, 0, 1)
        proj_x_v = video.permute(2, 0, 1)
        proj_x_l = text.permute(2, 0, 1)
        
        # 提取各模态的全局特征（取最大值）
        text_h = torch.max(proj_x_l, dim=0)[0]
        audio_h = torch.max(proj_x_a, dim=0)[0]
        video_h = torch.max(proj_x_v, dim=0)[0]
        
        # （V,A）→ L：音频、视觉特征向文本特征的交叉融合 对应架构中 “Cross-modal Layer” 等交叉融合部分
        h_l_with_as = self.trans_l_with_a(proj_x_l, proj_x_a, proj_x_a)    
        h_l_with_vs = self.trans_l_with_v(proj_x_l, proj_x_v, proj_x_v)    
        h_ls = torch.cat([h_l_with_as, h_l_with_vs], dim=2)
        # 文本记忆 Transformer 整合文本相关的交叉融合特征 对应模态内特征整合部分
        h_ls = self.trans_l_mem(h_ls)
        if type(h_ls) == tuple:
            h_ls = h_ls[0]
        last_h_l = h_ls[-1]   # 取最后一个输出用于预测

        # （L,V）→ A：文本、视觉特征向音频特征的交叉融合 对应 “Cross-modal Layer”
        h_a_with_ls = self.trans_a_with_l(proj_x_a, proj_x_l, proj_x_l)
        # 文本增强的音频与视觉交叉融合 对应“文本增强转换器(TET)”部分
        h_a_with_vs = self.trans_a_with_v(proj_x_v, proj_x_a, proj_x_l)
        h_as = torch.cat([h_a_with_ls, h_a_with_vs], dim=2)
        # 音频记忆Transformer 整合音频相关的交叉融合特征 对应模态内特征整合部分
        h_as = self.trans_a_mem(h_as)
        if type(h_as) == tuple:
            h_as = h_as[0]
        last_h_a = h_as[-1]
        
        # （L,A）→ V：文本、音频特征向视觉特征的交叉融合 对应 “Cross-modal Layer”
        h_v_with_ls = self.trans_v_with_l(proj_x_v, proj_x_l, proj_x_l)
        # 文本增强的视觉与音频交叉融合 对应“文本增强转换器(TET)”部分
        h_v_with_as = self.trans_v_with_a(proj_x_a, proj_x_v, proj_x_l)
        h_vs = torch.cat([h_v_with_ls, h_v_with_as], dim=2)
        # 视觉记忆 Transformer 整合视觉相关的交叉融合特征 对应模态内特征整合部分
        h_vs = self.trans_v_mem(h_vs)
        if type(h_vs) == tuple:
            h_vs = h_vs[0]
        last_h_v = h_vs[-1]

        # 多模态特征融合 对应架构中多模态特征融合后到“Prediction”的部分
        fusion_h = torch.cat([last_h_l, last_h_a, last_h_v], dim=-1)
        fusion_h = self.post_fusion_dropout(fusion_h)
        fusion_h = F.relu(self.post_fusion_layer_1(fusion_h), inplace=False)
        
        # 文本单模态特征处理 对应文本单模态“Prediction”前的特征处理 属于单峰标签生成模块(ULGM)流程
        text_h = self.post_text_dropout(text_h)
        text_h = F.relu(self.post_text_layer_1(text_h), inplace=False)
        # 音频单模态特征处理 对应音频单模态“Prediction”前的特征处理 属于单峰标签生成模块(ULGM)流程
        audio_h = self.post_audio_dropout(audio_h)
        audio_h = F.relu(self.post_audio_layer_1(audio_h), inplace=False)
        # 视觉单模态特征处理 对应视觉单模态“Prediction”前的特征处理 属于单峰标签生成模块(ULGM)流程
        video_h = self.post_video_dropout(video_h)
        video_h = F.relu(self.post_video_layer_1(video_h), inplace=False)

        # 多模态分类层前向计算 对应多模态“Prediction”
        x_f = F.relu(self.post_fusion_layer_2(fusion_h), inplace=False)
        output_fusion = self.post_fusion_layer_3(x_f)

        # 文本单模态分类层前向计算 对应文本“Prediction” 属于单峰标签生成模块(ULGM)
        x_t = F.relu(self.post_text_layer_2(text_h), inplace=False)
        output_text = self.post_text_layer_3(x_t)

        # 音频单模态分类层前向计算 对应音频“Prediction” 属于单峰标签生成模块(ULGM)
        x_a = F.relu(self.post_audio_layer_2(audio_h), inplace=False)
        output_audio = self.post_audio_layer_3(x_a)

        # 视觉单模态分类层前向计算 对应视觉“Prediction” 属于单峰标签生成模块(ULGM)
        x_v = F.relu(self.post_video_layer_2(video_h), inplace=False)
        output_video = self.post_video_layer_3(x_v)

        # 整理返回结果 包含多模态和各单模态预测结果、中间特征
        res = {
            'M': output_fusion, 
            'T': output_text,
            'A': output_audio,
            'V': output_video,
            'Feature_t': text_h,
            'Feature_a': audio_h,
            'Feature_v': video_h,
            'Feature_f': fusion_h,
        }
        return res

class TextEnhancedTransformer(nn.Module):
    """
    文本增强的 Transformer 类 实现文本对其他模态融合的增强 对应架构中的“文本增强转换器(TET)”
    """
    def __init__(self, embed_dim, num_heads, layers, attn_dropout, relu_dropout, res_dropout, embed_dropout) -> None:
        super().__init__()

        # 下层多头注意力 用于初步融合文本与其他模态特征
        self.lower_mha = TransformerEncoder(
            embed_dim=embed_dim,
            num_heads=num_heads,
            layers=1,
            attn_dropout=attn_dropout,
            relu_dropout=relu_dropout,
            res_dropout=res_dropout,
            embed_dropout=embed_dropout,
            position_embedding=True,
            attn_mask=True
        )

        # 上层多头注意力 进一步整合融合后的特征
        self.upper_mha = TransformerEncoder(
            embed_dim=embed_dim,
            num_heads=num_heads,
            layers=layers,
            attn_dropout=attn_dropout,
            relu_dropout=relu_dropout,
            res_dropout=res_dropout,
            embed_dropout=embed_dropout,
            position_embedding=True,
            attn_mask=True
        )
    
    def forward(self, query_m, key_m, text):
        """
        前向传播 实现文本增强的跨模态融合
        Args:
            query_m: 查询模态特征
            key_m: 键模态特征
            text: 文本模态特征
        Returns:
            增强融合后的特征
        """
        c = self.lower_mha(query_m, text, text)
        return self.upper_mha(key_m, c, c)
    
class AuViSubNet(nn.Module):
    
    def __init__(self, in_size, hidden_size, conv1d_kernel_size, dst_feature_dims, num_layers=1, dropout=0.2, bidirectional=False):
        '''
        初始化音频和视觉子网络，用于处理音频或视觉模态的特征
        Args:
            in_size: 输入特征维度
            hidden_size: LSTM隐藏层维度
            num_layers: LSTM层数
            dropout: dropout概率
            bidirectional: 是否使用双向LSTM
        对应架构图部分：“Acoustic” 分支的 LSTM、Conv1D 等处理部分 以及 “Visual” 分支的 LSTM、Conv1D 等处理部分
        '''
        super(AuViSubNet, self).__init__()
        # 定义 LSTM 层 用于对输入序列进行上下文编码
        self.rnn = nn.LSTM(in_size, hidden_size, num_layers=num_layers, dropout=dropout, bidirectional=bidirectional, batch_first=True)
        # 定义 1D 卷积层 用于对 LSTM 输出特征进行进一步处理 调整特征维度等
        self.conv = nn.Conv1d(hidden_size, dst_feature_dims, kernel_size=conv1d_kernel_size, bias=False)
        

    def forward(self, x, lengths):
        '''
        前向传播函数，处理输入的音频或视觉特征
        Args:
            x: 输入特征，形状为(batch_size, sequence_len, in_size)
            lengths: 序列长度信息
        Returns:
            处理后的特征
        对应架构图部分：“Acoustic”分支中LSTM和Conv1D的前向计算流程，以及“Visual”分支中LSTM和Conv1D的前向计算流程
        '''
        # LSTM 前向计算 得到隐藏状态 h
        h, _ = self.rnn(x)
        # 调整维度后进行 1D 卷积操作
        h = self.conv(h.transpose(1,2))
        return h