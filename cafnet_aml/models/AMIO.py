import torch.nn as nn

# 导入不同任务类型的模型（单任务、多任务、缺失模态任务）
from .multiTask import *  # 多任务模型（如 MTFN、MLMF 等）
from .singleTask import *  # 单任务模型（如 TFN、LMF 等）
from .missingTask import *  # 缺失模态任务模型（如 TFR_NET）
from .NewTask import *  # 新任务模型（如 CAMFN）
# 导入子网络：用于模态序列对齐的子网络
from .subNets import AlignSubNet
# 导入 BERT 配置工具（用于初始化基于 BERT 的模型）
from transformers import BertConfig

class AMIO(nn.Module):
    """
    多模型统一封装类，继承自 PyTorch 的 nn.Module
    支持自动加载不同类型的模型 并处理模态序列对齐逻辑
    """
    def __init__(self, args):
        """
        初始化 AMIO 实例 根据配置参数加载指定模型
        Args:
            args: 配置参数字典（包含model_name、是否需要对齐等关键参数）
        """
        super(AMIO, self).__init__()  # 调用父类 nn.Module 的初始化方法

        # 模型映射表：key 为模型名称字符串 value 为对应的模型类
        # 用于根据 args['model_name'] 自动选择并实例化模型
        self.MODEL_MAP = {
            # 单任务模型（仅处理多模态融合的主任务）
            'tfn': TFN,
            'lmf': LMF,
            'mfn': MFN,
            'ef_lstm': EF_LSTM,
            'lf_dnn': LF_DNN,
            'graph_mfn': Graph_MFN,
            'mctn': MCTN,
            'bert_mag': BERT_MAG,
            'mult': MULT,
            'misa': MISA,
            'mfm': MFM,
            'mmim': MMIM,
            'cenet': CENET,
            'almt': ALMT,
            # 多任务模型（同时处理主任务和辅助任务）
            'mtfn': MTFN,
            'mlmf': MLMF,
            'mlf_dnn': MLF_DNN,
            'self_mm': SELF_MM,
            'tetfn': TETFN,
            # 缺失模态任务模型（处理模态缺失场景）
            'tfr_net': TFR_NET,
            # 新任务模型（冲突感知多模态融合）
            'camfn': CAMFN,
            'camfnet_aml': CAMFN
        }

        # 记录模型是否需要模态序列对齐（从配置参数获取）
        self.need_model_aligned = args.get('need_model_aligned', None)

        # 若模型需要模态对齐 则初始化对齐子网络
        # 对齐子网络用于将文本、音频、视觉的序列长度统一（如通过平均池化降维）
        if self.need_model_aligned:
            # 初始化对齐子网络 使用平均池化（avg_pool）方式对齐
            self.alignNet = AlignSubNet(args, 'avg_pool')
            # 若配置中有序列长度参数 更新为对齐后的长度
            if 'seq_lens' in args.keys():
                args['seq_lens'] = self.alignNet.get_seq_len()

        # 根据模型名称从映射表中获取目标模型类
        lastModel = self.MODEL_MAP[args['model_name']]

        # 特殊处理 CENET 模型（基于BERT，需要从预训练模型加载配置）
        if args.model_name == 'cenet':
            # 从预训练 BERT 加载配置 并指定输出类别数（回归任务为1）
            config = BertConfig.from_pretrained(
                args.pretrained,
                num_labels=1,
                finetuning_task='sst'  # 微调任务标识（情感分析相关）
            )
            # 实例化 CENET 模型，传入预训练配置和额外参数
            self.Model = CENET.from_pretrained(
                args.pretrained,
                config=config,
                pos_tag_embedding=True,  # 启用词性标签嵌入
                senti_embedding=True,   # 启用情感嵌入
                polarity_embedding=True,  # 启用极性嵌入
                args=args  # 传入配置参数
            )
        else:
            # 其他模型直接通过类实例化 传入配置参数 args
            self.Model = lastModel(args)

    def forward(self, text_x, audio_x, video_x, *args, **kwargs):
        """
        前向传播函数：统一处理输入，执行模态对齐（若需要），调用模型前向计算
        Args:
            text_x: 文本模态输入特征
            audio_x: 音频模态输入特征
            video_x: 视觉模态输入特征
            *args: 其他位置参数（传给具体模型的forward方法）
            **kwargs: 其他关键字参数（传给具体模型的forward方法）
        Returns:
            模型的输出结果（如情感预测值）
        """
        # 若需要模态对齐 先通过对齐子网络处理输入
        if self.need_model_aligned:
            text_x, audio_x, video_x = self.alignNet(text_x, audio_x, video_x)
        
        # 调用具体模型的前向传播方法 返回计算结果
        return self.Model(text_x, audio_x, video_x, *args, **kwargs)