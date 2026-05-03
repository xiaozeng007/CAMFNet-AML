# 导入不同任务类型的训练器（与模型对应）
from .multiTask import *  # 多任务模型的训练器
from .singleTask import *  # 单任务模型的训练器
from .missingTask import *  # 缺失模态任务模型的训练器
from .NewTask import *  # 新任务模型的训练器

__all__ = ['ATIO']  # 对外暴露的类 仅 ATIO 可被外部 import


class ATIO():
    """
    训练器统一管理类，通过模型名称映射到对应的训练器类
    简化不同模型的训练逻辑调用 实现"一键获取训练器"
    """
    def __init__(self):
        """初始化训练器映射表，注册所有支持的模型-训练器对应关系"""
        # 训练器映射表：key 为模型名称字符串 value 为对应的训练器类
        # 训练器类封装了模型的训练、验证、测试逻辑（如损失计算、参数更新等）
        self.TRAIN_MAP = {
            # 单任务模型对应的训练器（与 AMIO 中的模型名称一一对应）
            'tfn': TFN,
            'lmf': LMF,
            'mfn': MFN,
            'ef_lstm': EF_LSTM,
            'lf_dnn': LF_DNN,
            'graph_mfn': Graph_MFN,
            'mult': MULT,
            'mctn': MCTN,
            'bert_mag': BERT_MAG,
            'misa': MISA,
            'mfm': MFM,
            'mmim': MMIM,
            'cenet': CENET,
            'almt': ALMT,
            # 多任务模型对应的训练器
            'mtfn': MTFN,
            'mlmf': MLMF,
            'mlf_dnn': MLF_DNN,
            'self_mm': SELF_MM,
            'tetfn': TETFN,
            # 缺失模态任务模型对应的训练器
            'tfr_net': TFR_NET,
            # 新任务模型对应的训练器
            'camfn': CAMFN,
            'camfnet_aml': CAMFN,
        }
    
    def getTrain(self, args):
        """
        根据模型名称获取对应的训练器实例
        Args:
            args: 配置参数字典（必须包含'model_name'键，指定模型名称）
        Returns:
            训练器实例，封装了对应模型的训练逻辑
        """
        # 从映射表中获取模型名称对应的训练器类，并传入配置参数实例化
        return self.TRAIN_MAP[args['model_name']](args)