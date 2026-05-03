import logging
import pickle

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data._utils.collate import default_collate

__all__ = ['MMDataLoader']

logger = logging.getLogger('MMSA')


class MMDataset(Dataset):
    """
    多模态情感分析（MMSA）专用数据集类，继承自PyTorch的Dataset
    负责加载、预处理多模态数据（文本、音频、视觉），支持数据对齐/非对齐、缺失数据生成、特征标准化等功能
    """
    def __init__(self, args, mode='train'):
        """
        初始化数据集实例
        Args:
            args: 配置参数字典（从config文件加载的edict），包含数据集路径、模型需求（如是否需要对齐）等
            mode: 数据模式，可选'train'（训练）、'valid'（验证）、'test'（测试），默认'train'
        """
        self.mode = mode  # 记录当前数据模式
        self.args = args  # 保存配置参数
        self.use_text = bool(self.args.get('use_text', True))
        self.use_audio = bool(self.args.get('use_audio', True))
        self.use_video = bool(self.args.get('use_video', True))
        self._initial_seq_lens = tuple(self.args.get('seq_lens', []))
        self.meta_info = {}  # 额外的样本元信息（如情绪标签等）
        # 数据集初始化函数映射：根据数据集名称调用对应的初始化方法
        DATASET_MAP = {
            'mosi': self.__init_mosi,
            'mosei': self.__init_mosei,
            'sims': self.__init_sims,
            'simsv2': self.__init_simsv2,
            'mustard': self.__init_mustard,  # 新增 MUStARD 数据集支持
            'mustardpp': self.__init_mustard,
            'sarcnet': self.__init_sarcnet,
        }
        # 调用对应数据集的初始化方法（如选择mosi则执行self.__init_mosi）
        DATASET_MAP[args['dataset_name']]()

    def __init_mosi(self):
        """
        加载并初始化MOSI数据集（核心逻辑，其他数据集复用此逻辑）
        步骤：1.加载特征文件 → 2.提取多模态特征 → 3.处理自定义模态特征 → 4.加载标签 → 5.处理非对齐数据 → 6.生成缺失数据 → 7.特征标准化
        """
        # 1. 加载特征文件：优先加载自定义特征，无则加载配置文件中指定的默认特征
        if self.args['custom_feature']:
            # 使用MMSA-FET工具提取的自定义特征文件
            with open(self.args['custom_feature'], 'rb') as f:
                data = pickle.load(f)
        else:
            # 使用配置文件中featurePath指定的默认特征文件（.pkl格式）
            with open(self.args['featurePath'], 'rb') as f:
                data = pickle.load(f)
        
        # 2. 提取文本特征：根据是否使用BERT区分特征类型，并更新文本特征维度
        self.text = None
        if self.use_text:
            if self.args.get('use_bert', None):
                self.text = data[self.mode]['text_bert'].astype(np.float32)  # BERT特征（768维）
                self.args['feature_dims'][0] = 768  # 更新文本特征维度为768
            else:
                self.text = data[self.mode]['text'].astype(np.float32)  # 普通文本特征
                self.args['feature_dims'][0] = self.text.shape[2]  # 从数据中自动获取文本维度
        
        # 提取音频、视觉特征，并更新对应模态的特征维度
        self.audio = data[self.mode]['audio'].astype(np.float32)
        self.args['feature_dims'][1] = self.audio.shape[2]
        self.vision = data[self.mode]['vision'].astype(np.float32)
        self.args['feature_dims'][2] = self.vision.shape[2]
        
        # 保存原始文本和样本ID（用于后续分析或可视化）
        self.raw_text = data[self.mode]['raw_text']
        self.ids = data[self.mode]['id']
        extra_keys = ['context', 'implicit_emotion', 'explicit_emotion', 'sarcasm_type', 'valence', 'arousal']
        self.meta_info = {}
        for key in extra_keys:
            if key in data[self.mode]:
                self.meta_info[key] = data[self.mode][key]
        if self.args['dataset_name'].lower() in ['mustard', 'mustardpp']:
            self._prepare_mustard_metadata()

        # 3. 处理自定义模态特征（若指定了单独的文本/音频/视觉特征文件，覆盖默认特征）
        if self.use_text and self.args['feature_T']:  # 自定义文本特征
            with open(self.args['feature_T'], 'rb') as f:
                data_T = pickle.load(f)
            if self.args.get('use_bert', None):
                self.text = data_T[self.mode]['text_bert'].astype(np.float32)
                self.args['feature_dims'][0] = 768
            else:
                self.text = data_T[self.mode]['text'].astype(np.float32)
                self.args['feature_dims'][0] = self.text.shape[2]
        
        if self.args['feature_A']:  # 自定义音频特征
            with open(self.args['feature_A'], 'rb') as f:
                data_A = pickle.load(f)
            self.audio = data_A[self.mode]['audio'].astype(np.float32)
            self.args['feature_dims'][1] = self.audio.shape[2]
        
        if self.args['feature_V']:  # 自定义视觉特征
            with open(self.args['feature_V'], 'rb') as f:
                data_V = pickle.load(f)
            self.vision = data_V[self.mode]['vision'].astype(np.float32)
            self.args['feature_dims'][2] = self.vision.shape[2]

        # 4. 加载标签：情感回归标签（M代表多模态融合标签），SIMS系列数据集额外加载单模态标签（T/A/V）
        self.labels = {
            'M': np.array(data[self.mode]['regression_labels']).astype(np.float32)  # 多模态回归标签
        }
        if 'sims' in self.args['dataset_name']:  # SIMS/SIMSv2数据集需加载单模态标签
            for m in "TAV":  # T=文本，A=音频，V=视觉
                self.labels[m] = data[self.mode]['regression' + '_labels_' + m].astype(np.float32)

        # 打印当前模式的样本数量（日志记录）
        logger.info(f"{self.mode} samples: {self.labels['M'].shape}")

        # 5. 处理非对齐数据：非对齐模型需记录音频/视觉的有效序列长度（用于后续padding/masking）
        if not self.args['need_data_aligned']:
            # 优先从自定义特征中获取长度，无则从默认数据中获取
            if self.args['feature_A']:
                self.audio_lengths = list(data_A[self.mode]['audio_lengths'])
            else:
                self.audio_lengths = data[self.mode]['audio_lengths']
            
            if self.args['feature_V']:
                self.vision_lengths = list(data_V[self.mode]['vision_lengths'])
            else:
                self.vision_lengths = data[self.mode]['vision_lengths']
        
        # 处理音频特征中的无穷小值（替换为0，避免训练报错）
        self.audio[self.audio == -np.inf] = 0

        # 6. 生成缺失数据（模拟模态缺失场景，仅支持非对齐数据）
        if self.args.get('data_missing'):
            # 生成文本缺失数据：输入mask、缺失率、随机种子控制缺失模式
            if self.use_text and self.text is not None:
                self.text_m, self.text_length, self.text_mask, self.text_missing_mask = self.generate_m(
                    self.text[:,0,:], self.text[:,1,:], None,
                    self.args.missing_rate[0], self.args.missing_seed[0], mode='text'
                )
                # 重组文本缺失特征（拼接Input_ids、Input_mask、Segment_ids，适配BERT输入格式）
                Input_ids_m = np.expand_dims(self.text_m, 1)
                Input_mask = np.expand_dims(self.text_mask, 1)
                Segment_ids = np.expand_dims(self.text[:,2,:], 1)
                self.text_m = np.concatenate((Input_ids_m, Input_mask, Segment_ids), axis=1)
            else:
                self.text_m = None
                self.text_length = None
                self.text_mask = None
                self.text_missing_mask = None

            # 若需要数据对齐：用文本mask长度作为音频/视觉的长度（强制对齐）
            if self.args['need_data_aligned']:
                if self.use_text and self.text is not None:
                    base_lengths = np.sum(self.text[:,1,:], axis=1, dtype=np.int32)
                else:
                    seq_len = self.audio.shape[1]
                    base_lengths = np.full((self.audio.shape[0],), seq_len, dtype=np.int32)
                self.audio_lengths = np.array(base_lengths, dtype=np.int32)
                self.vision_lengths = np.array(base_lengths, dtype=np.int32)

            # 生成音频缺失数据
            if self.use_audio:
                self.audio_m, self.audio_length, self.audio_mask, self.audio_missing_mask = self.generate_m(
                    self.audio, None, self.audio_lengths,
                    self.args.missing_rate[1], self.args.missing_seed[1], mode='audio'
                )
            else:
                self.audio_m = None
                self.audio_length = None
                self.audio_mask = None
                self.audio_missing_mask = None
            # 生成视觉缺失数据
            if self.use_video:
                self.vision_m, self.vision_length, self.vision_mask, self.vision_missing_mask = self.generate_m(
                    self.vision, None, self.vision_lengths,
                    self.args.missing_rate[2], self.args.missing_seed[2], mode='vision'
                )
            else:
                self.vision_m = None
                self.vision_length = None
                self.vision_mask = None
                self.vision_missing_mask = None

        # 7. 特征标准化（若配置要求标准化，调用__normalize方法）
        if self.args.get('need_normalized'):
            self.__normalize()
    
    def __init_mosei(self):
        """MOSEI数据集初始化：复用MOSI的逻辑（特征格式一致）"""
        return self.__init_mosi()  # 注：此处存在笔误，应为return self.__init_mosi() → 实际应改为self.__init_mosi()（无需return）

    def __init_sims(self):
        """SIMS数据集初始化：复用MOSI的逻辑（特征格式一致）"""
        return self.__init_mosi()  # 同上，笔误，实际应为self.__init_mosi()

    def __init_simsv2(self):
        """SIMSv2数据集初始化：复用MOSI的逻辑（特征格式一致）"""
        return self.__init_mosi()  # 同上，笔误，实际应为self.__init_mosi()

    def __init_mustard(self):
        """
        MUStARD 数据集初始化：复用 MOSI 的标准特征读取流程
        支持通过 featurePath 或 feature_T/A/V 参数指定特征文件路径
        数据格式要求与 MOSI 相同（包含 text_bert, audio, vision, regression_labels 等字段）
        """
        return self.__init_mosi()

    def __init_sarcnet(self):
        """Initialize SarcNet with the same pickle schema used by MOSI."""
        return self.__init_mosi()

    def _prepare_mustard_metadata(self):
        """
        Convert MUStARD/MUStARD++ metadata (explicit/implicit emotions, valence/arousal) to tensors
        """
        if not self.meta_info:
            return
        implicit = self.meta_info.get('implicit_emotion')
        explicit = self.meta_info.get('explicit_emotion')
        if implicit is None or explicit is None:
            return
        implicit = np.asarray(implicit)
        explicit = np.asarray(explicit)
        vocab = sorted(set(implicit.tolist()) | set(explicit.tolist()))
        mapper = {emo: idx for idx, emo in enumerate(vocab)}
        self.meta_info['implicit_label'] = np.array([mapper[e] for e in implicit], dtype=np.int64)
        self.meta_info['explicit_label'] = np.array([mapper[e] for e in explicit], dtype=np.int64)
        self.args['emotion_vocab'] = vocab
        self.args['num_emotions'] = len(vocab)
        valence = np.asarray(self.meta_info.get('valence', np.zeros(len(implicit))), dtype=np.float32)
        arousal = np.asarray(self.meta_info.get('arousal', np.zeros(len(implicit))), dtype=np.float32)
        valence = np.nan_to_num(valence, nan=0.0)
        arousal = np.nan_to_num(arousal, nan=0.0)
        self.meta_info['valence'] = valence
        self.meta_info['arousal'] = arousal
        self.meta_info['affect_vector'] = np.stack([valence, arousal], axis=1).astype(np.float32)


    def generate_m(self, modality, input_mask, input_len, missing_rate, missing_seed, mode='text'):
        """
        生成缺失模态数据（模拟模态部分缺失场景）
        Args:
            modality: 原始模态特征（文本/音频/视觉）
            input_mask: 输入mask（标记有效特征位置，1=有效，0=无效）
            input_len: 模态有效序列长度
            missing_rate: 缺失率（0~1，如0.2代表20%缺失）
            missing_seed: 随机种子（保证缺失模式可复现）
            mode: 模态类型（'text'/'audio'/'vision'）
        Returns:
            modality_m: 含缺失的模态特征
            input_len: 有效序列长度
            input_mask: 原始输入mask
            missing_mask: 缺失mask（1=保留，0=缺失）
        """
        # 1. 确定输入mask和有效长度（不同模态逻辑不同）
        if mode == 'text':
            # 文本模态：从input_mask中获取有效长度（找到第一个0的位置）
            input_len = np.argmin(input_mask, axis=1)
        elif mode == 'audio' or mode == 'vision':
            # 音频/视觉模态：从input_len生成input_mask（1=有效长度内，0=超出部分）
            input_mask = np.array([np.array([1] * length + [0] * (modality.shape[1] - length)) for length in input_len])
        
        # 2. 生成缺失mask：基于缺失率随机生成，仅在有效区域（input_mask=1）内缺失
        np.random.seed(missing_seed)  # 固定随机种子，保证复现
        missing_mask = (np.random.uniform(size=input_mask.shape) > missing_rate) * input_mask  # 1=保留，0=缺失
        assert missing_mask.shape == input_mask.shape  # 确保mask形状与输入一致
        
        # 3. 生成含缺失的模态特征（不同模态处理方式不同）
        if mode == 'text':
            # 文本模态：保留CLS和SEP token（不缺失），缺失部分用UNK token（100）填充
            for i, instance in enumerate(missing_mask):
                instance[0] = instance[input_len[i] - 1] = 1  # CLS（第0位）和SEP（最后一位有效位）不缺失
            # 缺失位置替换为UNK（100），有效位置保留原始特征
            modality_m = missing_mask * modality + (100 * np.ones_like(modality)) * (input_mask - missing_mask)
        elif mode == 'audio' or mode == 'vision':
            # 音频/视觉模态：缺失位置直接置0（通过mask与原始特征相乘实现）
            modality_m = missing_mask.reshape(modality.shape[0], modality.shape[1], 1) * modality
        
        return modality_m, input_len, input_mask, missing_mask

    def __truncate(self):
        """
        截断多模态特征到指定长度（注：当前实现存在硬编码长度20，需根据配置动态调整，暂未被调用）
        作用：统一不同样本的序列长度，避免过长序列导致训练效率低
        """
        # 截断单个模态特征的函数
        def do_truncate(modal_features, length):
            if length == modal_features.shape[1]:  # 长度一致，无需截断
                return modal_features
            truncated_feature = []
            padding = np.array([0 for i in range(modal_features.shape[2])])  #  padding值（0）
            for instance in modal_features:
                for index in range(modal_features.shape[1]):
                    # 找到第一个非padding位置，从该位置开始截断指定长度
                    if((instance[index] == padding).all()):
                        if(index + length >= modal_features.shape[1]):
                            truncated_feature.append(instance[index:index+20])  # 硬编码20，需优化
                            break
                    else:                        
                        truncated_feature.append(instance[index:index+20])  # 硬编码20，需优化
                        break
            truncated_feature = np.array(truncated_feature)
            return truncated_feature
        
        # 从配置中获取各模态目标长度，执行截断
        text_length, audio_length, video_length = self.args['seq_lens']
        self.vision = do_truncate(self.vision, video_length)
        if self.use_text and self.text is not None:
            self.text = do_truncate(self.text, text_length)
        self.audio = do_truncate(self.audio, audio_length)

    def __normalize(self):
        """
        多模态特征标准化（仅对音频和视觉模态执行时间维度上的均值归一化）
        作用：消除不同模态特征的量级差异，避免训练时梯度失衡
        """
        # 1. 转置特征维度：(样本数, 时间步, 特征维度) → (时间步, 样本数, 特征维度)
        self.vision = np.transpose(self.vision, (1, 0, 2))
        self.audio = np.transpose(self.audio, (1, 0, 2))
        
        # 2. 时间维度均值归一化：对每个样本的音频/视觉特征，在时间步上取平均（降维为1个时间步）
        # 目的：将时序特征转换为全局特征，简化模型输入（部分早期模型需求）
        self.vision = np.mean(self.vision, axis=0, keepdims=True)  # (1, 样本数, 特征维度)
        self.audio = np.mean(self.audio, axis=0, keepdims=True)    # (1, 样本数, 特征维度)

        # 3. 处理NaN值（归一化后可能出现，替换为0）
        self.vision[self.vision != self.vision] = 0
        self.audio[self.audio != self.audio] = 0

        # 4. 转置回原始维度：(1, 样本数, 特征维度) → (样本数, 1, 特征维度)
        self.vision = np.transpose(self.vision, (1, 0, 2))
        self.audio = np.transpose(self.audio, (1, 0, 2))

    def __len__(self):
        """Dataset类必须实现：返回数据集样本总数（以多模态标签数量为准）"""
        return len(self.labels['M'])

    def get_seq_len(self):
        """Return sequence lengths for text/audio/vision tensors."""
        if not self.use_text or self.text is None:
            text_len = self._initial_seq_lens[0] if len(self._initial_seq_lens) >= 1 else 0
        elif 'use_bert' in self.args and self.args['use_bert']:
            text_len = self.text.shape[2]
        else:
            text_len = self.text.shape[1]

        if not self.use_audio or self.audio is None:
            audio_len = self._initial_seq_lens[1] if len(self._initial_seq_lens) >= 2 else 0
        else:
            audio_len = self.audio.shape[1]

        if not self.use_video or self.vision is None:
            video_len = self._initial_seq_lens[2] if len(self._initial_seq_lens) >= 3 else 0
        else:
            video_len = self.vision.shape[1]

        return (text_len, audio_len, video_len)

    def get_feature_dim(self):
        """Return feature dimensions for text/audio/vision tensors."""
        if not self.use_text or self.text is None:
            text_dim = self.args['feature_dims'][0] if 'feature_dims' in self.args else 0
        else:
            text_dim = self.text.shape[2]

        if not self.use_audio or self.audio is None:
            audio_dim = self.args['feature_dims'][1] if 'feature_dims' in self.args else 0
        else:
            audio_dim = self.audio.shape[2]

        if not self.use_video or self.vision is None:
            video_dim = self.args['feature_dims'][2] if 'feature_dims' in self.args else 0
        else:
            video_dim = self.vision.shape[2]

        return text_dim, audio_dim, video_dim

    def __getitem__(self, index):
        """
        Dataset类必须实现：根据索引获取单个样本（转换为PyTorch张量）
        Args:
            index: 样本索引
        Returns:
            sample: 字典格式的单样本数据，包含各模态特征、标签、元信息等
        """
        # 基础样本数据：原始文本、各模态特征（转换为Tensor）、索引、ID、标签
        sample = {
            'raw_text': self.raw_text[index],  # 原始文本
            'text': torch.Tensor(self.text[index]) if (self.use_text and self.text is not None) else None,  # 文本特征（Tensor）
            'audio': torch.Tensor(self.audio[index]) if (self.use_audio and self.audio is not None) else None,  # 音频特征（Tensor）
            'vision': torch.Tensor(self.vision[index]) if (self.use_video and self.vision is not None) else None,  # 视觉特征（Tensor）
            'index': index,  # 样本索引
            'id': self.ids[index],  # 样本ID
            # 标签：转换为Tensor，reshape(-1)确保维度统一（如(1,)而非()）
            'labels': {k: torch.Tensor(v[index].reshape(-1)) for k, v in self.labels.items()}
        } 
        if self.meta_info:
            if 'implicit_label' in self.meta_info:
                sample['implicit_label'] = torch.tensor(self.meta_info['implicit_label'][index]).long()
            if 'explicit_label' in self.meta_info:
                sample['explicit_label'] = torch.tensor(self.meta_info['explicit_label'][index]).long()
            if 'affect_vector' in self.meta_info:
                sample['affect_vector'] = torch.tensor(self.meta_info['affect_vector'][index]).float()
            if 'valence' in self.meta_info:
                sample['valence'] = torch.tensor(self.meta_info['valence'][index]).float()
            if 'arousal' in self.meta_info:
                sample['arousal'] = torch.tensor(self.meta_info['arousal'][index]).float()
        # 非对齐数据：添加音频/视觉的有效长度（用于模型中的动态padding）
        if not self.args['need_data_aligned']:
            if self.use_audio:
                sample['audio_lengths'] = self.audio_lengths[index]
            if self.use_video:
                sample['vision_lengths'] = self.vision_lengths[index]
        # 缺失数据：添加含缺失的模态特征和缺失mask（用于缺失模态鲁棒性训练）
        if self.args.get('data_missing'):
            if self.use_text and self.text_m is not None:
                sample['text_m'] = torch.Tensor(self.text_m[index])  # 含缺失的文本特征
                sample['text_missing_mask'] = torch.Tensor(self.text_missing_mask[index])  # 文本缺失mask
            if self.use_audio and self.audio_m is not None:
                sample['audio_m'] = torch.Tensor(self.audio_m[index])  # 含缺失的音频特征
                sample['audio_missing_mask'] = torch.Tensor(self.audio_missing_mask[index])  # 音频缺失mask
            if self.use_audio:
                sample['audio_lengths'] = self.audio_lengths[index]
                if self.audio_mask is not None:
                    sample['audio_mask'] = self.audio_mask[index]
            if self.use_video and self.vision_m is not None:
                sample['vision_m'] = torch.Tensor(self.vision_m[index])  # 含缺失的视觉特征
                sample['vision_missing_mask'] = torch.Tensor(self.vision_missing_mask[index])  # 视觉缺失mask
            if self.use_video:
                sample['vision_lengths'] = self.vision_lengths[index]  # 视觉有效长度
                if self.vision_mask is not None:
                    sample['vision_mask'] = self.vision_mask[index]

        return sample


def _collate_with_optional(batch):
    """
    Custom collate_fn that keeps entries as None if any sample provides None
    (default_collate doesn't support NoneType).
    """
    elem0 = batch[0]
    collated = {}
    for key in elem0:
        values = [sample[key] for sample in batch]
        if any(v is None for v in values):
            collated[key] = None
        else:
            collated[key] = default_collate(values)
    return collated


def MMDataLoader(args, num_workers):
    """
    多模态数据加载器生成函数：创建训练/验证/测试集的DataLoader（PyTorch）
    Args:
        args: 配置参数字典（含batch_size、数据集名称等）
        num_workers: 数据加载的进程数（用于并行加载，提升效率）
    Returns:
        dataLoader: 字典格式的DataLoader，包含'train'/'valid'/'test'三个键
    """
    # 1. 创建MMDataset实例（分别对应训练、验证、测试集）
    datasets = {
        'train': MMDataset(args, mode='train'),
        'valid': MMDataset(args, mode='valid'),
        'test': MMDataset(args, mode='test')
    }

    # 2. 更新配置中的序列长度：从训练集获取实际序列长度（覆盖配置中的默认值）
    if 'seq_lens' in args:
        args['seq_lens'] = datasets['train'].get_seq_len() 

    # 3. 创建DataLoader：实现批量加载、并行处理、训练集打乱
    dataLoader = {
        ds: DataLoader(
            datasets[ds],
            batch_size=args['batch_size'],  # 批量大小（从配置获取）
            num_workers=num_workers,        # 并行进程数（传入参数）
            shuffle=True if ds == 'train' else False,  # 仅训练集打乱，验证/测试集不打乱
            collate_fn=_collate_with_optional
        )
        for ds in datasets.keys()  # 遍历'train'/'valid'/'test'，生成对应DataLoader
    }
    
    return dataLoader
