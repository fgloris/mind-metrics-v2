import cv2
import numpy as np
import torch
from transformers import AutoImageProcessor, AutoModel

PATCH_STRIDE = 16
REGISTER_TOKENS = 5  # 1 cls + 4 registers for DINOv3

def load_dinov3_model(model_path: str, device: str = 'cuda:0'):
    processor = AutoImageProcessor.from_pretrained(model_path)
    model = AutoModel.from_pretrained(model_path).to(device)
    model.eval()
    return model, processor


def extract_dinov3_features(frames: torch.Tensor, model, processor, device, batch_size=8) -> torch.Tensor:
    """
    从视频帧中提取DINOv3特征
    Args:
        frames: 视频帧张量，形状 [f, c, h, w]，范围 [0, 1]
        model: DINOv3模型（如果为None，会自动加载）
        processor: DINOv3 processor（如果为None，会自动加载）
        device: 计算设备
        batch_size: 批处理大小
    Returns:
        features: DINOv3特征张量，形状 [f, 3136, 768]
    """
    f = frames.shape[0]
    features_list = []

    # DINOv3期望输入范围:[0,1]
    with torch.inference_mode():
        for i in range(0, f, batch_size):
            batch_frames = frames[i:i+batch_size].to(device)  # [batch, c, h, w]

            # 直接使用tensor，processor会自动处理归一化
            inputs = processor(images=batch_frames, return_tensors="pt").to(device)

            # 提取特征
            outputs = model(**inputs, output_hidden_states=True)
            last_hidden_state = outputs.last_hidden_state  # [batch, 1 + 4 + 3136, 768]

            # 提取patch tokens (跳过前5个token)
            patch_tokens = last_hidden_state[:, 5:, :]  # [batch, 1 + 4 + 3136, 768]

            ## L2 normalization
            #patch_tokens_norm = torch.nn.functional.normalize(patch_tokens, p=2, dim=-1)
            features_list.append(patch_tokens.cpu())

            del batch_frames

    features = torch.cat(features_list, dim=0)  # [f, 1 + 4 + 3136, 768]
    return features
