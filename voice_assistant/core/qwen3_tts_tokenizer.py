#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Qwen3-TTS Tokenizer 实现 (简化版，兼容 ONNX embedding 3072)

由于 ONNX 模型的 embedding 只有 3072 个条目，我们必须使用简化的编码方式:
- 使用字符的 Unicode 码点作为 token ID
- 使用 1 作为 BOS，2 作为 EOS
- 不使用任何大于 3071 的 token ID
"""

import os
from typing import List


class Qwen3TTSTokenizer:
    """
    Qwen3-TTS Tokenizer (简化版，兼容 ONNX embedding 3072)
    """
    
    BOS_TOKEN_ID = 1
    EOS_TOKEN_ID = 2
    
    def __init__(self, model_path: str = None):
        """
        初始化 Tokenizer

        参数:
            model_path: Qwen3-TTS 模型目录路径
        """
        self.model_path = model_path or "/userdata/llm_models/Qwen3-TTS-12Hz-0.6B-CustomVoice"
        self.base_tokenizer = None
        self._load_tokenizer()
    
    def _load_tokenizer(self):
        """加载原始 tokenizer"""
        try:
            from transformers import AutoTokenizer
            if os.path.exists(self.model_path):
                self.base_tokenizer = AutoTokenizer.from_pretrained(
                    self.model_path,
                    trust_remote_code=True,
                    local_files_only=True
                )
                print(f"✓ 加载 base tokenizer 成功")
        except Exception as e:
            print(f"⚠️ 加载 base tokenizer 失败: {e}")
            self.base_tokenizer = None
    
    def encode_text(self, text: str) -> List[int]:
        """
        编码文本为 token IDs (使用 0-3071 范围内的 ID)

        策略:
        - 所有字符: 使用 ord(c) % 3072 (将大 ID 映射到 0-3071)
        """
        return [ord(c) % 3072 for c in text]
    
    def decode_text(self, token_ids: List[int]) -> str:
        """解码 token IDs 为文本（简化版）"""
        chars = []
        for tid in token_ids:
            if 0 <= tid <= 0x10FFFF:
                try:
                    chars.append(chr(tid))
                except:
                    pass
        return ''.join(chars)
    
    def build_tts_prompt(
        self,
        text: str,
        speaker: str = "Vivian",
        language: str = "chinese",
        instruct: str = ""
    ) -> List[int]:
        """
        构建 TTS 输入 prompt (简化格式，兼容 ONNX embedding)

        格式: [BOS][text_tokens][EOS]
        所有 token ID 都在 0-3071 范围内
        """
        text_tokens = self.encode_text(text)
        
        prompt = [self.BOS_TOKEN_ID]
        prompt.extend(text_tokens)
        prompt.append(self.EOS_TOKEN_ID)
        
        return prompt
    
    def extract_codec_tokens(self, token_ids: List[int]) -> List[int]:
        """
        从生成的 token 中提取 codec tokens

        Codec token 范围: 0-1024
        过滤掉特殊 token 和非 codec token

        参数:
            token_ids: 生成的 token IDs

        返回:
            codec token 列表
        """
        codec_tokens = []
        for tid in token_ids:
            if 0 <= tid < 1024:
                codec_tokens.append(tid)
        return codec_tokens
    
    def get_vocab_size(self) -> int:
        """获取 vocab 大小"""
        return 3072
