#!/usr/bin/env python3
"""创建 Matcha-TTS 的 tokens.txt 和 lexicon.txt"""

from matcha.text.symbols import symbols

# 创建 tokens.txt
print("Creating tokens.txt...")
with open("matcha_tokens.txt", "w", encoding="utf-8") as f:
    for i, symbol in enumerate(symbols):
        # 处理特殊字符
        if symbol == " ":
            symbol = "<space>"
        elif symbol == "_":
            symbol = "<pad>"
        f.write(f"{symbol}\t{i}\n")

print(f"Created matcha_tokens.txt with {len(symbols)} tokens")

# 创建简单的 lexicon.txt (英文音素映射)
# 注意：Matcha-TTS 使用 phonemizer 进行 G2P，这里创建一个简单的示例
print("Creating lexicon.txt...")
with open("matcha_lexicon.txt", "w", encoding="utf-8") as f:
    # 添加一些常见词作为示例
    f.write("hello\th ə ˈ l oʊ\n")
    f.write("world\tw ɜːr l d\n")
    f.write("the\tð ə\n")
    f.write("a\tə\n")
    f.write("is\tɪ z\n")

print("Created matcha_lexicon.txt")
print("\nNote: For full English TTS, you should use espeak-ng for G2P conversion.")
print("The lexicon file is optional if --matcha-data-dir is provided with espeak-ng data.")
