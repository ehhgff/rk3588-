"""
QAT (Quantization-Aware Training) for SenseVoiceSmall Encoder

Usage:
    python3 qat_sensevoice.py --mode calibrate
    python3 qat_sensevoice.py --mode finetune --train-steps 500
    python3 qat_sensevoice.py --mode export
    python3 qat_sensevoice.py --mode all --train-steps 500
"""

import os
import sys
import argparse
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.quantization import QuantStub, DeQuantStub, QuantWrapper

from funasr.models.sense_voice.model import SenseVoiceSmall

INPUT_LEN = 124
CALIB_DIR = "/home/ubuntu/桌面/ai/calib_data/data_large"


def sequence_mask(lengths, maxlen=None, dtype=torch.float32, device=None):
    if maxlen is None:
        maxlen = lengths.max()
    row_vector = torch.arange(0, maxlen, 1).to(lengths.device)
    matrix = torch.unsqueeze(lengths, dim=-1)
    mask = row_vector < matrix
    return mask.type(dtype).to(device) if device is not None else mask.type(dtype)


def encoder_forward(self, xs_pad, ilens):
    maxlen = xs_pad.size(1)
    masks = sequence_mask(ilens, maxlen=maxlen, device=ilens.device)[:, None, :]
    xs_pad *= self.output_size() ** 0.5
    xs_pad = self.embed(xs_pad)

    for layer_idx, encoder_layer in enumerate(self.encoders0):
        encoder_outs = encoder_layer(xs_pad, masks)
        xs_pad, masks = encoder_outs[0], encoder_outs[1]

    for layer_idx, encoder_layer in enumerate(self.encoders):
        encoder_outs = encoder_layer(xs_pad, masks)
        xs_pad, masks = encoder_outs[0], encoder_outs[1]

    xs_pad = self.after_norm(xs_pad)
    olens = masks.squeeze(1).sum(1).int()

    for layer_idx, encoder_layer in enumerate(self.tp_encoders):
        encoder_outs = encoder_layer(xs_pad, masks)
        xs_pad, masks = encoder_outs[0], encoder_outs[1]

    xs_pad = self.tp_norm(xs_pad)
    return xs_pad, olens


def model_forward(self, x, x_length, language, text_norm):
    language_query = self.embed(language.view(-1)).unsqueeze(1)
    text_norm_query = self.embed(text_norm.view(-1)).unsqueeze(1)
    event_emo_query = self.embed(torch.LongTensor([[1, 2]])).repeat(x.size(0), 1, 1)
    x = torch.cat((language_query, event_emo_query, text_norm_query, x), dim=1)
    x_length += 4
    encoder_out, encoder_out_lens = self.encoder(x, x_length)
    if isinstance(encoder_out, tuple):
        encoder_out = encoder_out[0]
    ctc_logits = self.ctc.ctc_lo(encoder_out)
    return ctc_logits


# ------------------------------------------------------------
# Quantization wrapper replacement
# ------------------------------------------------------------
def replace_module(model, condition_fn, new_module_fn):
    for name, child in model.named_children():
        if condition_fn(name, child):
            setattr(model, name, new_module_fn(child))
        else:
            replace_module(child, condition_fn, new_module_fn)
    return model


def prepare_qat_model(model):
    """Wrap Linear and Conv1d with QuantWrapper for QAT."""
    model = replace_module(
        model,
        lambda n, m: isinstance(m, nn.Linear),
        lambda m: QuantWrapper(m),
    )
    model = replace_module(
        model,
        lambda n, m: isinstance(m, nn.Conv1d),
        lambda m: QuantWrapper(m),
    )
    return model


def load_model():
    model, params = SenseVoiceSmall.from_pretrained(
        model="iic/SenseVoiceSmall", device="cpu"
    )
    model.__class__.forward = model_forward
    model.encoder.__class__.forward = encoder_forward
    model.eval()
    return model, params


# ------------------------------------------------------------
# Data loading
# ------------------------------------------------------------
def load_calib_data(data_dir=CALIB_DIR, num_samples=None):
    files = sorted(glob.glob(os.path.join(data_dir, "feat_*.npy")))
    if num_samples is not None:
        files = files[:num_samples]
    data = []
    for f in files:
        arr = np.load(f)
        data.append(arr)
    return data


class CalibDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir=CALIB_DIR, num_samples=None, seq_len=100,
                 target_len=124):
        self.features = load_calib_data(data_dir, num_samples)
        self.seq_len = seq_len
        self.target_len = target_len

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        feat = self.features[idx]
        feat = feat[0, :self.seq_len, :]
        actual_len = feat.shape[0]
        if actual_len < self.target_len:
            pad = np.zeros((self.target_len - actual_len, feat.shape[1]),
                          dtype=feat.dtype)
            feat = np.concatenate([feat, pad], axis=0)
        x_length = torch.tensor([actual_len], dtype=torch.int32)
        language = torch.tensor([3], dtype=torch.int32)
        text_norm = torch.tensor([15], dtype=torch.int32)
        return torch.from_numpy(feat).float(), x_length, language, text_norm


# ------------------------------------------------------------
# Cosine similarity evaluation
# ------------------------------------------------------------
@torch.no_grad()
def evaluate_cosine(model_qat, model_fp16, calib_loader, num_batches=20):
    cos_sims = []
    model_qat.eval()
    model_fp16.eval()

    for i, (x, x_len, lang, tn) in enumerate(calib_loader):
        if i >= num_batches:
            break
        out_qat = model_qat(x, x_len, lang, tn)
        out_fp16 = model_fp16(x, x_len, lang, tn)

        a = out_qat.flatten().float().numpy()
        b = out_fp16.flatten().float().numpy()
        dot = np.dot(a, b)
        norm = np.linalg.norm(a) * np.linalg.norm(b)
        cos = float(dot / norm) if norm > 1e-10 else 1.0
        cos_sims.append(cos)

    return float(np.mean(cos_sims))


# ------------------------------------------------------------
# Calibration
# ------------------------------------------------------------
def calibrate(model, calib_loader, num_steps=143):
    print(f"\n>>> Calibration: {num_steps} steps")
    model.eval()
    with torch.no_grad():
        for i, (x, x_len, lang, tn) in enumerate(calib_loader):
            if i >= num_steps:
                break
            _ = model(x, x_len, lang, tn)
            if (i + 1) % 20 == 0:
                print(f"  Calibrated {i+1}/{num_steps}")
    print("  Calibration done.")
    return model


# ------------------------------------------------------------
# QAT Fine-tuning
# ------------------------------------------------------------
def qat_finetune(model, calib_loader, train_steps=500, lr=2e-5, log_interval=50):
    print(f"\n>>> QAT Fine-tuning: {train_steps} steps, lr={lr}")

    # Training: update quantization parameters and final layers
    quant_params = []
    final_params = []
    for name, param in model.named_parameters():
        low = name.lower()
        if 'fake_quant' in low or 'scale' in low or 'zero_point' in low:
            quant_params.append(param)
        elif ('after_norm' in name or 'tp_norm' in name or
              'ctc_lo' in name or 'linear_out' in name or
              'w_2' in name):
            final_params.append(param)

    optimizer = torch.optim.AdamW([
        {'params': quant_params, 'lr': lr * 10},
        {'params': final_params, 'lr': lr},
    ], weight_decay=1e-5)

    model_fp16_ref, _ = load_model()

    model.train()
    step = 0
    best_cos = 0.0
    iter_loader = iter(calib_loader)

    while step < train_steps:
        try:
            x, x_len, lang, tn = next(iter_loader)
        except StopIteration:
            iter_loader = iter(calib_loader)
            x, x_len, lang, tn = next(iter_loader)

        optimizer.zero_grad()
        out_qat = model(x, x_len, lang, tn)

        with torch.no_grad():
            out_fp16 = model_fp16_ref(x, x_len, lang, tn)

        loss = nn.MSELoss()(out_qat, out_fp16)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        step += 1

        if step % log_interval == 0:
            cos = evaluate_cosine(model, model_fp16_ref, calib_loader, num_batches=10)
            print(f"  Step {step}/{train_steps}: loss={loss.item():.6f}, cos={cos:.6f}")
            if cos > best_cos:
                best_cos = cos
                torch.save(model.state_dict(), "qat_best_checkpoint.pt")
                print(f"  -> New best cos={cos:.6f}, checkpoint saved")
            model.train()

    print(f">>> QAT done. Best cos={best_cos:.6f}")
    return model


# ------------------------------------------------------------
# ONNX Export - FP16 version with QAT weights (single input)
# ------------------------------------------------------------
class QATEncoderWrapper(nn.Module):
    """Wrapper that takes only audio features x and hardcodes auxiliary inputs."""
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        x_length = torch.full((x.size(0),), x.size(1), dtype=torch.int32)
        language = torch.full((x.size(0),), 3, dtype=torch.int32)
        text_norm = torch.full((x.size(0),), 15, dtype=torch.int32)
        return self.model(x, x_length, language, text_norm)


def export_fp16_onnx(model, state_dict, output_path="sensevoice_qat.onnx"):
    """
    Load QAT-trained weights into a clean FP16 model and export to ONNX.
    Only takes audio features as input; auxiliary parameters are hardcoded.
    This avoids PyTorch 2.11's torch.export issue with FakeQuantize/Quantized ops.
    RKNN will use rknn.build(do_quantization=True) with calibration data.
    """
    print(f"\n>>> Exporting FP16 ONNX (with QAT weights): {output_path}")

    # Create fresh model (no QAT modifications)
    model_fp16, _ = load_model()

    # Map QAT state_dict keys to FP16 model keys
    # QAT: "encoder.encoders0.0.self_attn.linear_out.module.weight"
    # FP16: "encoder.encoders0.0.self_attn.linear_out.weight"
    # Strip ".module" prefix from QAT weight/bias keys
    fp16_state = {}
    for key in state_dict:
        if not (key.endswith('.weight') or key.endswith('.bias')):
            continue
        fp16_key = key.replace('.module.', '.')
        if fp16_key in model_fp16.state_dict():
            fp16_state[fp16_key] = state_dict[key]
    for key in state_dict:
        if not key.endswith('.weight') and not key.endswith('.bias'):
            continue
        if '.module.' in key:
            continue
        if key in model_fp16.state_dict() and key not in fp16_state:
            fp16_state[key] = state_dict[key]

    model_fp16.load_state_dict(fp16_state, strict=False)
    model_fp16.eval()
    print(f"  Loaded QAT weights into FP16 model ({len(fp16_state)}/{len(model_fp16.state_dict())} keys)")

    wrapped = QATEncoderWrapper(model_fp16)
    wrapped.eval()

    x = torch.randn(1, 100, 560, dtype=torch.float32)

    torch.onnx.export(
        wrapped,
        x,
        output_path,
        opset_version=13,
        input_names=["x"],
        output_names=["logits"],
        do_constant_folding=True,
        dynamo=False,
    )
    print(f"  ONNX exported: {output_path}")

    import onnx
    onnx_model = onnx.load(output_path)
    print(f"  Inputs: {[i.name for i in onnx_model.graph.input]}")
    print(f"  Outputs: {[o.name for o in onnx_model.graph.output]}")
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"  Size: {size_mb:.2f} MB")
    return output_path


# ------------------------------------------------------------
# ONNX Export with FakeQuantize (for QAT-aware converters)
# ------------------------------------------------------------
def export_qat_onnx(model, output_path="sensevoice_qat_qat.onnx"):
    print(f"\n>>> Exporting QAT ONNX (with FakeQuantize nodes): {output_path}")
    model.eval()

    x = torch.randn(1, INPUT_LEN, 560, dtype=torch.float32)
    x_length = torch.tensor([100], dtype=torch.int32)
    language = torch.tensor([3], dtype=torch.int32)
    text_norm = torch.tensor([15], dtype=torch.int32)

    # Try different export strategies
    strategies = [
        ("torch.jit.trace + save", 0),
        ("torch.onnx.export", 1),
    ]

    for name, strategy in strategies:
        try:
            if strategy == 0:
                traced = torch.jit.trace(model, (x, x_length, language, text_norm))
                traced.save(output_path.replace('.onnx', '.pt'))
                print(f"  JIT script saved: {output_path.replace('.onnx', '.pt')}")
            elif strategy == 1:
                torch.onnx.export(
                    model,
                    (x, x_length, language, text_norm),
                    output_path,
                    opset_version=13,
                    input_names=["x", "x_length", "language", "text_norm"],
                    output_names=["logits"],
                    do_constant_folding=True,
                )
                print(f"  ONNX exported: {output_path}")
            return output_path
        except Exception as e:
            print(f"  Strategy '{name}' failed: {type(e).__name__}")
            continue

    print("  WARNING: All export strategies failed.")
    print("  Falling back to FP16 export with QAT weights.")
    return None


# ------------------------------------------------------------
# Main pipeline
# ------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="SenseVoice QAT Pipeline")
    parser.add_argument("--mode", choices=["calibrate", "finetune", "export", "all"],
                        default="all")
    parser.add_argument("--calib-steps", type=int, default=143)
    parser.add_argument("--train-steps", type=int, default=500)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--output", type=str, default="sensevoice_qat.onnx")
    args = parser.parse_args()

    print("=" * 60)
    print("SenseVoice QAT Pipeline")
    print("=" * 60)
    print(f"Mode: {args.mode}")

    # 1. Load model
    model, params = load_model()
    print(f"Model: {sum(p.numel() for p in model.parameters()):,} params")

    # 2. Replace Linear/Conv1d with QuantWrapper
    model = prepare_qat_model(model)
    print(f"QuantWrapper applied to Linear and Conv1d layers")

    # 3. Set qconfig and prepare for QAT
    qconfig = torch.quantization.QConfig(
        activation=torch.quantization.FakeQuantize.with_args(
            observer=torch.quantization.MovingAverageMinMaxObserver,
            quant_min=-128, quant_max=127,
            dtype=torch.qint8,
            qscheme=torch.per_tensor_affine,
            reduce_range=False,
        ),
        weight=torch.quantization.FakeQuantize.with_args(
            observer=torch.quantization.MovingAverageMinMaxObserver,
            quant_min=-128, quant_max=127,
            dtype=torch.qint8,
            qscheme=torch.per_tensor_symmetric,
            reduce_range=False,
        ),
    )
    model.qconfig = qconfig
    model.embed.qconfig = None
    model.train()
    model = torch.quantization.prepare_qat(model, inplace=True)
    print(f"QAT observers inserted")

    # 4. Load calibration data
    dataset = CalibDataset(data_dir=CALIB_DIR, num_samples=143, seq_len=100)
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)
    print(f"Calibration data: {len(dataset)} samples")

    # 5. Calibrate
    if args.mode in ("calibrate", "all"):
        model.eval()
        calibrate(model, loader, num_steps=args.calib_steps)
        torch.save(model.state_dict(), "qat_calibrated.pt")
        print("  Checkpoint saved: qat_calibrated.pt")

        if args.mode == "calibrate":
            model_fp16, _ = load_model()
            cos = evaluate_cosine(model, model_fp16, loader, num_batches=20)
            print(f"  Post-calibration cos: {cos:.6f}")
            return

    # 6. Fine-tune
    if args.mode in ("finetune", "all"):
        if args.checkpoint:
            model.load_state_dict(torch.load(args.checkpoint))
            print(f"  Loaded: {args.checkpoint}")

        model.train()
        model = qat_finetune(model, loader, train_steps=args.train_steps)

        if os.path.exists("qat_best_checkpoint.pt"):
            model.load_state_dict(torch.load("qat_best_checkpoint.pt"))
            print("  Loaded best checkpoint")

        if args.mode == "finetune":
            model_fp16, _ = load_model()
            cos = evaluate_cosine(model, model_fp16, loader, num_batches=30)
            print(f"  Final cos: {cos:.6f}")
            torch.save(model.state_dict(), "qat_finetuned.pt")
            return

    # 7. Convert and export
    if args.mode in ("export", "all"):
        if args.checkpoint:
            model.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
            print(f"  Loaded checkpoint: {args.checkpoint}")
        elif os.path.exists("qat_best_checkpoint.pt"):
            model.load_state_dict(torch.load("qat_best_checkpoint.pt", map_location="cpu"))
            print("  Loaded best checkpoint")
        model.eval()
        # Export FP16 ONNX with QAT-trained weights.
        # Don't use torch.quantization.convert() - PyTorch 2.11's ONNX
        # exporter doesn't support quantized.LinearPackedParamsBase.
        # RKNN will apply do_quantization=True with calibration data.
        state_dict = model.state_dict()
        export_fp16_onnx(model, state_dict, args.output)

    print("\nDone.")


if __name__ == "__main__":
    main()