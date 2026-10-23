"""
sequence_model_tear.py
======================
v18: the crop+sequence network (sequence_model.SequenceUNet, 5-slice bottleneck ConvLSTM) with two
extra outputs, trained from the synthetic tears (labels known exactly because we made them):

  "seg"  (B, 3, H, W)  meniscus segmentation, the SAME head as v16 (this is what gets evaluated)
  "tear" (B, 1, H, W)  tear map logits: where the tear is (painted signal, removed tissue, changed
                       outline, gap + displaced piece; see tear_patterns.synth_pattern)
  "cls"  (B, K)        pattern class logits over dataset_2_5d_tear_patterns.PATTERN_CLASSES:
                       none, inplace_<variant> (7), ragged, lobe, truncate, absent, flap, bucket.
                       p(torn) = 1 - p(none); a mode's probability = sum of its classes.

Tear head: one 3x3 conv on the decoder output (16 ch), like v14's extra heads.
Class head: global average + max pooling of the ConvLSTM state (sees all 5 slices) -> small MLP.
"""
import torch
import torch.nn as nn

from sequence_model import SequenceUNet


class SequenceTearUNet(SequenceUNet):
    def __init__(self, base_unet, n_patterns, bottleneck_channels=512):
        super().__init__(base_unet, bottleneck_channels)
        ch = self.segmentation_head[0].in_channels
        self.tear_head = nn.Conv2d(ch, 1, 3, padding=1)
        self.cls_head = nn.Sequential(nn.Linear(2 * bottleneck_channels, 256), nn.ReLU(inplace=True),
                                      nn.Dropout(0.2), nn.Linear(256, n_patterns))

    def forward(self, x_seq):
        B, T, H, W = x_seq.shape
        c_idx = T // 2
        center_features, bottlenecks = None, []
        for t in range(T):
            feats = self._encode_one(x_seq[:, t:t + 1])
            bottlenecks.append(feats[-1])
            if t == c_idx:
                center_features = feats
        h, c = self.lstm.init_state(B, bottlenecks[0].shape[-2:], bottlenecks[0].device, bottlenecks[0].dtype)
        for t in range(T):
            h, c = self.lstm(bottlenecks[t], (h, c))
        d = self.decoder(center_features[:-1] + [h])
        pooled = torch.cat([h.mean((-2, -1)), h.amax((-2, -1))], 1)
        return {"seg": self.segmentation_head(d), "tear": self.tear_head(d), "cls": self.cls_head(pooled)}


class SegOnly(nn.Module):
    """Makes a SequenceTearUNet look like a plain SequenceUNet (logits only), so the v7-sequence
    eval / dump helpers can be reused unchanged."""
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model(x)["seg"]


def build_sequence_tear_model(ckpt_path, device, n_patterns, init_from=None):
    """TRAINING constructor: same start as v16 / build_sequence_model (5-class DESS baseline, head swapped
    to 3 classes), wrapped with the new heads. init_from: optional trained SequenceUNet checkpoint (e.g.
    v16 ckpt_best.pth) to warm-start the shared part; the new heads always start fresh."""
    import segmentation_models_pytorch as smp
    base = smp.Unet(encoder_name="resnet34", encoder_weights=None, in_channels=3, classes=5)
    base.load_state_dict(torch.load(ckpt_path, map_location=device))
    old = base.segmentation_head[0]
    base.segmentation_head[0] = nn.Conv2d(old.in_channels, 3, kernel_size=old.kernel_size,
                                          stride=old.stride, padding=old.padding)
    model = SequenceTearUNet(base, n_patterns)
    if init_from:
        missing, unexpected = model.load_state_dict(torch.load(init_from, map_location=device), strict=False)
        assert not unexpected, f"unexpected keys in {init_from}: {unexpected[:5]}"
        assert all(k.startswith(("tear_head", "cls_head")) for k in missing), missing[:5]
    return model.to(device)


def load_trained_sequence_tear_model(ckpt_path, device, n_patterns):
    """INFERENCE loader for a trained v18 checkpoint."""
    import segmentation_models_pytorch as smp
    base = smp.Unet(encoder_name="resnet34", encoder_weights=None, in_channels=3, classes=3)
    model = SequenceTearUNet(base, n_patterns)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    return model.to(device)
