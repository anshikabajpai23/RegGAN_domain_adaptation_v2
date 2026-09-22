"""
sequence_model.py
==================
Slice-sequence architecture (Anshika's candidate approach) — 5-slice,
bottleneck-only ConvLSTM aggregation over the existing smp.Unet(resnet34).

Design, and why:
  - Check 2 showed the model ignores neighbors at FIXED channel positions
    (variant B [i,i,i] ~= variant A [i-1,i,i+1]; variant C [i-2,i,i+2] hurt
    17/17). That result is about a model that never had to LEARN to combine
    neighbors — they were just concatenated as extra input channels. This
    architecture gives the model an actual mechanism to learn how much
    weight each neighboring slice's features deserve, instead of baking
    z-context into fixed channel positions.
  - 5 slices (i-2..i+2), matching P4's wider-context idea, but here the
    neighbors are each encoded SEPARATELY (not concatenated as channels)
    and combined by a learned recurrent step at the bottleneck only. Skip
    connections stay from the CENTER slice's own encoder pass, untouched —
    fine spatial detail (where the actual thin-structure fragmentation
    happens) is not temporally blurred, only the deep semantic feature is.

smp.Unet internal interface, CONFIRMED against a real BigRed run (2026-09-12,
first attempt crashed with exactly this mismatch — TypeError: UnetDecoder.
forward() takes 2 positional arguments but 7 were given):
    features = self.encoder(x)          # list, shallow -> deep, features[-1] = bottleneck
    decoder_out = self.decoder(features)   # takes the LIST itself, NOT *features
    logits = self.segmentation_head(decoder_out)
The encoder-side assumption (features is a shallow->deep list, [-1] is the
bottleneck) held on first try and needed no fix — only the decoder call
convention was wrong.

Checkpoint compatibility: baseline_best_model.pth's first conv layer expects
3 input channels. Each of the 5 slices here is a single grayscale image,
replicated to 3 identical channels before encoding (same trick used to feed
grayscale into RGB-pretrained backbones) — this is what makes starting from
the SAME baseline checkpoint possible with zero architecture surgery on the
input layer, consistent with every other Phase 4 run.
"""
import torch
import torch.nn as nn


class ConvLSTMCell(nn.Module):
    """Standard convolutional LSTM cell (Shi et al. 2015 formulation).
    Operates on (B, C, H, W) feature maps; hidden/cell state same shape as input.
    """
    def __init__(self, channels, kernel_size=3):
        super().__init__()
        pad = kernel_size // 2
        # single conv producing all 4 gates at once (input, forget, output, candidate)
        self.conv = nn.Conv2d(channels * 2, channels * 4, kernel_size, padding=pad)
        self.channels = channels

    def forward(self, x, state):
        h, c = state
        gates = self.conv(torch.cat([x, h], dim=1))
        i, f, o, g = gates.chunk(4, dim=1)
        i, f, o = torch.sigmoid(i), torch.sigmoid(f), torch.sigmoid(o)
        g = torch.tanh(g)
        c_new = f * c + i * g
        h_new = o * torch.tanh(c_new)
        return h_new, c_new

    def init_state(self, batch_size, spatial_shape, device, dtype):
        h = torch.zeros(batch_size, self.channels, *spatial_shape, device=device, dtype=dtype)
        c = torch.zeros(batch_size, self.channels, *spatial_shape, device=device, dtype=dtype)
        return h, c


class SequenceUNet(nn.Module):
    """
    Wraps an existing smp.Unet. Input: (B, T, H, W) — T=5 grayscale slices,
    center at index T//2. Each timestep is replicated to 3 channels and run
    through the SAME (shared-weight) encoder. A ConvLSTM scans the sequence
    of bottleneck features left-to-right; the hidden state after the LAST
    step (having seen all 5) is used as the aggregated bottleneck for the
    center slice's decoder pass. Skip connections (features[:-1]) come from
    the CENTER slice's own encoder pass, unmodified.
    """
    def __init__(self, base_unet, bottleneck_channels=512):
        super().__init__()
        self.encoder = base_unet.encoder
        self.decoder = base_unet.decoder
        self.segmentation_head = base_unet.segmentation_head
        self.lstm = ConvLSTMCell(bottleneck_channels)

    def _encode_one(self, x_1ch):
        x3 = x_1ch.repeat(1, 3, 1, 1)   # (B,1,H,W) -> (B,3,H,W), matches baseline's conv1
        return self.encoder(x3)         # list of feature maps, shallow -> deep

    def forward(self, x_seq):
        """x_seq: (B, T, H, W), single-channel slices, T odd, center = T//2."""
        B, T, H, W = x_seq.shape
        center_idx = T // 2

        center_features = None
        bottlenecks = []
        for t in range(T):
            feats = self._encode_one(x_seq[:, t:t+1])
            bottlenecks.append(feats[-1])
            if t == center_idx:
                center_features = feats   # keep full skip-connection list from center slice

        device, dtype = bottlenecks[0].device, bottlenecks[0].dtype
        spatial = bottlenecks[0].shape[-2:]
        h, c = self.lstm.init_state(B, spatial, device, dtype)
        for t in range(T):
            h, c = self.lstm(bottlenecks[t], (h, c))
        # h after all T steps = aggregated bottleneck, replaces the center's own
        aggregated_features = center_features[:-1] + [h]

        decoder_out = self.decoder(aggregated_features)   # UnetDecoder.forward(features: list), NOT *features
        logits = self.segmentation_head(decoder_out)
        return logits


def build_sequence_model(ckpt_path, device, pretrained_n_classes=5, n_classes=3):
    """TRAINING-TIME constructor. Loads baseline_best_model.pth -- the UNTRAINED
    5-class DESS checkpoint, a plain smp.Unet with no LSTM -- into a standard
    3-class-headed smp.Unet (same as every other Phase 4 build_model()), then
    wraps it for sequence processing. Bottleneck channel count (512) matches
    resnet34's standard smp encoder output -- verify this against the actual
    model if the encoder ever changes.

    Do NOT use this to load an already-trained run_v7_sequence/ckpt_best.pth --
    that checkpoint's state_dict is the full SequenceUNet (3-class head AND
    lstm.* weights already present), not a plain 5-class Unet. Loading it here
    fails with "Unexpected key(s): lstm.conv.weight/bias" plus a 3-vs-5 head
    shape mismatch (confirmed against a real BigRed run, 2026-09-12).
    Use load_trained_sequence_model() below for that case instead.
    """
    import segmentation_models_pytorch as smp

    base = smp.Unet(encoder_name="resnet34", encoder_weights=None,
                    in_channels=3, classes=pretrained_n_classes)
    state = torch.load(ckpt_path, map_location=device)
    base.load_state_dict(state)

    old_head = base.segmentation_head[0]
    base.segmentation_head[0] = nn.Conv2d(
        old_head.in_channels, n_classes,
        kernel_size=old_head.kernel_size, stride=old_head.stride, padding=old_head.padding)

    model = SequenceUNet(base, bottleneck_channels=512)
    return model.to(device)


def load_trained_sequence_model(ckpt_path, device, n_classes=3):
    """INFERENCE-TIME loader for an already-trained run_v7_sequence/ckpt_best.pth.
    Builds the SequenceUNet shape directly with the FINAL head size (n_classes),
    then loads the full state_dict (3-class head + lstm.* weights already
    present) into the wrapped model itself -- not into a bare smp.Unet the way
    build_sequence_model() does for the untrained baseline case above."""
    import segmentation_models_pytorch as smp

    base = smp.Unet(encoder_name="resnet34", encoder_weights=None,
                    in_channels=3, classes=n_classes)
    model = SequenceUNet(base, bottleneck_channels=512)

    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    return model.to(device)
