"""
sequence_model_pos.py
======================
Phase 5, E1 — slice position as an extra input channel, on the crop+sequence base.

WHAT IT ADDS
Each of the 5 timesteps currently goes in as one grayscale slice replicated to 3
channels (SequenceUNet._encode_one). E1 appends a 4th channel holding a single
number: where that slice sits along the scan axis. The model currently decides
"is the meniscus still here?" from in-plane appearance alone; this is the one
piece of through-plane information that survived check 2.

The position feature (doc §1.2), symmetric on purpose:
    z_frac = slice_idx / (n_slices - 1)      # 0..1
    pos    = 2 * abs(z_frac - 0.5)           # 0 at volume centre (notch), 1 at outer edges
Raw z_frac flips meaning between left and right knees; pos does not. It also
lands 0 and 1 on exactly the two places the errors live: the notch and the
outer cliffs.

Each timestep gets ITS OWN position, not the centre's, so the model also sees
how the window is spaced and whether it is clamped at a volume boundary.

ZERO-INIT (the point of this design)
The 4th input channel starts at exactly zero weight, so at step 0 the model is
NUMERICALLY IDENTICAL to the crop+sequence baseline. Any difference can then
only come from training, not from re-initialised weights. verify_zero_init()
below proves it; the training script calls it before the first epoch.

Do NOT build this with smp.Unet(in_channels=4) instead: smp re-initialises the
first conv by tiling and rescaling, which silently changes the baseline.

INTERFACE FACTS, verified empirically against smp 0.5.0 (not assumed):
  - encoder.conv1 is the first conv: in=3, out=64, k=7x7, stride=2, pad=3, bias=False
  - encoder.forward() routes through self.conv1
  - encoder.out_channels = [3, 64, 64, 128, 256, 512] -> bottleneck 512
  - encoder returns features[0] == the RAW INPUT, but UnetDecoder IGNORES it,
    so widening the input to 4 channels does not break the decoder (tested).
"""
import torch
import torch.nn as nn

from sequence_model import SequenceUNet


def widen_first_conv_zero_init(encoder, extra_channels=1):
    """3 -> 3+extra input channels on encoder.conv1, new channels at ZERO weight.
    Pretrained weights for the original channels are copied across untouched."""
    conv = encoder.conv1
    in_old = conv.in_channels
    new = nn.Conv2d(in_old + extra_channels, conv.out_channels,
                    kernel_size=conv.kernel_size, stride=conv.stride,
                    padding=conv.padding, dilation=conv.dilation,
                    groups=conv.groups, bias=conv.bias is not None)
    with torch.no_grad():
        new.weight.zero_()
        new.weight[:, :in_old] = conv.weight
        if conv.bias is not None:
            new.bias.copy_(conv.bias)
    encoder.conv1 = new
    return encoder


class SequenceUNetPos(SequenceUNet):
    """SequenceUNet with a per-timestep position channel.

    forward(x_seq, pos_seq):
        x_seq   (B, T, H, W)  single-channel slices, centre at T//2
        pos_seq (B, T)        position in [0,1] for the slice at each timestep
    """

    def __init__(self, base_unet, bottleneck_channels=512):
        super().__init__(base_unet, bottleneck_channels)
        widen_first_conv_zero_init(self.encoder, extra_channels=1)

    def _encode_one_pos(self, x_1ch, pos_t):
        """x_1ch (B,1,H,W); pos_t (B,) -> encoder features from a 4-channel input."""
        x3 = x_1ch.repeat(1, 3, 1, 1)
        pos_map = pos_t.view(-1, 1, 1, 1).expand(-1, 1, x3.shape[2], x3.shape[3])
        return self.encoder(torch.cat([x3, pos_map.to(x3.dtype)], dim=1))

    def forward(self, x_seq, pos_seq):
        B, T, H, W = x_seq.shape
        center_idx = T // 2

        center_features = None
        bottlenecks = []
        for t in range(T):
            feats = self._encode_one_pos(x_seq[:, t:t + 1], pos_seq[:, t])
            bottlenecks.append(feats[-1])
            if t == center_idx:
                center_features = feats

        device, dtype = bottlenecks[0].device, bottlenecks[0].dtype
        spatial = bottlenecks[0].shape[-2:]
        h, c = self.lstm.init_state(B, spatial, device, dtype)
        for t in range(T):
            h, c = self.lstm(bottlenecks[t], (h, c))

        aggregated_features = center_features[:-1] + [h]
        decoder_out = self.decoder(aggregated_features)
        return self.segmentation_head(decoder_out)

    def position_weight_norm(self):
        """L2 norm of the position channel's weights. Starts at exactly 0.
        If it is still ~0 after training, the model found no use for position
        (doc §1.7) -- that is a real finding, not a bug."""
        return float(self.encoder.conv1.weight[:, 3].norm())


def build_sequence_pos_model(ckpt_path, device, pretrained_n_classes=5, n_classes=3):
    """TRAINING-TIME constructor. Same baseline checkpoint as every other run;
    the position channel is added on top, at zero weight."""
    import segmentation_models_pytorch as smp

    base = smp.Unet(encoder_name="resnet34", encoder_weights=None,
                    in_channels=3, classes=pretrained_n_classes)
    state = torch.load(ckpt_path, map_location=device)
    base.load_state_dict(state)

    old_head = base.segmentation_head[0]
    base.segmentation_head[0] = nn.Conv2d(
        old_head.in_channels, n_classes,
        kernel_size=old_head.kernel_size, stride=old_head.stride, padding=old_head.padding)

    model = SequenceUNetPos(base, bottleneck_channels=512)
    return model.to(device)


def load_trained_sequence_pos_model(ckpt_path, device, n_classes=3):
    """INFERENCE-TIME loader for a trained run_e1_pos checkpoint (4-ch conv1,
    3-class head, lstm.* all already present)."""
    import segmentation_models_pytorch as smp

    base = smp.Unet(encoder_name="resnet34", encoder_weights=None,
                    in_channels=3, classes=n_classes)
    model = SequenceUNetPos(base, bottleneck_channels=512)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    return model.to(device)


@torch.no_grad()
def verify_zero_init(device="cpu", atol=1e-5, seed=0, verbose=True):
    """Proves the position model is numerically IDENTICAL to the plain
    SequenceUNet at step 0, for arbitrary position values. Returns (ok, max_diff).

    If this fails, E1's result is uninterpretable: a difference could come from
    re-initialised weights rather than from the position information."""
    import segmentation_models_pytorch as smp

    torch.manual_seed(seed)
    base_plain = smp.Unet("resnet34", encoder_weights=None, in_channels=3, classes=3)
    plain = SequenceUNet(base_plain, bottleneck_channels=512).to(device).eval()

    # same weights, then widened -> only the zero 4th channel differs
    base_pos = smp.Unet("resnet34", encoder_weights=None, in_channels=3, classes=3)
    base_pos.load_state_dict(base_plain.state_dict())
    pos_model = SequenceUNetPos(base_pos, bottleneck_channels=512).to(device).eval()
    pos_model.lstm.load_state_dict(plain.lstm.state_dict())

    x = torch.randn(2, 5, 64, 64, device=device)
    pos = torch.rand(2, 5, device=device)          # arbitrary, must not matter at step 0

    out_plain = plain(x)
    out_pos = pos_model(x, pos)
    max_diff = float((out_plain - out_pos).abs().max())
    ok = max_diff < atol

    if verbose:
        print(f"zero-init check: max|plain - pos| = {max_diff:.3e}  "
              f"(tol {atol})  -> {'PASS' if ok else 'FAIL'}")
        print(f"position-channel weight norm at init: {pos_model.position_weight_norm():.3e} (must be 0)")
    return ok, max_diff


if __name__ == "__main__":
    ok, _ = verify_zero_init()
    raise SystemExit(0 if ok else 1)
