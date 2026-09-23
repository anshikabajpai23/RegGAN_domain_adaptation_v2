"""
model_2_5d_pos.py
==================
Phase 5, E1 on the PLAIN 2.5D architecture -- independent of crop+sequence,
so this answers "does position help the actual replica recipe" rather than
"does position help on top of two other already-stacked changes."

Same idea as sequence_model_pos.py (append slice position as a 4th input
channel, zero-init so the model starts numerically identical to the plain
3-channel baseline), applied to the STANDARD non-recurrent smp.Unet used by
finetune_meniscus_v7_replica.py instead of the 5-slice ConvLSTM SequenceUNet.

Difference from the crop+sequence version: no per-timestep loop, no ConvLSTM --
position is a single scalar per sample here (the CENTER slice's own position,
the same index the 3-channel [i-1,i,i+1] stack is built around), appended as
one extra channel.

Reuses widen_first_conv_zero_init from sequence_model_pos.py verbatim -- it is
architecture-agnostic (only needs an encoder with a .conv1 attribute), so there
is no reason to duplicate it.
"""
import torch
import torch.nn as nn

from sequence_model_pos import widen_first_conv_zero_init


class Unet2_5DPos(nn.Module):
    """Wraps a plain smp.Unet; widens conv1 to 4 channels at zero-init.
    forward(x, pos): x (B,3,H,W), pos (B,) float in [0,1]."""

    def __init__(self, base_unet):
        super().__init__()
        self.encoder = base_unet.encoder
        self.decoder = base_unet.decoder
        self.segmentation_head = base_unet.segmentation_head
        widen_first_conv_zero_init(self.encoder, extra_channels=1)

    def forward(self, x, pos):
        pos_map = pos.view(-1, 1, 1, 1).expand(-1, 1, x.shape[2], x.shape[3]).to(x.dtype)
        x4 = torch.cat([x, pos_map], dim=1)
        features = self.encoder(x4)
        decoder_out = self.decoder(features)      # UnetDecoder.forward(features: list)
        return self.segmentation_head(decoder_out)

    def position_weight_norm(self):
        """Starts at exactly 0. Still ~0 after training means the model found
        no use for position at the input on the plain architecture either."""
        return float(self.encoder.conv1.weight[:, 3].norm())


def build_2_5d_pos_model(ckpt_path, device, pretrained_n_classes=5, n_classes=3):
    """TRAINING-TIME constructor. Same baseline checkpoint as replica/P1/P2/P4."""
    import segmentation_models_pytorch as smp

    base = smp.Unet(encoder_name="resnet34", encoder_weights=None,
                    in_channels=3, classes=pretrained_n_classes)
    base.load_state_dict(torch.load(ckpt_path, map_location=device))

    old_head = base.segmentation_head[0]
    base.segmentation_head[0] = nn.Conv2d(
        old_head.in_channels, n_classes,
        kernel_size=old_head.kernel_size, stride=old_head.stride, padding=old_head.padding)

    return Unet2_5DPos(base).to(device)


def load_trained_2_5d_pos_model(ckpt_path, device, n_classes=3):
    """INFERENCE-TIME loader for an already-trained checkpoint (4-ch conv1,
    3-class head already present)."""
    import segmentation_models_pytorch as smp

    base = smp.Unet(encoder_name="resnet34", encoder_weights=None,
                    in_channels=3, classes=n_classes)
    model = Unet2_5DPos(base)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    return model.to(device)


@torch.no_grad()
def verify_zero_init(device="cpu", atol=1e-5, seed=0, verbose=True):
    """Proves the position model is numerically IDENTICAL to plain smp.Unet
    at step 0, for arbitrary position values."""
    import segmentation_models_pytorch as smp

    torch.manual_seed(seed)
    plain = smp.Unet("resnet34", encoder_weights=None, in_channels=3, classes=3).to(device).eval()

    base_pos = smp.Unet("resnet34", encoder_weights=None, in_channels=3, classes=3)
    base_pos.load_state_dict(plain.state_dict())
    pos_model = Unet2_5DPos(base_pos).to(device).eval()

    x = torch.randn(2, 3, 64, 64, device=device)
    pos = torch.rand(2, device=device)

    max_diff = float((plain(x) - pos_model(x, pos)).abs().max())
    ok = max_diff < atol
    if verbose:
        print(f"2.5D E1 zero-init check: max|plain - pos| = {max_diff:.3e}  (tol {atol})  "
              f"-> {'PASS' if ok else 'FAIL'}")
        print(f"position-channel weight norm at init: {pos_model.position_weight_norm():.3e} (must be 0)")
    return ok, max_diff


if __name__ == "__main__":
    ok, _ = verify_zero_init()
    raise SystemExit(0 if ok else 1)
