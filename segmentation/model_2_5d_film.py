"""
model_2_5d_film.py
====================
Phase 5, E2 on the PLAIN 2.5D architecture -- independent of crop+sequence.

The crop+sequence version of E2 (sequence_model_film.py) gates the ConvLSTM
hidden state AFTER EACH of 5 timesteps, because the whole point there was
letting position change *how neighbours get combined* across a recurrence.
The plain 2.5D architecture has no recurrence and no per-timestep hidden state
to gate -- so this is a genuinely different formulation, not a copy-paste port,
noted up front rather than force-fitting the crop+sequence design where it
doesn't apply.

What this does instead: the SAME FiLM mechanism (zero-init gate: gamma, beta =
MLP(pos); feat * (1+gamma) + beta) applied ONCE, to the single bottleneck
feature map of a standard non-recurrent smp.Unet, conditioned on the CENTER
slice's own position -- injecting position where the network has semantic
features, right before the decoder, instead of at the raw input (that's E1).

Reuses PosFiLM from sequence_model_film.py verbatim (architecture-agnostic:
it only needs a channel count) rather than reimplementing it.
"""
import torch
import torch.nn as nn

from sequence_model_film import PosFiLM


class Unet2_5DFiLM(nn.Module):
    """forward(x, pos): x (B,3,H,W), pos (B,) float in [0,1]."""

    def __init__(self, base_unet, bottleneck_channels=512, hidden=64):
        super().__init__()
        self.encoder = base_unet.encoder
        self.decoder = base_unet.decoder
        self.segmentation_head = base_unet.segmentation_head
        self.film = PosFiLM(bottleneck_channels, hidden)

    def forward(self, x, pos):
        features = list(self.encoder(x))
        features[-1] = self.film(features[-1], pos.view(-1, 1).to(features[-1].dtype))
        decoder_out = self.decoder(features)
        return self.segmentation_head(decoder_out)

    def position_weight_norm(self):
        """Named to match E1's method so the trainer can log either uniformly."""
        return self.film.gate_weight_norm()


def build_2_5d_film_model(ckpt_path, device, pretrained_n_classes=5, n_classes=3, hidden=64):
    """TRAINING-TIME constructor. Same baseline checkpoint as replica/P1/P2/P4."""
    import segmentation_models_pytorch as smp

    base = smp.Unet(encoder_name="resnet34", encoder_weights=None,
                    in_channels=3, classes=pretrained_n_classes)
    base.load_state_dict(torch.load(ckpt_path, map_location=device))

    old_head = base.segmentation_head[0]
    base.segmentation_head[0] = nn.Conv2d(
        old_head.in_channels, n_classes,
        kernel_size=old_head.kernel_size, stride=old_head.stride, padding=old_head.padding)

    return Unet2_5DFiLM(base, bottleneck_channels=512, hidden=hidden).to(device)


def load_trained_2_5d_film_model(ckpt_path, device, n_classes=3, hidden=64):
    """INFERENCE-TIME loader for an already-trained checkpoint."""
    import segmentation_models_pytorch as smp

    base = smp.Unet(encoder_name="resnet34", encoder_weights=None,
                    in_channels=3, classes=n_classes)
    model = Unet2_5DFiLM(base, bottleneck_channels=512, hidden=hidden)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    return model.to(device)


@torch.no_grad()
def verify_zero_init(device="cpu", atol=1e-5, seed=0, verbose=True):
    """Proves the FiLM model is numerically IDENTICAL to plain smp.Unet at
    step 0, for arbitrary position values."""
    import segmentation_models_pytorch as smp

    torch.manual_seed(seed)
    plain = smp.Unet("resnet34", encoder_weights=None, in_channels=3, classes=3).to(device).eval()

    base_film = smp.Unet("resnet34", encoder_weights=None, in_channels=3, classes=3)
    base_film.load_state_dict(plain.state_dict())
    film_model = Unet2_5DFiLM(base_film, bottleneck_channels=512).to(device).eval()

    x = torch.randn(2, 3, 64, 64, device=device)
    pos = torch.rand(2, device=device)

    max_diff = float((plain(x) - film_model(x, pos)).abs().max())
    ok = max_diff < atol
    if verbose:
        print(f"2.5D E2 zero-init check: max|plain - film| = {max_diff:.3e}  (tol {atol})  "
              f"-> {'PASS' if ok else 'FAIL'}")
        print(f"FiLM gate weight norm at init: {film_model.position_weight_norm():.3e} (must be 0)")
    return ok, max_diff


if __name__ == "__main__":
    ok, _ = verify_zero_init()
    raise SystemExit(0 if ok else 1)
