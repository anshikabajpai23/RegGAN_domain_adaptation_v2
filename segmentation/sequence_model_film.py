"""
sequence_model_film.py
=======================
Phase 5, E2 — slice position injected as GATING at the bottleneck (FiLM),
on the crop+sequence base.

E1 vs E2, the actual difference:
  E1 appends position as a 4th INPUT channel -- the network has to carry that
     information all the way up through the encoder itself.
  E2 injects it where the network already has semantic features, as a learned
     per-channel scale and shift on the ConvLSTM hidden state.

Doc §2.3 specifies FiLM on the HIDDEN STATE rather than on the aggregator input,
and gives the reason: it lets position change *how neighbouring slices are
combined*, which is the whole point of having a recurrent aggregator. Applied at
every timestep with that timestep's own position.

    gamma, beta = MLP(pos)
    h <- h * (1 + gamma) + beta

ZERO-INIT: the MLP's final layer is zeroed, so gamma = beta = 0 and FiLM is
exactly the identity at step 0. The model is therefore NUMERICALLY IDENTICAL to
the crop+sequence baseline before training, and any difference can only come
from training. verify_zero_init_film() proves it; the trainer refuses to run if
it fails.

Interface facts verified empirically against smp 0.5.0, same as E1:
  encoder.out_channels[-1] == 512 for resnet34, matching SequenceUNet's default.
"""
import torch
import torch.nn as nn

from sequence_model import SequenceUNet


class PosFiLM(nn.Module):
    """gamma, beta = MLP(pos); feat * (1 + gamma) + beta.
    Final layer zero-initialised, so this is the identity at step 0."""

    def __init__(self, channels, hidden=64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(1, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 2 * channels),
        )
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)
        self.channels = channels

    def forward(self, feat, pos):
        """feat (B,C,h,w); pos (B,1) float in [0,1]."""
        gamma, beta = self.mlp(pos).chunk(2, dim=1)
        return feat * (1 + gamma[:, :, None, None]) + beta[:, :, None, None]

    def gate_weight_norm(self):
        """L2 norm of the output layer. Starts at exactly 0. Still ~0 after
        training means the model found no use for position at the bottleneck --
        a real finding, the counterpart of E1's pos_wnorm."""
        return float(self.mlp[-1].weight.norm())


class SequenceUNetFiLM(SequenceUNet):
    """SequenceUNet whose ConvLSTM hidden state is FiLM-modulated by position
    at each timestep. Input stays 3-channel: unlike E1, nothing is appended to
    the image.

    forward(x_seq, pos_seq):
        x_seq   (B, T, H, W)
        pos_seq (B, T)   position in [0,1] of the slice at each timestep
    """

    def __init__(self, base_unet, bottleneck_channels=512, hidden=64):
        super().__init__(base_unet, bottleneck_channels)
        self.film = PosFiLM(bottleneck_channels, hidden)

    def forward(self, x_seq, pos_seq):
        B, T, H, W = x_seq.shape
        center_idx = T // 2

        center_features = None
        bottlenecks = []
        for t in range(T):
            feats = self._encode_one(x_seq[:, t:t + 1])
            bottlenecks.append(feats[-1])
            if t == center_idx:
                center_features = feats

        device, dtype = bottlenecks[0].device, bottlenecks[0].dtype
        spatial = bottlenecks[0].shape[-2:]
        h, c = self.lstm.init_state(B, spatial, device, dtype)
        for t in range(T):
            h, c = self.lstm(bottlenecks[t], (h, c))
            # position modulates the aggregated state AFTER each step, so it can
            # change how the next neighbour is folded in (doc §2.3)
            h = self.film(h, pos_seq[:, t:t + 1].to(dtype))

        aggregated_features = center_features[:-1] + [h]
        decoder_out = self.decoder(aggregated_features)
        return self.segmentation_head(decoder_out)

    def position_weight_norm(self):
        """Named to match E1's method so the trainer can log either uniformly."""
        return self.film.gate_weight_norm()


def build_sequence_film_model(ckpt_path, device, pretrained_n_classes=5, n_classes=3, hidden=64):
    """TRAINING-TIME constructor. Same baseline checkpoint as every other run."""
    import segmentation_models_pytorch as smp

    base = smp.Unet(encoder_name="resnet34", encoder_weights=None,
                    in_channels=3, classes=pretrained_n_classes)
    base.load_state_dict(torch.load(ckpt_path, map_location=device))

    old_head = base.segmentation_head[0]
    base.segmentation_head[0] = nn.Conv2d(
        old_head.in_channels, n_classes,
        kernel_size=old_head.kernel_size, stride=old_head.stride, padding=old_head.padding)

    return SequenceUNetFiLM(base, bottleneck_channels=512, hidden=hidden).to(device)


def load_trained_sequence_film_model(ckpt_path, device, n_classes=3, hidden=64):
    """INFERENCE-TIME loader for a trained run_e2_film checkpoint."""
    import segmentation_models_pytorch as smp

    base = smp.Unet(encoder_name="resnet34", encoder_weights=None,
                    in_channels=3, classes=n_classes)
    model = SequenceUNetFiLM(base, bottleneck_channels=512, hidden=hidden)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    return model.to(device)


@torch.no_grad()
def verify_zero_init_film(device="cpu", atol=1e-5, seed=0, verbose=True):
    """Proves the FiLM model is numerically IDENTICAL to the plain SequenceUNet
    at step 0, for arbitrary position values."""
    import segmentation_models_pytorch as smp

    torch.manual_seed(seed)
    base_plain = smp.Unet("resnet34", encoder_weights=None, in_channels=3, classes=3)
    plain = SequenceUNet(base_plain, bottleneck_channels=512).to(device).eval()

    base_film = smp.Unet("resnet34", encoder_weights=None, in_channels=3, classes=3)
    base_film.load_state_dict(base_plain.state_dict())
    film_model = SequenceUNetFiLM(base_film, bottleneck_channels=512).to(device).eval()
    film_model.lstm.load_state_dict(plain.lstm.state_dict())

    x = torch.randn(2, 5, 64, 64, device=device)
    pos = torch.rand(2, 5, device=device)

    max_diff = float((plain(x) - film_model(x, pos)).abs().max())
    ok = max_diff < atol
    if verbose:
        print(f"FiLM zero-init check: max|plain - film| = {max_diff:.3e}  (tol {atol})  "
              f"-> {'PASS' if ok else 'FAIL'}")
        print(f"FiLM gate weight norm at init: {film_model.position_weight_norm():.3e} (must be 0)")
    return ok, max_diff


if __name__ == "__main__":
    ok, _ = verify_zero_init_film()
    raise SystemExit(0 if ok else 1)
