"""
model_tear_amodal.py
====================
v14 "tear-amodal" model: the replica smp.Unet (ResNet34, 3-slice 2.5D input)
with its encoder, decoder and 3-class head UNCHANGED, plus optional extra heads
on the same 16-channel decoder output:

  whole    3-class (bg / lateral / medial): the full meniscus wedge, tear ignored.
           This IS the replica's segmentation head (same weights, same role), so
           inference output stays comparable with every earlier run.
  visible  1 channel (logit): the dark meniscus that is actually visible.
  contour  1 channel (logit): thin band on the outer edge of the wedge.

Tear map at inference = p(whole meniscus) - p(visible), clipped at 0.

smp.Unet internals used here (same convention as sequence_model.py, confirmed on
BigRed): features = encoder(x) is a list; decoder(features) takes the LIST.
"""
import torch
import torch.nn as nn
import segmentation_models_pytorch as smp

N_CLASSES = 3
PRETRAINED_N_CLASSES = 5


# [EXPLAIN] WHAT THIS CLASS IS: the replica's network (smp.Unet, ResNet34) wrapped so it can return SEVERAL outputs.
# [EXPLAIN] REPLICA (before): model(x) returned ONE tensor, the 3-class logits (B,3,384,384). Nothing else existed.
# [EXPLAIN] NOW (after):  model(x) returns a dict: 'whole' (the same 3-class logits) plus, only if switched on,
# [EXPLAIN]               'visible' and/or 'contour' (1-channel logits each).
# [EXPLAIN] ARCHITECTURE (unchanged part): x (B,3,384,384) = slices [i-1, i, i+1] -> ResNet34 encoder (5 stages,
# [EXPLAIN]   down to 512ch @ 12x12) -> U-Net decoder with skip connections (up to 16ch @ 384x384) -> head.
# [EXPLAIN] NEW part: extra heads that read the SAME 16-channel decoder output; they add only 145 weights each (16x3x3 + 1).
class TearAmodalUNet(nn.Module):
    def __init__(self, base_unet, visible_head=False, contour_head=False):
        super().__init__()
        # [EXPLAIN] Reuse the replica's encoder and decoder objects as-is (same weights, same pretrained start).
        self.encoder = base_unet.encoder
        self.decoder = base_unet.decoder
        # [EXPLAIN] This is the replica's own 3-class head. It is now called the 'whole' head. The attribute name is kept
        # [EXPLAIN] so a replica checkpoint's keys (segmentation_head.*) still load into this class unchanged.
        self.segmentation_head = base_unet.segmentation_head   # = "whole" head; name kept so
        ch = base_unet.segmentation_head[0].in_channels       # replica checkpoints load as-is
        # [EXPLAIN] R2/R3/R4 only: one 3x3 conv (16 -> 1 channel) predicting 'dark meniscus actually visible' (wedge minus tear).
        # [EXPLAIN] None when the flag is off, so R1 and R1c have NO visible head at all.
        self.visible_head = nn.Conv2d(ch, 1, 3, padding=1) if visible_head else None
        # [EXPLAIN] R3/R4/R1c only: conv 16 -> 1 predicting a thin band on the OUTER edge of the whole wedge.
        # [EXPLAIN] None when the flag is off, so R1 and R2 have NO contour head.
        self.contour_head = nn.Conv2d(ch, 1, 3, padding=1) if contour_head else None

    # [EXPLAIN] REPLICA forward was: return self.segmentation_head(self.decoder(self.encoder(x)))  -> one tensor.
    # [EXPLAIN] Here the shared trunk runs ONCE, then each enabled head reads the same decoder output d.
    def forward(self, x):
        d = self.decoder(self.encoder(x))
        # [EXPLAIN] 'whole' is what you evaluate and what inference writes as *_meniscus_pred.nii.gz (same as every earlier run).
        out = {"whole": self.segmentation_head(d)}
        if self.visible_head is not None:
            out["visible"] = self.visible_head(d)
        if self.contour_head is not None:
            out["contour"] = self.contour_head(d)
        return out


# [EXPLAIN] Same start as the replica: 5-class DESS baseline -> swap the head to 3 classes -> (new) wrap with extra heads.
def build_from_dess_baseline(ckpt_path, device, visible_head=False, contour_head=False):
    """TRAINING constructor, same start as the replica: 5-class DESS baseline,
    head swapped to 3 classes, then wrapped."""
    base = smp.Unet(encoder_name="resnet34", encoder_weights=None,
                    in_channels=3, classes=PRETRAINED_N_CLASSES)
    base.load_state_dict(torch.load(ckpt_path, map_location=device))
    old = base.segmentation_head[0]
    base.segmentation_head[0] = nn.Conv2d(old.in_channels, N_CLASSES, kernel_size=old.kernel_size,
                                          stride=old.stride, padding=old.padding)
    return TearAmodalUNet(base, visible_head, contour_head).to(device)


def build_from_replica(ckpt_path, device, visible_head=False, contour_head=False):
    """WARM-START constructor: a trained 3-class replica checkpoint
    (e.g. run_v7_replica_seed42/ckpt_best.pth). Faster pilot, but not a clean
    one-change comparison against the replica."""
    base = smp.Unet(encoder_name="resnet34", encoder_weights=None,
                    in_channels=3, classes=N_CLASSES)
    base.load_state_dict(torch.load(ckpt_path, map_location=device))
    return TearAmodalUNet(base, visible_head, contour_head).to(device)


def load_trained_tear_model(ckpt_path, device):
    """INFERENCE loader for a finished v14 checkpoint. Which extra heads exist is
    read from the state dict, so the caller does not need the training flags."""
    state = torch.load(ckpt_path, map_location=device)
    base = smp.Unet(encoder_name="resnet34", encoder_weights=None,
                    in_channels=3, classes=N_CLASSES)
    model = TearAmodalUNet(base,
                           visible_head=any(k.startswith("visible_head.") for k in state),
                           contour_head=any(k.startswith("contour_head.") for k in state))
    model.load_state_dict(state)
    return model.to(device).eval()
