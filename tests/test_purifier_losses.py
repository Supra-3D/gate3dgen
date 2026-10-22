"""Numeric equations and a real training-worker forward/backward, with tiny components."""
from types import SimpleNamespace
import pytest
torch = pytest.importorskip("torch")
import torch.nn.functional as F
from gate3dgen.training.purifier_losses import compute_v2_base_components, weighted_v2_base_total, masked_mean


def components(pred, gt, source, mask):
    return compute_v2_base_components(pred, torch.zeros_like(pred), torch.zeros_like(pred),
        torch.zeros_like(pred), torch.ones(1, 1, 1, 1), decoded=pred, gt_pixels=gt,
        degraded_pixels=source, subject_mask=mask, restoration_v3=True)


def test_spatial_losses_match_broadcast_and_adjacent_soft_mask_equations():
    pred = torch.arange(3 * 9 * 9, dtype=torch.float32).reshape(1, 3, 9, 9) / 243
    pred.requires_grad_()
    gt = torch.zeros_like(pred)
    source = torch.ones_like(pred)
    mask = torch.linspace(0, 1, 81).reshape(1, 1, 9, 9)
    c = components(pred, gt, source, mask)
    weighted = lambda a, w: (a.abs() * w).sum() / w.expand_as(a).sum().clamp_min(1)
    residual = F.avg_pool2d(torch.full_like(mask, 0.5), 7, 1, 3)
    dx, dy = pred[..., :, 1:] - pred[..., :, :-1], pred[..., 1:, :] - pred[..., :-1, :]
    expected_edge = .5 * (weighted(dx, mask[..., :, 1:] * mask[..., :, :-1])
                         + weighted(dy, mask[..., 1:, :] * mask[..., :-1, :]))
    assert torch.allclose(c["pixel_subject_l1"], weighted(pred, mask))
    assert torch.allclose(c["pixel_affected_l1"], weighted(pred, residual))
    assert torch.allclose(c["affected_mask_mean"], residual.mean())
    assert torch.allclose(c["pixel_subject_gradient"], expected_edge)
    for name in ("latent_x0", "pixel_full_l1", "pixel_subject_l1", "pixel_affected_l1", "pixel_subject_gradient"):
        gradient, = torch.autograd.grad(c[name], pred, retain_graph=True)
        assert torch.isfinite(gradient).all() and gradient.abs().sum() > 0
    assert masked_mean(pred, torch.zeros_like(mask)).item() == 0


def test_worker_runs_endpoint_forward_backward_and_optimizer_update():
    pytest.importorskip("diffusers")
    from PIL import Image
    from diffusers import FlowMatchEulerDiscreteScheduler
    from gate3dgen.training.vendor.train.src.forward_step import forward_step_impl
    class TinyVAE:
        device, dtype, temperal_downsample = "cpu", torch.float32, []
        def encode(self, value):
            return (SimpleNamespace(mode=lambda: value),)
        def decode(self, value, return_dict=False):
            return (value,)
    class TinyTransformer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = torch.nn.Linear(12, 12)
        def forward(self, hidden_states, **kwargs):
            return (self.proj(hidden_states),)
    args = SimpleNamespace(seed=42, condition_encoder_mode="offline", vae_encode_mode="mode",
        x0_pixel_loss=True, x0_sigma_band="1.0,1.0", train_sampling_steps=10,
        weighting_scheme="none", restoration_v3_loss=True, pixel_lpips_weight=.5,
        pixel_l1_weight=0., flow_loss_weight=0., latent_x0_weight=.1,
        pixel_full_l1_weight=1., pixel_subject_l1_weight=1., pixel_affected_l1_weight=1.,
        pixel_subject_gradient_weight=1.)
    class TinyPerceptual(torch.nn.Module):
        def forward(self, a, b):
            return (a-b).square().mean((1, 2, 3), keepdim=True)
    extra = {"noise_scheduler": FlowMatchEulerDiscreteScheduler(), "idx_sampling": None,
             "latents_mean": torch.zeros(1, 3, 1, 1, 1), "latents_std": torch.ones(1, 3, 1, 1, 1),
             "processor": None, "lpips_model": TinyPerceptual()}
    batch = {"pixel_values": torch.zeros(1, 3, 16, 16),
             "source_pixel_values": torch.ones(1, 3, 16, 16),
             "source_images_transposed": [[Image.new("RGB", (16, 16), "white")]],
             "encoder_attention_mask": torch.ones(1, 4), "encoder_hidden_states": torch.ones(1, 4, 8),
             "text": ["purify"], "subject_mask_values": torch.ones(1, 1, 16, 16)}
    model = TinyTransformer()
    before = model.proj.weight.detach().clone()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss = forward_step_impl(args, 0, model, TinyVAE(), None, extra, batch, torch.float32, "cpu")
    loss.backward()
    assert torch.isfinite(loss) and model.proj.weight.grad.abs().sum() > 0
    optimizer.step()
    assert not torch.equal(before, model.proj.weight)
    assert extra["loss_components"]["sigma_mean"] == 1
    assert extra["loss_components"]["pixel_lpips"] > 0
