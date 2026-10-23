"""Small CUDA regression cases for independent encoder/decoder padding."""
import json
from pathlib import Path
from types import SimpleNamespace

import torch
from compare_attention import BACKENDS, register_cross_attention_fix
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS


def main():
    records = []
    for backend in ("fa2", "fa3"):
        register_cross_attention_fix(backend)
        fn = ALL_ATTENTION_FUNCTIONS[BACKENDS[backend]]
        module = SimpleNamespace(is_decoder=True, is_causal=False,
                                 config=SimpleNamespace(_attn_implementation=BACKENDS[backend]))
        for qlen in (1, 8, 24, 40):
            for side in ("left", "right"):
                torch.manual_seed(42)
                q = torch.randn(3, 8, qlen, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
                k = torch.randn(3, 8, 24, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
                v = torch.randn_like(k, requires_grad=True)
                pos = torch.arange(24, device="cuda")[None, :]
                lengths = torch.tensor([3, 24, 13], device="cuda")[:, None]
                mask = pos < lengths if side == "right" else pos >= 24-lengths
                ref = torch.nn.functional.scaled_dot_product_attention(
                    q, k, v, attn_mask=mask[:, None, None, :]).transpose(1, 2)
                actual, _ = fn(module, q, k, v, mask, scaling=128**-.5)
                upstream = torch.randn_like(ref)
                rg = torch.autograd.grad(ref, [q, k, v], upstream, retain_graph=True)
                ag = torch.autograd.grad(actual, [q, k, v], upstream)
                errors = [((a.float()-r.float()).norm()/r.float().norm()).item() for a, r in zip(ag, rg)]
                error = ((actual.float()-ref.float()).norm()/ref.float().norm()).item()
                assert error < .01 and max(errors) < .02, (backend, qlen, side, error, errors)
                records.append(dict(backend=backend, q=qlen, k=24, padding=side,
                                    output_relative_l2=error, gradient_relative_l2=errors))
    path = Path(__file__).with_name("cross_attention_cases.json")
    path.write_text(json.dumps(records, indent=2))
    print(f"PASS: {len(records)} cases; {path}")


if __name__ == "__main__":
    main()
