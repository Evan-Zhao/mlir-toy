"""Reproducible operator cases and initialized kernel arguments."""

from dataclasses import asdict, dataclass

from neptune_mlir.testing import (
    make_attn_inputs,
    make_mamba_inputs,
    reference_attn,
    reference_mamba,
)


@dataclass(frozen=True)
class Case:
    operator: str
    batch: int = 1
    seq_len: int = 2048
    heads: int = 8
    head_dim: int = 64
    channels: int = 1536
    state_dim: int = 16
    dtype: str = "bfloat16"
    block_m: int = 128
    block_n: int = 64
    block_channels: int = 128

    def validate(self):
        from neptune_mlir.schedules import AttentionTileConfig, MambaTileConfig

        if self.operator not in {"attn", "mamba"}:
            raise ValueError(f"unknown operator: {self.operator}")
        for name, value in asdict(self).items():
            if isinstance(value, int) and value <= 0:
                raise ValueError(f"{name} must be positive")
        if self.dtype not in {"float16", "bfloat16", "float32"}:
            raise ValueError(f"unsupported dtype: {self.dtype}")
        if self.operator == "attn":
            AttentionTileConfig(self.block_m, self.block_n).validate()
            if any(n & (n - 1) for n in (self.block_m, self.block_n)):
                raise ValueError(
                    "attn block sizes must be powers of two for the benchmark backends"
                )
            if self.seq_len % self.block_m or self.seq_len % self.block_n:
                raise ValueError("attn seq_len must be divisible by block_m and block_n")
            if self.head_dim < 16 or self.head_dim & (self.head_dim - 1):
                raise ValueError("attn head_dim must be a power of two >= 16")
            if self.dtype != "float16":
                raise ValueError("the dense attn exporter currently uses float16")
        else:
            MambaTileConfig(self.block_channels).validate()
            if self.block_channels & (self.block_channels - 1):
                raise ValueError("block_channels must be a power of two for the benchmark backends")
            if self.channels % self.block_channels:
                raise ValueError("channels must be divisible by block_channels")
            if self.state_dim & (self.state_dim - 1):
                raise ValueError("state_dim must be a power of two for the benchmark backends")

    @property
    def name(self):
        prefix = f"{self.operator}-b{self.batch}-s{self.seq_len}"
        if self.operator == "attn":
            return f"{prefix}-h{self.heads}-d{self.head_dim}"
        return f"{prefix}-c{self.channels}-n{self.state_dim}"

    def config(self):
        """Only include fields that actually affect this operator."""
        names = ["operator", "batch", "seq_len", "dtype"]
        names += (
            ["heads", "head_dim", "block_m", "block_n"]
            if self.operator == "attn"
            else ["channels", "state_dim", "block_channels"]
        )
        return {name: getattr(self, name) for name in names}

    def lower(self):
        from neptune_mlir.pipeline import (
            export_attention_to_htile_mlir,
            export_mamba_to_htile_mlir,
        )
        from neptune_mlir.schedules import AttentionTileConfig, MambaTileConfig

        self.validate()
        if self.operator == "attn":
            return export_attention_to_htile_mlir(
                variant="causal",
                batch=self.batch,
                q_heads=self.heads,
                seq_len=self.seq_len,
                head_dim=self.head_dim,
                tile_config=AttentionTileConfig(self.block_m, self.block_n),
            )
        return export_mamba_to_htile_mlir(
            batch=self.batch,
            sequence_length=self.seq_len,
            model_dim=self.channels,
            expand=1,
            state_dim=self.state_dim,
            activation_dtype=self.dtype,
            tile_config=MambaTileConfig(self.block_channels),
        )

    def make_args(self, torch, seed):
        dtype = getattr(torch, self.dtype)
        if self.operator == "attn":
            inputs = make_attn_inputs(
                self.batch, self.heads, self.seq_len, self.head_dim, dtype, "cuda", seed, scale=0.2
            )
            return [*inputs, torch.empty_like(inputs[0])]
        inputs = make_mamba_inputs(
            self.batch, self.seq_len, self.channels, self.state_dim, dtype, "cuda", seed, scale=0.2
        )
        # These are outlining/ABI details, not selective-scan operator inputs.
        return [
            *inputs,
            torch.zeros(
                (self.batch, self.channels, self.state_dim), dtype=torch.float32, device="cuda"
            ),
            torch.zeros(1, dtype=torch.float32, device="cuda"),
            torch.empty_like(inputs[0]),
        ]

    def check(self, torch, args):
        """Reference check outside timing; attention samples rows to bound reference memory."""
        output = args[-1]
        if not bool(torch.isfinite(output).all()):
            raise ValueError("kernel produced non-finite output")
        if self.operator == "attn":
            q, k, v, _ = args
            # Include tile boundaries and both ends, across every batch and head.
            rows = sorted(
                {
                    0,
                    self.seq_len - 1,
                    self.seq_len // 2,
                    min(self.block_m - 1, self.seq_len - 1),
                    min(self.block_m, self.seq_len - 1),
                }
            )
            ref = reference_attn(q, k, v, rows=rows)
            actual = output[:, :, rows].float()
        else:
            ref = reference_mamba(*args[:7]).float()
            actual = output.float()
        torch.testing.assert_close(actual, ref, rtol=1e-2, atol=2e-2)
        return {
            "scope": "sampled_rows" if self.operator == "attn" else "full",
            "max_abs_error": (actual - ref).abs().max().item(),
            "rtol": 1e-2,
            "atol": 2e-2,
        }


def suite():
    return [
        Case("attn", dtype="float16"),
        Case("mamba", batch=8, seq_len=128),
        Case("mamba", batch=8, seq_len=2048),
    ]
