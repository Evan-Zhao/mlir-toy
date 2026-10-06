"""Benchmark presets and policy over the shared export/test operator infrastructure."""

import argparse
import hashlib
import json

from neptune_mlir.cli.operators import operator_options, validate_runtime_options
from neptune_mlir.operator.variants import VARIANTS
from neptune_mlir.testing import document_lengths, make_arguments, reference_output


class Case:
    def __init__(self, operator, *, doc_lengths=None, **options):
        self.operator = "dense" if operator == "attn" else operator
        self.options = operator_options(self.operator, options)
        unknown = options.keys() - self.options.keys()
        if unknown:
            raise ValueError(f"unknown {self.operator} options: {', '.join(sorted(unknown))}")
        self.doc_lengths = None if doc_lengths is None else tuple(doc_lengths)

    @classmethod
    def from_args(cls, args):
        operator = "dense" if args.operator == "attn" else args.operator
        return cls(
            operator,
            doc_lengths=getattr(args, "doc_lengths", None),
            **operator_options(operator, vars(args)),
        )

    def validate(self):
        validate_runtime_options(self.operator, self.options)
        if self.operator == "varlen":
            document_lengths(self.options, self.doc_lengths)
        elif self.doc_lengths is not None:
            raise ValueError("doc_lengths only applies to varlen")

    @property
    def name(self):
        o = self.options
        if self.operator == "dense":
            label = f"dense-{o['variant']}-b{o['batch']}-s{o['seq_len']}-h{o['q_heads']}"
        elif self.operator == "varlen":
            label = f"varlen-docs{o['num_docs']}-t{o['total_tokens']}-h{o['heads']}"
        else:
            label = f"mamba-b{o['batch']}-s{o['sequence_length']}-c{o['model_dim'] * o['expand']}"
        # Distinguish dtype, tiles, rectangular/GQA shapes and document packings.
        digest = hashlib.sha256(json.dumps(self.config(), sort_keys=True).encode()).hexdigest()[:8]
        return f"{label}-{digest}"

    def config(self):
        config = {"operator": self.operator, **self.options}
        if self.operator == "varlen":
            config["doc_lengths"] = list(document_lengths(self.options, self.doc_lengths))
        return config

    def lower(self):
        from neptune_mlir.cli.export import export_at_stage

        self.validate()
        return export_at_stage(
            argparse.Namespace(operator=self.operator, stage="htile", **self.options)
        )

    def make_args(self, torch, seed):
        return make_arguments(self.operator, self.options, seed=seed, lengths=self.doc_lengths)

    def check(self, torch, args):
        """Benchmark sanity check; pipeline tests use the same helpers on every row."""
        output = args[-1]
        if not bool(torch.isfinite(output).all()):
            raise ValueError("kernel produced non-finite output")
        rows = None
        if self.operator == "dense":
            seq_len, block_m = self.options["seq_len"], self.options["block_m"]
            rows = sorted(
                {
                    0,
                    seq_len - 1,
                    seq_len // 2,
                    min(block_m - 1, seq_len - 1),
                    min(block_m, seq_len - 1),
                }
            )
        with torch.no_grad():
            ref = reference_output(self.operator, self.options, args, rows=rows)
        actual = output[:, :, rows].float() if rows is not None else output.float()
        torch.testing.assert_close(actual, ref, rtol=1e-2, atol=2e-2)
        return {
            "scope": "sampled_rows" if rows is not None else "full",
            "max_abs_error": (actual - ref).abs().max().item(),
            "rtol": 1e-2,
            "atol": 2e-2,
        }


def suite():
    return [
        *[Case("dense", variant=variant, q_heads=8, seq_len=2048) for variant in VARIANTS],
        Case("varlen"),
        Case("mamba", sequence_length=128),
        Case("mamba", sequence_length=2048),
    ]
