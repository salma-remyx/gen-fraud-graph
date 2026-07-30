#!/usr/bin/env python
# Copyright (c) 2026 Santander Group
# SPDX-License-Identifier: Apache-2.0
"""Evaluation: verifies multi-rail scam-chain injection (target) alongside
the existing cyclic fraud-ring injection (guardrail), using the real
gen_fraud_graph package end-to-end on a tiny in-code fixture.

On `main` (pre-PR) the `gen_fraud_graph.scam_chains` module does not exist,
so the target metric degrades to 0.0 while the guardrail (fraud rings,
unaffected by the change) is still measured against the real code.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
import tempfile

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_PATH = os.path.join(REPO_ROOT, "src")
for p in (REPO_ROOT, SRC_PATH):
    if p not in sys.path:
        sys.path.insert(0, p)


def _read_cases(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", default=None)
    parser.add_argument("--ref", default=None)
    parser.add_argument("--seed", default=None)
    parser.parse_args()

    tmp_dir = tempfile.mkdtemp(prefix="gen_fraud_graph_eval_")
    fraud_ring_verification_rate = 0.0
    scam_chain_verification_rate = 0.0

    try:
        from gen_fraud_graph.embeddings import EmbeddingGenerator
        from gen_fraud_graph.typologies import FraudRingGenerator
        from gen_fraud_graph.verify import verify_fraud_patterns

        embedder = EmbeddingGenerator("fake", dim=16)

        # --- Guardrail: cyclic fraud rings (unaffected by this PR) --------
        try:
            ring_gen = FraudRingGenerator(num_rings=5, depth_range=(4, 7))
            n_ring_tx, next_tx_id = ring_gen.generate(
                max_account_id=1000,
                start_tx_id=10_000,
                embedder=embedder,
                output_dir=tmp_dir,
                fmt="csv",
                compress=False,
            )
            cases = _read_cases(os.path.join(tmp_dir, "fraud", "fraud_cases.csv"))
            ring_ok = bool(cases) and n_ring_tx > 0 and verify_fraud_patterns(cases, tmp_dir)
            fraud_ring_verification_rate = 1.0 if ring_ok else 0.0
        except Exception:
            fraud_ring_verification_rate = 0.0
            next_tx_id = 10_000 + 5000

        # --- Target: multi-rail scam chains (this PR) ----------------------
        try:
            from gen_fraud_graph.scam_chains import ScamRailGenerator, verify_scam_chains

            scam_gen = ScamRailGenerator(num_chains=5)
            n_scam_tx, _ = scam_gen.generate(
                max_account_id=1000,
                start_tx_id=next_tx_id,
                embedder=embedder,
                output_dir=tmp_dir,
                fmt="csv",
                compress=False,
            )
            scam_cases_path = os.path.join(tmp_dir, "fraud", "scam_cases.csv")
            scam_ok = (
                n_scam_tx > 0
                and os.path.exists(scam_cases_path)
                and verify_scam_chains(scam_cases_path)
            )
            scam_chain_verification_rate = 1.0 if scam_ok else 0.0
        except ImportError:
            # Baseline (pre-PR): scam_chains typology does not exist yet.
            scam_chain_verification_rate = 0.0
        except Exception:
            scam_chain_verification_rate = 0.0

    except Exception:
        # Even core imports (embeddings/typologies/verify) missing/broken:
        # emit zeroed metrics rather than crashing.
        fraud_ring_verification_rate = 0.0
        scam_chain_verification_rate = 0.0
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    print(
        json.dumps(
            {
                "scam_chain_verification_rate": scam_chain_verification_rate,
                "fraud_ring_verification_rate": fraud_ring_verification_rate,
            }
        )
    )
    sys.exit(0)


if __name__ == "__main__":
    main()