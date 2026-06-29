# Copyright (c) 2026 Santander Group
# SPDX-License-Identifier: Apache-2.0

"""Multi-rail scam-chain typology for synthetic graph injection.

Where :class:`~gen_fraud_graph.typologies.FraudRingGenerator` injects a single
cyclic money-laundering pattern, this module injects **multi-rail scam chains**:
linear, temporally ordered paths whose consecutive edges each represent a stage
of a real scam lifecycle.

The rail taxonomy and the observation that scam processes are predominantly
*multi-rail*, *temporally ordered*, and vary *systematically in path complexity*
by scam type are adapted from:

    "Understanding Scam Trends and Rail Paths from Reddit Self-Disclosure
    Narratives" (arXiv:2606.16874).

Each generated chain advances through an ordered subset of the four rails
(identity, communication, platform, payment), so the resulting graph carries a
strictly increasing timestamp sequence and a recoverable scam path rather than a
single isolated signal.
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta

import numpy as np

from gen_fraud_graph.embeddings import EmbeddingGenerator
from gen_fraud_graph.exporters import get_headers, write_output

# ---------------------------------------------------------------------------
# Rail taxonomy (paper: identity / communication / platform / payment)
# ---------------------------------------------------------------------------

#: The four scam rails, with the suspicious edge descriptions used per stage.
RAIL_DESCRIPTIONS: dict[str, str] = {
    "identity": "stolen or spoofed identity used to onboard victim",
    "communication": "out-of-band contact luring victim off platform",
    "platform": "fake marketplace or portal staging the scam",
    "payment": "irreversible payment rail moving victim funds",
}

#: Scam types mapped to their ordered rail paths.  Path *complexity* (number of
#: rails) varies systematically by type, mirroring the paper's finding.
SCAM_TYPE_RAILS: dict[str, list[str]] = {
    "phishing": ["communication", "identity", "payment"],
    "tech_support_scam": ["communication", "platform", "payment"],
    "marketplace_scam": ["platform", "communication", "payment"],
    "investment_scam": ["platform", "communication", "identity", "payment"],
    "romance_scam": ["platform", "communication", "identity", "payment"],
}

#: Base timestamp; each rail stage advances by ``stage_step`` from here.
_BASE_TIME = datetime(2024, 1, 1, 0, 0, 0)


@dataclass
class ScamRailGenerator:
    """Generate multi-rail, temporally ordered scam chains.

    Each chain is a linear path ``acc_a -> acc_b -> ...`` whose edges are the
    ordered rails of a randomly chosen scam type.  Timestamps increase by
    ``stage_step_hours`` per rail, so the emitted edges encode a recoverable,
    temporally ordered scam path.

    Args:
        num_chains: How many scam chains to create.
        amount: Fixed transaction amount injected on each rail edge.
        stage_step_hours: Hours between consecutive rail stages.
        scam_type_rails: Scam-type → ordered-rail-path mapping.
    """

    num_chains: int = 100
    amount: float = 4999.00
    stage_step_hours: int = 6
    scam_type_rails: dict[str, list[str]] = field(default_factory=lambda: SCAM_TYPE_RAILS)

    # Header for the scam-case manifest (super-set of the ring manifest, with
    # explicit rail ordering and per-stage timestamps for path analysis).
    CASE_HEADERS = [
        "pattern_id",
        "start_acc_id",
        "pattern_type",
        "depth",
        "involved_accounts",
        "rails",
        "timestamps",
    ]

    def generate(
        self,
        max_account_id: int,
        start_tx_id: int,
        embedder: EmbeddingGenerator,
        output_dir: str,
        fmt: str = "csv",
        compress: bool = False,
    ) -> tuple[int, int]:
        """Generate scam chains and write output files.

        Mirrors the :class:`FraudRingGenerator.generate` I/O contract but writes
        to ``transactions_scam`` / ``scam_cases`` so it composes with — rather
        than clobbers — the fraud-ring output.

        Returns:
            ``(num_scam_transactions, next_tx_id)``
        """
        fraud_dir = os.path.join(output_dir, "fraud")
        os.makedirs(fraud_dir, exist_ok=True)

        headers_tx = get_headers("transaction", fmt)  # type: ignore[arg-type]

        tx_rows: list[list] = []
        case_rows: list[list] = []
        current_tx_id = start_tx_id
        scam_types = list(self.scam_type_rails)

        for pattern_id in range(self.num_chains):
            scam_type = scam_types[pattern_id % len(scam_types)]
            rails = self.scam_type_rails[scam_type]
            depth = len(rails)

            # A path of ``depth`` rails spans ``depth + 1`` accounts.
            span = depth + 1
            start_node = 0 if max_account_id < span else random.randint(0, max_account_id - span)
            accounts = [f"acc_{start_node + d}" for d in range(span)]

            batch_texts: list[str] = []
            batch_rows: list[list] = []
            timestamps: list[str] = []

            for stage, rail in enumerate(rails):
                src = accounts[stage]
                dst = accounts[stage + 1]
                desc = RAIL_DESCRIPTIONS[rail]
                ts = (_BASE_TIME + timedelta(hours=self.stage_step_hours * stage)).isoformat()
                timestamps.append(ts)
                batch_texts.append(desc)

                row: list = [f"tx_{current_tx_id}", src, dst]
                if fmt == "neptune":
                    row.append("TRANSFER")
                row.extend([self.amount, ts, desc])
                batch_rows.append(row)
                current_tx_id += 1

            embeddings = embedder.generate(batch_texts)

            for idx, r in enumerate(batch_rows):
                if fmt == "neptune":
                    tx_rows.append(r)
                else:
                    vec = embeddings[idx]
                    if isinstance(vec, np.ndarray):
                        vec = vec.tolist()
                    tx_rows.append(r + ["|".join(map(str, vec))])

            case_rows.append(
                [
                    f"scam_{pattern_id}",
                    accounts[0],
                    scam_type,
                    depth,
                    "|".join(accounts),
                    "|".join(rails),
                    "|".join(timestamps),
                ]
            )

        file_tx = os.path.join(fraud_dir, "transactions_scam")
        file_cases = os.path.join(fraud_dir, "scam_cases")
        write_output(file_tx, headers_tx, tx_rows, compress=compress)
        write_output(file_cases, self.CASE_HEADERS, case_rows, compress=compress)

        return len(tx_rows), current_tx_id


# ---------------------------------------------------------------------------
# Chain verification (parallel to verify.verify_fraud_patterns)
# ---------------------------------------------------------------------------


def verify_scam_chains(scam_cases_path: str) -> bool:
    """Check that every scam chain's rail edges exist and are time-ordered.

    For each case this asserts (a) every consecutive rail edge along the path is
    backed by a real ``transactions_scam`` edge, and (b) the per-stage
    timestamps are strictly increasing — the temporal-ordering property the
    paper identifies as intrinsic to multi-rail scam processes.

    Args:
        scam_cases_path: Path to ``scam_cases.csv``.

    Returns:
        ``True`` if all chains are valid, ``False`` otherwise.
    """
    import csv
    import sys
    from collections import defaultdict

    scam_tx_path = os.path.join(os.path.dirname(scam_cases_path), "transactions_scam.csv")
    if not os.path.exists(scam_tx_path):
        print(f"ERROR: {scam_tx_path} not found", file=sys.stderr)
        return False

    print("Loading scam transaction edges...")
    edges: dict[str, set[str]] = defaultdict(set)
    with open(scam_tx_path) as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            src = row.get("src_id") or row.get("~from", "")
            dst = row.get("dst_id") or row.get("~to", "")
            if src and dst:
                edges[src].add(dst)

    print("Verifying scam chains...")
    all_valid = True
    with open(scam_cases_path) as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            pattern_id = row["pattern_id"]
            accounts = row["involved_accounts"].split("|")
            rails = row["rails"].split("|")
            timestamps = row["timestamps"].split("|")

            # (a) Each rail edge along the linear path must exist.
            for stage in range(len(rails)):
                src = accounts[stage]
                dst = accounts[stage + 1]
                if dst not in edges.get(src, set()):
                    print(f"  FAIL: {pattern_id} — missing rail edge {src} -> {dst}")
                    all_valid = False
                    break
            else:
                # (b) Timestamps must be strictly increasing across rails.
                if any(timestamps[i] >= timestamps[i + 1] for i in range(len(timestamps) - 1)):
                    print(f"  FAIL: {pattern_id} — rails not temporally ordered")
                    all_valid = False

    if all_valid:
        print("All scam chains verified successfully.")
    else:
        print("Some scam chains failed verification.", file=sys.stderr)

    return all_valid
