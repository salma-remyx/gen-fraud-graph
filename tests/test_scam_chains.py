# Copyright (c) 2026 Santander Group
# SPDX-License-Identifier: Apache-2.0

"""Tests for the multi-rail scam-chain typology and its generator wiring."""

from __future__ import annotations

import csv
import os
import shutil
import sys
import tempfile

import pytest

from gen_fraud_graph.config import Config
from gen_fraud_graph.embeddings import EmbeddingGenerator

# Imports from NON-NEW modules — proves the new typology is wired into the
# existing generator orchestration and verification CLI.
from gen_fraud_graph.generator import FraudGraphGenerator
from gen_fraud_graph.scam_chains import (
    SCAM_TYPE_RAILS,
    ScamRailGenerator,
    verify_scam_chains,
)
from gen_fraud_graph.verify import main as verify_main

_TX_HEADERS = ["tx_id", "src_id", "dst_id", "amount", "timestamp", "description", "embedding"]


@pytest.fixture()
def tmp_dir():
    d = tempfile.mkdtemp(prefix="gen_fraud_graph_scam_")
    yield d
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture()
def small_config(tmp_dir):
    return Config(
        scale_factor=0.0001,
        embedding_provider="fake",
        workers=1,
        batches_per_worker=1,
        output_dir=tmp_dir,
    )


# ---------------------------------------------------------------------------
# Config derivation
# ---------------------------------------------------------------------------


class TestScamConfig:
    def test_default_derived(self):
        assert Config().num_scam_chains == 500

    def test_scaled_floor(self):
        assert Config(scale_factor=0.0001).num_scam_chains == 5

    def test_explicit(self):
        assert Config(num_scam_chains=12).num_scam_chains == 12


# ---------------------------------------------------------------------------
# ScamRailGenerator — direct
# ---------------------------------------------------------------------------


class TestScamRailGenerator:
    def test_generate_creates_files(self, tmp_dir):
        emb = EmbeddingGenerator("fake", dim=32)
        gen = ScamRailGenerator(num_chains=6)
        n_tx, next_id = gen.generate(
            max_account_id=1000,
            start_tx_id=100,
            embedder=emb,
            output_dir=tmp_dir,
            fmt="csv",
        )
        assert n_tx > 0
        assert next_id == 100 + n_tx
        assert os.path.exists(os.path.join(tmp_dir, "fraud", "transactions_scam.csv"))
        assert os.path.exists(os.path.join(tmp_dir, "fraud", "scam_cases.csv"))

    def test_cases_are_multi_rail_and_ordered(self, tmp_dir):
        emb = EmbeddingGenerator("fake", dim=16)
        ScamRailGenerator(num_chains=len(SCAM_TYPE_RAILS)).generate(
            max_account_id=500,
            start_tx_id=0,
            embedder=emb,
            output_dir=tmp_dir,
        )
        with open(os.path.join(tmp_dir, "fraud", "scam_cases.csv")) as fh:
            rows = list(csv.DictReader(fh))

        assert {r["pattern_type"] for r in rows} == set(SCAM_TYPE_RAILS)
        for r in rows:
            rails = r["rails"].split("|")
            timestamps = r["timestamps"].split("|")
            accounts = r["involved_accounts"].split("|")
            # Multi-rail: every known scam type spans >= 2 rails.
            assert len(rails) >= 2
            # A path of N rails spans N+1 accounts.
            assert len(accounts) == len(rails) + 1
            # Temporal ordering is strictly increasing.
            assert timestamps == sorted(timestamps)
            assert len(set(timestamps)) == len(timestamps)

    def test_neptune_format(self, tmp_dir):
        emb = EmbeddingGenerator("fake", dim=16)
        n_tx, _ = ScamRailGenerator(num_chains=3).generate(
            max_account_id=50,
            start_tx_id=0,
            embedder=emb,
            output_dir=tmp_dir,
            fmt="neptune",
        )
        assert n_tx > 0
        path = os.path.join(tmp_dir, "fraud", "transactions_scam.csv")
        with open(path) as fh:
            header = next(csv.reader(fh))
        assert "~from" in header

    def test_small_max_account_id_fallback(self, tmp_dir):
        emb = EmbeddingGenerator("fake", dim=8)
        n_tx, _ = ScamRailGenerator(num_chains=2).generate(
            max_account_id=2,
            start_tx_id=0,
            embedder=emb,
            output_dir=tmp_dir,
        )
        assert n_tx > 0

    def test_compress(self, tmp_dir):
        emb = EmbeddingGenerator("fake", dim=8)
        ScamRailGenerator(num_chains=2).generate(
            max_account_id=50,
            start_tx_id=0,
            embedder=emb,
            output_dir=tmp_dir,
            compress=True,
        )
        assert os.path.exists(os.path.join(tmp_dir, "fraud", "scam_cases.csv.zip"))


# ---------------------------------------------------------------------------
# verify_scam_chains
# ---------------------------------------------------------------------------


class TestVerifyScamChains:
    def test_valid_chains(self, tmp_dir):
        emb = EmbeddingGenerator("fake", dim=8)
        ScamRailGenerator(num_chains=4).generate(
            max_account_id=200,
            start_tx_id=0,
            embedder=emb,
            output_dir=tmp_dir,
        )
        cases = os.path.join(tmp_dir, "fraud", "scam_cases.csv")
        assert verify_scam_chains(cases) is True

    def test_missing_tx_file(self, tmp_dir, capsys):
        cases = os.path.join(tmp_dir, "scam_cases.csv")
        with open(cases, "w", newline="") as fh:
            csv.writer(fh).writerow(ScamRailGenerator.CASE_HEADERS)
        assert verify_scam_chains(cases) is False
        assert "not found" in capsys.readouterr().err

    def test_missing_edge_detected(self, tmp_dir):
        tx = os.path.join(tmp_dir, "transactions_scam.csv")
        cases = os.path.join(tmp_dir, "scam_cases.csv")
        with open(tx, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(_TX_HEADERS)
            # Only the first hop is present; the second rail edge is missing.
            w.writerow(["tx_0", "acc_0", "acc_1", 1.0, "2024-01-01T00:00:00", "x", ""])
        with open(cases, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(ScamRailGenerator.CASE_HEADERS)
            w.writerow(
                [
                    "scam_0",
                    "acc_0",
                    "phishing",
                    2,
                    "acc_0|acc_1|acc_2",
                    "communication|payment",
                    "2024-01-01T00:00:00|2024-01-01T06:00:00",
                ]
            )
        assert verify_scam_chains(cases) is False

    def test_unordered_timestamps_detected(self, tmp_dir):
        tx = os.path.join(tmp_dir, "transactions_scam.csv")
        cases = os.path.join(tmp_dir, "scam_cases.csv")
        with open(tx, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(_TX_HEADERS)
            w.writerow(["tx_0", "acc_0", "acc_1", 1.0, "t1", "x", ""])
            w.writerow(["tx_1", "acc_1", "acc_2", 1.0, "t0", "x", ""])
        with open(cases, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(ScamRailGenerator.CASE_HEADERS)
            # Edges exist but timestamps go backwards → ordering failure.
            w.writerow(
                [
                    "scam_0",
                    "acc_0",
                    "phishing",
                    2,
                    "acc_0|acc_1|acc_2",
                    "communication|payment",
                    "2024-01-01T06:00:00|2024-01-01T00:00:00",
                ]
            )
        assert verify_scam_chains(cases) is False


# ---------------------------------------------------------------------------
# End-to-end wiring through the existing orchestrator + verify CLI
# ---------------------------------------------------------------------------


class TestScamWiring:
    def test_pipeline_injects_scam_chains(self, small_config):
        FraudGraphGenerator(small_config).run()
        fraud_dir = os.path.join(small_config.output_dir, "fraud")
        # Ring outputs and scam-chain outputs coexist.
        assert os.path.exists(os.path.join(fraud_dir, "fraud_cases.csv"))
        assert os.path.exists(os.path.join(fraud_dir, "scam_cases.csv"))
        assert os.path.exists(os.path.join(fraud_dir, "transactions_scam.csv"))

    def test_verify_cli_checks_scam_chains(self, small_config, monkeypatch):
        FraudGraphGenerator(small_config).run()
        monkeypatch.setattr(sys, "argv", ["verify", "--data-dir", small_config.output_dir])
        with pytest.raises(SystemExit) as exc:
            verify_main()
        assert exc.value.code == 0
