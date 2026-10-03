"""Ingestion: PSV schema checks, malformed files fail, labels kept separate."""

import numpy as np
import pandas as pd
import pytest

from sepsis.data.schema import COLUMNS, PSVValidationError, parse_psv_bytes
from sepsis.data.synthetic import simulate_record, write_psv

HEADER = "|".join(COLUMNS)


def _row(label=0, iculos=1):
    vals = ["NaN"] * len(COLUMNS)
    vals[COLUMNS.index("HR")] = "80"
    vals[COLUMNS.index("ICULOS")] = str(iculos)
    vals[COLUMNS.index("SepsisLabel")] = str(label)
    return "|".join(vals)


def test_valid_file_parses_41_columns():
    raw = "\n".join([HEADER, _row(0, 1), _row(0, 2), _row(1, 3)]).encode()
    df = parse_psv_bytes(raw, "p1")
    assert df.shape == (3, 41)
    assert list(df.columns) == COLUMNS


@pytest.mark.parametrize("raw,msg", [
    (b"", "empty"),
    (b"<!DOCTYPE html><html><body>Not found</body></html>", "HTML"),
    (("HR|O2Sat\n1|2").encode(), "header"),
    ((HEADER + "\n" + "1|2|3").encode(), "malformed"),
    ((HEADER + "\n" + _row(0, 1) + "\n" + _row(0, 3)).encode(), "hourly"),
    ((HEADER + "\n" + _row(1, 1) + "\n" + _row(0, 2)).encode(), "reverts"),
    ((HEADER + "\n" + _row(2, 1)).encode(), "0/1"),
    ((HEADER + "\n").encode(), "no data"),
])
def test_malformed_files_fail(raw, msg):
    with pytest.raises(PSVValidationError, match=msg):
        parse_psv_bytes(raw, "bad")


def test_synthetic_fixture_roundtrip(tmp_path):
    rng = np.random.default_rng(0)
    arr = simulate_record(rng, septic=True)
    write_psv(arr, tmp_path / "p000001.psv")
    df = parse_psv_bytes((tmp_path / "p000001.psv").read_bytes())
    np.testing.assert_allclose(df.to_numpy(), arr, rtol=1e-6, equal_nan=True)


def test_prepare_separates_labels(corpus):
    assert "SepsisLabel" not in corpus.hourly.columns
    assert "label" not in corpus.hourly.columns
    assert set(corpus.outcomes.columns) == {"stay_id", "target_id", "hour_index", "label", "label_version"}
    assert not corpus.hourly.duplicated(["stay_id", "hour_index"]).any()
    assert corpus.hourly["stay_id"].str.startswith("physionet2019:").all()
    q = corpus.quality
    assert q["schema"]["n_columns"] == 41
    assert q["invalid_files"] == []


def test_prepare_rejects_malformed_corpus(tmp_path):
    from sepsis.data.prepare import prepare
    from sepsis.data.synthetic import make_fixture

    raw = make_fixture(tmp_path / "raw", n_a=3, n_b=2)
    (raw / "training_setA" / "p000002.psv").write_text("<html>oops</html>")
    with pytest.raises(RuntimeError, match="invalid source files"):
        prepare(raw_root=raw, out_dir=tmp_path / "out", workers=1, require_expected_counts=False)


def test_prepare_enforces_expected_counts(tmp_path):
    from sepsis.data.prepare import prepare
    from sepsis.data.synthetic import make_fixture

    raw = make_fixture(tmp_path / "raw", n_a=3, n_b=2)
    with pytest.raises(RuntimeError, match="expected 20336"):
        prepare(raw_root=raw, out_dir=tmp_path / "out", workers=1, require_expected_counts=True)
