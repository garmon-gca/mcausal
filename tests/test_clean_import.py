from __future__ import annotations

import mcausal


def test_public_api():
    assert hasattr(mcausal, "update_reality")
    assert hasattr(mcausal, "history_shift")
    assert hasattr(mcausal, "residual_match")
    assert hasattr(mcausal, "write_json")
    assert hasattr(mcausal, "write_html")
    assert mcausal.__version__ == "0.1.1"
