import pytest

from hydroda.data.protocol import ProtocolConfig
from hydroda.data.leakage_guard import LeakageGuard


def test_protocol_roles():
    p = ProtocolConfig()
    assert p.role_for_date("2019-06-01") == "source_fit"
    assert p.role_for_date("2020-06-01") == "source_fit"
    assert p.role_for_date("2021-06-01") == "source_fit"
    assert p.role_for_date("2022-06-01") == "source_val"
    assert p.role_for_date("2023-06-01") == "target_eval"


def test_guard_rejects_query_labels_for_normalization():
    guard = LeakageGuard(ProtocolConfig())
    with pytest.raises(ValueError):
        guard.check_normalization_scope(["2023-01-01"], scope_name="source_fit_only")


def test_guard_accepts_historical_target_adaptation_dates():
    guard = LeakageGuard(ProtocolConfig())
    guard.check_support_dates(["2016-03-01", "2021-09-01"])
