import pytest

from hydroda.data.protocol import ProtocolConfig
from hydroda.data.leakage_guard import LeakageGuard


def test_protocol_roles():
    p = ProtocolConfig()
    assert p.role_for_date("2019-06-01") == "source_fit"
    assert p.role_for_date("2020-06-01") == "source_val"
    assert p.role_for_date("2021-06-01") == "target_context"
    assert p.role_for_date("2023-06-01") == "target_query"


def test_guard_rejects_query_labels_for_normalization():
    guard = LeakageGuard(ProtocolConfig())
    with pytest.raises(ValueError):
        guard.check_normalization_scope(["2023-01-01"], scope_name="source_train_only")


def test_guard_accepts_support_dates_in_2021():
    guard = LeakageGuard(ProtocolConfig())
    guard.check_support_dates(["2022-03-01", "2022-09-01"])
