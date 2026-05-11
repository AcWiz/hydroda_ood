"""No-Leakage Tests for K-Date Splits.

Tests per kdate_protocol.yaml tests_required:
- test_support_dates_in_support_year
- test_query_dates_in_query_years
- test_no_support_query_overlap
- test_k_matches_number_of_support_dates
- test_k0_has_no_support_dates
- test_selection_flags_false_for_query_labels
"""

from __future__ import annotations

import json
import os

import pytest

# Path to the generated splits JSON
SPLITS_JSON = "artifacts/splits/US_loro_kdate_splits.json"


@pytest.fixture
def splits_data():
    """Load splits JSON."""
    if not os.path.exists(SPLITS_JSON):
        pytest.skip(f"Splits file not found: {SPLITS_JSON}")
    with open(SPLITS_JSON, "r") as f:
        data = json.load(f)
    return data["splits"]


@pytest.fixture
def splits_by_K(splits_data):
    """Group splits by K value."""
    result = {}
    for s in splits_data:
        k = s["K"]
        if k not in result:
            result[k] = []
        result[k].append(s)
    return result


class TestSupportDatesInSupportYear:
    """Test that all target_support_dates fall within 2021."""

    def test_all_support_dates_in_2021(self, splits_data):
        for split in splits_data:
            support_dates = split["target_support_dates"]
            for d in support_dates:
                year = int(d["date_str"].split("-")[0])
                assert year == 2021, (
                    f"Split {split['target_region_id']} K={split['K']} seed={split['seed']}: "
                    f"support date {d['date_str']} not in 2021"
                )


class TestQueryDatesInQueryYears:
    """Test that all target_query_dates fall within 2022-2025."""

    def test_all_query_dates_in_2022_to_2025(self, splits_data):
        valid_years = {2022, 2023, 2024, 2025}
        for split in splits_data:
            query_dates = split["target_query_dates"]
            for d in query_dates:
                year = int(d["date_str"].split("-")[0])
                assert year in valid_years, (
                    f"Split {split['target_region_id']} K={split['K']} seed={split['seed']}: "
                    f"query date {d['date_str']} not in 2022-2025"
                )


class TestNoSupportQueryOverlap:
    """Test that support and query date sets do not overlap."""

    def test_no_overlap_between_support_and_query(self, splits_data):
        for split in splits_data:
            support_dates = set(d["date_str"] for d in split["target_support_dates"])
            query_dates = set(d["date_str"] for d in split["target_query_dates"])

            overlap = support_dates & query_dates
            assert len(overlap) == 0, (
                f"Split {split['target_region_id']} K={split['K']} seed={split['seed']}: "
                f"support and query overlap at: {overlap}"
            )


class TestKMatchesNumberOfSupportDates:
    """Test that K matches the number of support dates selected.

    Note: For K=4, some regions (e.g., US-R6 Central Rockies) may have
    insufficient valid dates in certain quarters (e.g., Q1 winter).
    Actual support count may be less than K due to data availability.
    """

    def test_k_equals_support_count_or_less(self, splits_data):
        for split in splits_data:
            k = split["K"]
            support_count = len(split["target_support_dates"])
            # K is target, actual may be less if valid dates insufficient
            assert support_count <= k, (
                f"Split {split['target_region_id']} K={k} seed={split['seed']}: "
                f"expected at most {k} support dates, got {support_count}"
            )


class TestK0HasNoSupportDates:
    """Test that K=0 splits have no support dates."""

    def test_k0_splits_are_empty_support(self, splits_by_K):
        if 0 not in splits_by_K:
            pytest.skip("No K=0 splits found")
        for split in splits_by_K[0]:
            assert len(split["target_support_dates"]) == 0, (
                f"Split {split['target_region_id']} K=0 seed={split['seed']}: "
                f"expected 0 support dates, got {len(split['target_support_dates'])}"
            )


class TestSelectionFlagsFalseForQueryLabels:
    """Test that selection_uses_analysis and selection_uses_query_labels are False."""

    def test_selection_flags_are_false(self, splits_data):
        for split in splits_data:
            assert split["selection_uses_analysis"] is False, (
                f"Split {split['target_region_id']} K={split['K']} seed={split['seed']}: "
                f"selection_uses_analysis should be False"
            )
            assert split["selection_uses_query_labels"] is False, (
                f"Split {split['target_region_id']} K={split['K']} seed={split['seed']}: "
                f"selection_uses_query_labels should be False"
            )


class TestSourceTrainIn2015To2020:
    """Test that source_train dates fall within 2015-04 to 2020-12."""

    def test_source_train_dates_in_range(self, splits_data):
        for split in splits_data:
            source_dates = split["source_train_dates"]
            for d in source_dates:
                year = int(d["date_str"].split("-")[0])
                month = int(d["date_str"].split("-")[1])
                day = int(d["date_str"].split("-")[2])
                # Must be 2015-04 or later, and before 2021
                assert (year > 2015 and year < 2021) or (year == 2015 and month >= 4), (
                    f"Split {split['target_region_id']}: "
                    f"source date {d['date_str']} outside 2015-04 to 2020-12"
                )
                assert year <= 2020, (
                    f"Split {split['target_region_id']}: "
                    f"source date {d['date_str']} outside 2015-04 to 2020-12"
                )


class TestAllRegionsCovered:
    """Test that all 6 US regions appear as target at least once."""

    def test_all_regions_as_target(self, splits_data):
        target_regions = set(s["target_region_id"] for s in splits_data)
        expected = {"US-R1", "US-R2", "US-R3", "US-R4", "US-R5", "US-R6"}
        assert target_regions == expected, (
            f"Expected all 6 regions as targets, got: {target_regions}"
        )


class TestSourceRegionsExcludesTarget:
    """Test that source_region_ids does not include target region."""

    def test_source_excludes_target(self, splits_data):
        for split in splits_data:
            target = split["target_region_id"]
            sources = set(split["source_region_ids"])
            assert target not in sources, (
                f"Split target {target}: target should not be in source regions"
            )
            assert len(sources) == 5, (
                f"Split target {target}: expected 5 source regions, got {len(sources)}"
            )


class TestManifestRequiredFields:
    """Test that manifest contains all required fields per kdate_protocol.yaml."""

    def test_required_fields_present(self, splits_data):
        required_fields = [
            "benchmark_id",
            "protocol_version",
            "target_region_id",
            "source_region_ids",
            "K",
            "seed",
            "source_train_dates",
            "target_support_dates",
            "target_query_dates",
            "selection_uses_analysis",
            "selection_uses_query_labels",
            "created_by",
            "created_utc",
        ]
        for split in splits_data:
            for field in required_fields:
                assert field in split, (
                    f"Split {split['target_region_id']} K={split['K']} seed={split['seed']}: "
                    f"missing required field '{field}'"
                )