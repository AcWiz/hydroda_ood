import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest


def _payload(values):
    arr = np.ascontiguousarray(np.asarray(values, dtype=np.float32))
    return {
        "shape": list(arr.shape),
        "dtype": "float32",
        "sha256": hashlib.sha256(arr.tobytes()).hexdigest(),
        "values": arr.reshape(-1).tolist(),
    }


def _record(
    *,
    sample_idx=0,
    query_date="2023-01-01",
    surface_forecast=None,
    surface_ref=None,
    surface_pred=None,
    rootzone_forecast=None,
    rootzone_ref=None,
    rootzone_pred=None,
    metric_mask=None,
    latitude_weight=None,
):
    surface_forecast = np.asarray(
        surface_forecast if surface_forecast is not None else [[0.2, 0.2], [0.2, 0.2]],
        dtype=np.float32,
    )
    surface_ref = np.asarray(
        surface_ref if surface_ref is not None else [[1.0, -2.0], [100.0, 4.0]],
        dtype=np.float32,
    )
    surface_pred = np.asarray(
        surface_pred if surface_pred is not None else [[0.0, -1.0], [0.0, 6.0]],
        dtype=np.float32,
    )
    rootzone_forecast = np.asarray(
        rootzone_forecast if rootzone_forecast is not None else [[0.4, 0.4], [0.4, 0.4]],
        dtype=np.float32,
    )
    rootzone_ref = np.asarray(
        rootzone_ref if rootzone_ref is not None else [[2.0, 1.0], [100.0, -3.0]],
        dtype=np.float32,
    )
    rootzone_pred = np.asarray(
        rootzone_pred if rootzone_pred is not None else [[1.0, 2.0], [0.0, -1.0]],
        dtype=np.float32,
    )
    surface_analysis = surface_forecast + surface_ref
    rootzone_analysis = rootzone_forecast + rootzone_ref
    mask = np.asarray(
        metric_mask if metric_mask is not None else [[1.0, 1.0], [0.0, 1.0]],
        dtype=np.float32,
    )
    latw = np.asarray(
        latitude_weight if latitude_weight is not None else [[1.0, 2.0], [10.0, 1.0]],
        dtype=np.float32,
    )
    return {
        "schema_version": "hydroda_prediction_record_v1",
        "sample_idx": sample_idx,
        "query_time_index": 100 + sample_idx,
        "query_date": query_date,
        "target_region_id": "US-R1",
        "split_role": "target_eval",
        "method": "HyperDA-TRUST",
        "arrays": {
            "forecast_surface": _payload(surface_forecast),
            "forecast_rootzone": _payload(rootzone_forecast),
            "analysis_surface": _payload(surface_analysis),
            "analysis_rootzone": _payload(rootzone_analysis),
            "increment_surface": _payload(surface_ref),
            "increment_rootzone": _payload(rootzone_ref),
            "pred_increment_surface": _payload(surface_pred),
            "pred_increment_rootzone": _payload(rootzone_pred),
            "metric_mask": _payload(mask),
            "latitude_weight": _payload(latw),
        },
    }


def test_build_sample_map_decodes_arrays_and_computes_weighted_maps():
    from scripts.analysis.plot_ref_pred_rmse_maps import build_sample_map, weighted_rmse

    sample = build_sample_map(_record())

    surface = sample.variables["surface"]
    expected_model = np.sqrt((1.0 * 1.0 + 2.0 * 1.0 + 1.0 * 4.0) / 4.0)
    expected_forecast = np.sqrt((1.0 * 1.0 + 2.0 * 4.0 + 1.0 * 16.0) / 4.0)

    assert np.isclose(surface.model_wrmse, expected_model)
    assert np.isclose(surface.forecast_wrmse, expected_forecast)
    assert np.isclose(surface.skill_vs_forecast, 1.0 - expected_model / expected_forecast)
    assert np.isclose(weighted_rmse(surface.pred, surface.ref, sample.mask, sample.latitude_weight), expected_model)

    np.testing.assert_allclose(surface.ref[[0, 0, 1], [0, 1, 1]], [1.2, -1.8, 4.2])
    np.testing.assert_allclose(surface.pred[[0, 0, 1], [0, 1, 1]], [0.2, -0.8, 6.2])
    np.testing.assert_allclose(surface.rmse_map[[0, 0, 1], [0, 1, 1]], [1.0, 1.0, 2.0])
    assert np.isnan(surface.ref[1, 0])
    assert np.isnan(surface.pred[1, 0])
    assert np.isnan(surface.rmse_map[1, 0])


def test_load_prediction_records_accepts_jsonl(tmp_path):
    from scripts.analysis.plot_ref_pred_rmse_maps import load_prediction_records

    path = tmp_path / "records.jsonl"
    records = [_record(sample_idx=0), _record(sample_idx=1, query_date="2023-01-02")]
    path.write_text("\n".join(json.dumps(record) for record in records) + "\n")

    loaded = load_prediction_records(path)

    assert [record["query_date"] for record in loaded] == ["2023-01-01", "2023-01-02"]


def test_select_representative_sample_prefers_positive_two_variable_skill():
    from scripts.analysis.plot_ref_pred_rmse_maps import build_sample_map, select_representative_sample

    high_update_but_worse = build_sample_map(
        _record(
            sample_idx=0,
            query_date="2023-01-01",
            surface_ref=np.ones((2, 2), dtype=np.float32) * 5.0,
            surface_pred=np.ones((2, 2), dtype=np.float32) * 12.0,
            rootzone_ref=np.ones((2, 2), dtype=np.float32) * 4.0,
            rootzone_pred=np.ones((2, 2), dtype=np.float32) * -10.0,
        )
    )
    positive_lower_strength = build_sample_map(
        _record(
            sample_idx=1,
            query_date="2023-01-02",
            surface_ref=np.ones((2, 2), dtype=np.float32) * 2.0,
            surface_pred=np.ones((2, 2), dtype=np.float32) * 1.0,
            rootzone_ref=np.ones((2, 2), dtype=np.float32) * 3.0,
            rootzone_pred=np.ones((2, 2), dtype=np.float32) * 2.0,
        )
    )

    selected = select_representative_sample([high_update_but_worse, positive_lower_strength])

    assert selected.sample_idx == 1
    assert selected.selection_reason == "balanced_visual_with_nontrivial_increment"


def test_select_representative_sample_prefers_balanced_visual_case_with_signal_and_skill_floor():
    from scripts.analysis.plot_ref_pred_rmse_maps import build_sample_map, select_representative_sample

    high_skill_higher_error = build_sample_map(
        _record(
            sample_idx=0,
            query_date="2023-01-01",
            surface_ref=np.ones((2, 2), dtype=np.float32) * 10.0,
            surface_pred=np.ones((2, 2), dtype=np.float32) * 8.0,
            rootzone_ref=np.ones((2, 2), dtype=np.float32) * 10.0,
            rootzone_pred=np.ones((2, 2), dtype=np.float32) * 8.0,
        )
    )
    display_case = build_sample_map(
        _record(
            sample_idx=1,
            query_date="2023-01-02",
            surface_ref=np.ones((2, 2), dtype=np.float32) * 1.0,
            surface_pred=np.ones((2, 2), dtype=np.float32) * 0.5,
            rootzone_ref=np.ones((2, 2), dtype=np.float32) * 1.0,
            rootzone_pred=np.ones((2, 2), dtype=np.float32) * 0.5,
        )
    )
    low_skill_low_error = build_sample_map(
        _record(
            sample_idx=2,
            query_date="2023-01-03",
            surface_ref=np.ones((2, 2), dtype=np.float32) * 0.001,
            surface_pred=np.ones((2, 2), dtype=np.float32) * 0.00001,
            rootzone_ref=np.ones((2, 2), dtype=np.float32) * 0.001,
            rootzone_pred=np.ones((2, 2), dtype=np.float32) * 0.00001,
        )
    )

    selected = select_representative_sample(
        [high_skill_higher_error, display_case, low_skill_low_error],
        min_signal_quantile=0.5,
    )

    assert selected.sample_idx == 1
    assert selected.selection_reason == "balanced_visual_with_nontrivial_increment"
    assert selected.selection_metadata["selection_mode"] == "balanced_visual_with_signal"
    assert selected.selection_metadata["combined_relative_skill_vs_forecast"] < 0.8
    assert selected.selection_metadata["combined_model_wrmse"] < 1.0
    assert selected.selection_metadata["eligible_candidate_count"] == 2
    assert selected.selection_metadata["balanced_visual_score"] > 0
    assert selected.selection_metadata["signal_threshold"] > selected.selection_metadata[
        "filtered_below_signal_threshold"
    ][0]["true_increment_strength"]


def test_select_representative_sample_rejects_near_zero_increment_easy_case():
    from scripts.analysis.plot_ref_pred_rmse_maps import build_sample_map, select_representative_sample

    low_error_no_signal = build_sample_map(
        _record(
            sample_idx=0,
            query_date="2023-01-01",
            surface_ref=np.ones((2, 2), dtype=np.float32) * 0.001,
            surface_pred=np.ones((2, 2), dtype=np.float32) * 0.001,
            rootzone_ref=np.ones((2, 2), dtype=np.float32) * 0.001,
            rootzone_pred=np.ones((2, 2), dtype=np.float32) * 0.001,
        )
    )
    nontrivial_signal = build_sample_map(
        _record(
            sample_idx=1,
            query_date="2023-01-02",
            surface_ref=np.ones((2, 2), dtype=np.float32) * 1.0,
            surface_pred=np.ones((2, 2), dtype=np.float32) * 0.8,
            rootzone_ref=np.ones((2, 2), dtype=np.float32) * 1.0,
            rootzone_pred=np.ones((2, 2), dtype=np.float32) * 0.8,
        )
    )

    selected = select_representative_sample([low_error_no_signal, nontrivial_signal])

    assert selected.sample_idx == 1
    assert selected.selection_reason == "balanced_visual_with_nontrivial_increment"
    assert selected.selection_metadata["positive_skill_candidate_count"] == 2
    assert selected.selection_metadata["signal_threshold"] > selected.selection_metadata[
        "filtered_below_signal_threshold"
    ][0]["true_increment_strength"]
    assert selected.selection_metadata["true_increment_strength"] >= selected.selection_metadata["signal_threshold"]


def test_select_representative_sample_balanced_visual_avoids_edge_concentrated_case():
    from scripts.analysis.plot_ref_pred_rmse_maps import (
        _sample_visual_selection_metrics,
        build_sample_map,
        select_representative_sample,
    )

    low_global_error_bad_left_edge_ref = np.ones((4, 4), dtype=np.float32)
    low_global_error_bad_left_edge_pred = np.ones((4, 4), dtype=np.float32) * 0.9
    low_global_error_bad_left_edge_pred[:, 0] = 1.45
    balanced_ref = np.ones((4, 4), dtype=np.float32) * 1.2
    balanced_pred = np.ones((4, 4), dtype=np.float32) * 0.9

    edge_concentrated = build_sample_map(
        _record(
            sample_idx=0,
            query_date="2023-01-01",
            surface_forecast=np.zeros((4, 4), dtype=np.float32),
            surface_ref=low_global_error_bad_left_edge_ref,
            surface_pred=low_global_error_bad_left_edge_pred,
            rootzone_forecast=np.zeros((4, 4), dtype=np.float32),
            rootzone_ref=low_global_error_bad_left_edge_ref,
            rootzone_pred=low_global_error_bad_left_edge_pred,
            metric_mask=np.ones((4, 4), dtype=np.float32),
            latitude_weight=np.ones((4, 4), dtype=np.float32),
        )
    )
    balanced = build_sample_map(
        _record(
            sample_idx=1,
            query_date="2023-01-02",
            surface_forecast=np.zeros((4, 4), dtype=np.float32),
            surface_ref=balanced_ref,
            surface_pred=balanced_pred,
            rootzone_forecast=np.zeros((4, 4), dtype=np.float32),
            rootzone_ref=balanced_ref,
            rootzone_pred=balanced_pred,
            metric_mask=np.ones((4, 4), dtype=np.float32),
            latitude_weight=np.ones((4, 4), dtype=np.float32),
        )
    )

    selected = select_representative_sample(
        [edge_concentrated, balanced],
        min_signal_quantile=0.0,
        min_skill_quantile=0.0,
    )

    assert selected.sample_idx == 1
    assert _sample_visual_selection_metrics(edge_concentrated)[
        "edge_error_concentration"
    ] > _sample_visual_selection_metrics(balanced)["edge_error_concentration"]
    assert selected.selection_metadata["edge_error_concentration"] == pytest.approx(
        _sample_visual_selection_metrics(balanced)["edge_error_concentration"]
    )


def test_lowest_rmse_selection_mode_remains_available_for_reproduction():
    from scripts.analysis.plot_ref_pred_rmse_maps import build_sample_map, select_representative_sample

    low_error = build_sample_map(
        _record(
            sample_idx=0,
            query_date="2023-01-01",
            surface_ref=np.ones((2, 2), dtype=np.float32),
            surface_pred=np.ones((2, 2), dtype=np.float32) * 0.9,
            rootzone_ref=np.ones((2, 2), dtype=np.float32),
            rootzone_pred=np.ones((2, 2), dtype=np.float32) * 0.9,
        )
    )
    higher_error = build_sample_map(
        _record(
            sample_idx=1,
            query_date="2023-01-02",
            surface_ref=np.ones((2, 2), dtype=np.float32),
            surface_pred=np.ones((2, 2), dtype=np.float32) * 0.5,
            rootzone_ref=np.ones((2, 2), dtype=np.float32),
            rootzone_pred=np.ones((2, 2), dtype=np.float32) * 0.5,
        )
    )

    selected = select_representative_sample(
        [higher_error, low_error],
        selection_mode="lowest_rmse_with_signal",
        min_signal_quantile=0.0,
        min_skill_quantile=0.0,
    )

    assert selected.sample_idx == 0
    assert selected.selection_reason == "lowest_rmse_with_nontrivial_increment"
    assert selected.selection_metadata["selection_mode"] == "lowest_rmse_with_signal"


def test_select_representative_sample_has_deterministic_no_skill_fallback():
    from scripts.analysis.plot_ref_pred_rmse_maps import build_sample_map, select_representative_sample

    weaker = build_sample_map(
        _record(
            sample_idx=0,
            query_date="2023-01-01",
            surface_ref=np.ones((2, 2), dtype=np.float32),
            surface_pred=np.ones((2, 2), dtype=np.float32) * 3.0,
            rootzone_ref=np.ones((2, 2), dtype=np.float32),
            rootzone_pred=np.ones((2, 2), dtype=np.float32) * 3.0,
        )
    )
    stronger = build_sample_map(
        _record(
            sample_idx=1,
            query_date="2023-01-02",
            surface_ref=np.ones((2, 2), dtype=np.float32) * 2.0,
            surface_pred=np.ones((2, 2), dtype=np.float32) * 5.0,
            rootzone_ref=np.ones((2, 2), dtype=np.float32) * 2.0,
            rootzone_pred=np.ones((2, 2), dtype=np.float32) * 5.0,
        )
    )

    selected = select_representative_sample([weaker, stronger])

    assert selected.sample_idx == 1
    assert selected.selection_reason == "fallback_high_true_increment_no_positive_skill"


def test_sample_metadata_records_best_skill_selection_scores(tmp_path):
    from scripts.analysis.plot_ref_pred_rmse_maps import (
        build_sample_map,
        sample_to_metadata,
        select_representative_sample,
    )

    weak = build_sample_map(
        _record(
            sample_idx=0,
            query_date="2023-01-01",
            surface_ref=np.ones((2, 2), dtype=np.float32) * 0.001,
            surface_pred=np.ones((2, 2), dtype=np.float32) * 0.001,
            rootzone_ref=np.ones((2, 2), dtype=np.float32) * 0.001,
            rootzone_pred=np.ones((2, 2), dtype=np.float32) * 0.001,
        )
    )
    selected = select_representative_sample(
        [
            weak,
            build_sample_map(
                _record(
                    sample_idx=1,
                    query_date="2023-01-02",
                    surface_ref=np.ones((2, 2), dtype=np.float32),
                    surface_pred=np.ones((2, 2), dtype=np.float32) * 0.8,
                    rootzone_ref=np.ones((2, 2), dtype=np.float32),
                    rootzone_pred=np.ones((2, 2), dtype=np.float32) * 0.8,
                )
            ),
        ]
    )

    metadata = sample_to_metadata(
        selected,
        prediction_record_path=tmp_path / "records.jsonl",
        prediction_record_hash="abc123",
        checkpoint=tmp_path / "missing.pt",
        output_files={"png": "figure.png"},
        candidate_review_files={"candidate_ranking_csv": "ranking.csv"},
        coordinate_metadata={"coordinate_mode": "grid"},
        crop_metadata={"crop_applied": False},
        outline_metadata={"outline_available": False},
        model_label="model",
        candidate_id="candidate",
        max_samples=2,
    )

    assert metadata["selection_usage"] == "visualization_only_not_model_selection"
    assert metadata["selection_reason"] == "balanced_visual_with_nontrivial_increment"
    assert metadata["selection_score"] == metadata["selection_scores"]["balanced_visual_score"]
    assert metadata["selection_scores"]["candidate_pool_size"] == 2
    assert metadata["selection_scores"]["eligible_candidate_count"] == 1
    assert metadata["selection_scores"]["selection_mode"] == "balanced_visual_with_signal"
    assert metadata["selection_scores"]["skill_quantile"] == 0.25
    assert metadata["selection_scores"]["skill_threshold"] is not None
    assert metadata["selection_scores"]["edge_error_concentration"] is not None
    assert metadata["candidate_review_files"]["candidate_ranking_csv"] == "ranking.csv"
    assert metadata["selection_scores"]["true_increment_strength"] > metadata["selection_scores"][
        "filtered_below_signal_threshold"
    ][0]["true_increment_strength"]
    assert metadata["figure_style"]["rmse_panels"] == "analysis_space_absolute_error_cartopy_map"
    assert "single target_eval case" in metadata["caption"]
    assert "not an aggregate" in metadata["caption"]


def test_crop_sample_to_bbox_reduces_arrays_and_recomputes_metrics():
    from scripts.analysis.plot_ref_pred_rmse_maps import build_sample_map, crop_sample_to_bbox

    sample = build_sample_map(
        _record(
            surface_ref=np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
            surface_pred=np.asarray([[1.0, 1.0], [1.0, 1.0]], dtype=np.float32),
            rootzone_ref=np.asarray([[2.0, 2.0], [6.0, 6.0]], dtype=np.float32),
            rootzone_pred=np.asarray([[1.0, 1.0], [3.0, 3.0]], dtype=np.float32),
        )
    )

    cropped = crop_sample_to_bbox(sample, {"y_start": 0, "y_end": 0, "x_start": 0, "x_end": 1})

    assert cropped.mask.shape == (1, 2)
    np.testing.assert_allclose(cropped.variables["surface"].ref, [[1.2, 2.2]])
    np.testing.assert_allclose(cropped.variables["surface"].pred, [[1.2, 1.2]])
    assert np.isclose(cropped.variables["surface"].model_wrmse, np.sqrt(2.0 / 3.0))


def test_lower_left_diagnostic_uses_cropped_coordinates_mask_and_latitude_weight():
    from scripts.analysis.plot_ref_pred_rmse_maps import (
        build_error_diagnostic_summary,
        build_sample_map,
    )

    forecast = np.zeros((4, 4), dtype=np.float32)
    ref = np.ones((4, 4), dtype=np.float32)
    pred = np.ones((4, 4), dtype=np.float32) * 0.25
    pred[2:, :2] = 2.25
    pred[3, 0] = 100.0
    metric_mask = np.ones((4, 4), dtype=np.float32)
    metric_mask[3, 0] = 0.0
    latitude_weight = np.ones((4, 4), dtype=np.float32)
    latitude_weight[2, 0] = 4.0

    samples = [
        build_sample_map(
            _record(
                surface_forecast=forecast,
                surface_ref=ref,
                surface_pred=pred,
                rootzone_forecast=forecast,
                rootzone_ref=ref * 2.0,
                rootzone_pred=pred * 2.0,
                metric_mask=metric_mask,
                latitude_weight=latitude_weight,
            )
        )
    ]

    summary = build_error_diagnostic_summary(samples, lower_left_fraction=0.5)
    surface = summary["variables"]["surface"]
    lower_left = surface["lower_left_window"]
    outside = surface["outside_lower_left_window"]

    assert summary["diagnostic_usage"] == "error_diagnostic_only_not_model_selection"
    assert lower_left["window"] == {"y_start": 2, "y_end": 4, "x_start": 0, "x_end": 2}
    assert lower_left["valid_pixel_count"] == 3
    assert lower_left["valid_pixel_fraction_of_panel"] == pytest.approx(3 / 15)
    assert lower_left["model_rmse"] > outside["model_rmse"]
    assert lower_left["mean_skill_vs_forecast"] < 0.0
    assert lower_left["mean_model_abs_error_rank_from_worst"] == 1
    assert np.isclose(
        lower_left["model_rmse"],
        np.sqrt((4.0 * 1.25**2 + 1.0 * 1.25**2 + 1.0 * 1.25**2) / 6.0),
    )


def test_region_outline_segments_from_mask_follow_outer_boundary():
    from scripts.analysis.plot_ref_pred_rmse_maps import region_outline_segments_from_mask

    lon = np.asarray(
        [
            [-115.0, -114.0, -113.0],
            [-115.0, -114.0, -113.0],
            [-115.0, -114.0, -113.0],
        ],
        dtype=np.float32,
    )
    lat = np.asarray(
        [
            [37.0, 37.0, 37.0],
            [36.0, 36.0, 36.0],
            [35.0, 35.0, 35.0],
        ],
        dtype=np.float32,
    )
    mask = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 1.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )

    segments = region_outline_segments_from_mask(mask, lon, lat)

    assert segments
    assert all(segment.shape == (2, 2) for segment in segments)
    segment_keys = {
        tuple(tuple(round(float(value), 3) for value in point) for point in segment)
        for segment in segments
    }
    assert ((-114.5, 36.5), (-113.5, 36.5)) in segment_keys
    assert ((-114.5, 36.5), (-114.5, 35.5)) in segment_keys


def test_create_figure_axes_uses_map_for_all_panels():
    cartopy = pytest.importorskip("cartopy.crs")
    from scripts.analysis.plot_ref_pred_rmse_maps import create_figure_axes

    fig, axes = create_figure_axes(use_map_axes=True)

    try:
        assert axes.shape == (2, 3)
        for ax in axes.reshape(-1):
            assert hasattr(ax, "projection")
            assert isinstance(ax.projection, cartopy.PlateCarree)
    finally:
        plt.close(fig)


def test_error_panel_can_be_drawn_as_subtle_map_with_region_outline():
    ccrs = pytest.importorskip("cartopy.crs")
    from scripts.analysis.plot_ref_pred_rmse_maps import _plot_map_panel

    lon = np.asarray([[-115.0, -114.0], [-115.0, -114.0]], dtype=np.float32)
    lat = np.asarray([[37.0, 37.0], [36.0, 36.0]], dtype=np.float32)
    values = np.asarray([[0.01, 0.02], [0.03, np.nan]], dtype=np.float32)
    region_mask = np.asarray([[1.0, 1.0], [1.0, 0.0]], dtype=np.float32)
    fig, ax = plt.subplots(subplot_kw={"projection": ccrs.PlateCarree()})

    try:
        _plot_map_panel(
            ax,
            lon,
            lat,
            values,
            title="",
            cmap="magma",
            vmin=0.0,
            vmax=1.0,
            region_mask=region_mask,
            show_ylabel=False,
            subtle_boundaries=True,
            show_grid_labels=False,
        )

        assert len(ax.lines) > 0
        assert list(ax.get_xticks()) == []
        assert list(ax.get_yticks()) == []
    finally:
        plt.close(fig)


def test_map_gridline_ticks_are_coarse_for_compact_panels():
    from scripts.analysis.plot_ref_pred_rmse_maps import _coarse_degree_ticks

    lon_ticks = _coarse_degree_ticks(-115.2, -108.8)
    lat_ticks = _coarse_degree_ticks(31.4, 37.1)

    assert lon_ticks == [-114.0, -112.0, -110.0]
    assert lat_ticks == [32.0, 34.0, 36.0]
    assert len(lon_ticks) <= 4
    assert len(lat_ticks) <= 4


def test_colorbar_style_uses_compact_paper_sizing():
    from scripts.analysis.plot_ref_pred_rmse_maps import COLORBAR_STYLE

    assert set(COLORBAR_STYLE) == {
        "width",
        "height",
        "x_offset",
        "tick_labelsize",
        "tick_width",
        "tick_length",
        "outline_width",
    }
    assert COLORBAR_STYLE["width"] == "2.4%"
    assert COLORBAR_STYLE["height"] == "68%"
    assert COLORBAR_STYLE["x_offset"] == 1.03
    assert COLORBAR_STYLE["tick_labelsize"] <= 5.6
    assert COLORBAR_STYLE["tick_length"] <= 1.4
    assert COLORBAR_STYLE["outline_width"] <= 0.5


def test_compact_panel_colorbar_geometry_matches_anchor_panel():
    from scripts.analysis.plot_ref_pred_rmse_maps import _add_compact_panel_colorbar

    fig, axes = plt.subplots(1, 2, figsize=(4.0, 2.2))
    try:
        meshes = [
            ax.imshow(np.arange(4, dtype=np.float32).reshape(2, 2), vmin=0.0, vmax=3.0)
            for ax in axes
        ]

        colorbars = [
            _add_compact_panel_colorbar(fig, mesh, anchor_ax=ax)
            for mesh, ax in zip(meshes, axes)
        ]
        fig.canvas.draw()

        renderer = fig.canvas.get_renderer()
        anchor_bbox = axes[0].get_window_extent(renderer)
        first_bbox = colorbars[0].ax.get_window_extent(renderer)
        second_bbox = colorbars[1].ax.get_window_extent(renderer)

        assert np.isclose(first_bbox.width, second_bbox.width, atol=0.5)
        assert np.isclose(first_bbox.height, second_bbox.height, atol=0.5)
        assert first_bbox.height < anchor_bbox.height * 0.75
        assert second_bbox.height < anchor_bbox.height * 0.75
        assert first_bbox.x0 > anchor_bbox.x1
    finally:
        plt.close(fig)


def test_shared_soil_moisture_colorbar_leaves_gutter_before_error_panel():
    from scripts.analysis.plot_ref_pred_rmse_maps import (
        _add_compact_panel_colorbar,
        create_figure_axes,
    )

    fig, axes = create_figure_axes(use_map_axes=False)
    try:
        mesh = axes[0, 1].imshow(
            np.arange(4, dtype=np.float32).reshape(2, 2),
            vmin=0.0,
            vmax=0.32,
        )
        cbar = _add_compact_panel_colorbar(fig, mesh, anchor_ax=axes[0, 1])
        fig.canvas.draw()

        renderer = fig.canvas.get_renderer()
        colorbar_tight_bbox = cbar.ax.get_tightbbox(renderer)
        error_panel_bbox = axes[0, 2].get_window_extent(renderer)
        panel_width = axes[0, 1].get_window_extent(renderer).width

        assert error_panel_bbox.x0 - colorbar_tight_bbox.x1 >= panel_width * 0.18
    finally:
        plt.close(fig)
