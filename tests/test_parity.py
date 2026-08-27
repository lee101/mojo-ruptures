from __future__ import annotations

import inspect

import numpy as np
import pytest
import ruptures as upstream
import ruptures.metrics as upstream_metrics

import mojoruptures as mojo
from mojoruptures.exceptions import BadSegmentationParameters, NotEnoughPoints


@pytest.fixture(scope="module")
def signal_1d():
    rng = np.random.default_rng(12)
    return np.r_[
        rng.normal(-2.0, 0.7, 43),
        rng.normal(3.0, 0.7, 51),
        rng.normal(0.5, 0.7, 39),
    ]


@pytest.fixture(scope="module")
def signal_3d():
    rng = np.random.default_rng(31)
    return np.vstack(
        [
            rng.normal(-1.0, 1.0, (45, 3)),
            rng.normal(2.5, 1.0, (55, 3)),
            rng.normal(0.0, 1.0, (37, 3)),
        ]
    )


@pytest.mark.parametrize("multivariate", [False, True])
def test_cost_l2_segment_parity(signal_1d, signal_3d, multivariate):
    signal = signal_3d if multivariate else signal_1d
    ours = mojo.CostL2().fit(signal)
    theirs = upstream.costs.CostL2().fit(signal)
    segments = [(0, len(signal)), (7, 38), (40, 94), (94, len(signal))]
    for start, end in segments:
        assert ours.error(start, end) == pytest.approx(
            theirs.error(start, end), rel=1e-12, abs=1e-11
        )


def test_cost_l2_many_parity(signal_3d):
    starts = np.array([0, 5, 31, 77, 100])
    ends = np.array([20, 49, 83, 120, len(signal_3d)])
    ours = mojo.CostL2().fit(signal_3d).error_many(starts, ends)
    theirs = np.array(
        [
            upstream.costs.CostL2().fit(signal_3d).error(start, end)
            for start, end in zip(starts, ends)
        ]
    )
    assert np.allclose(ours, theirs, rtol=1e-12, atol=1e-11)


def test_cost_l2_sum_of_costs_parity(signal_3d):
    bkps = [45, 100, len(signal_3d)]
    ours = mojo.CostL2().fit(signal_3d).sum_of_costs(bkps)
    theirs = upstream.costs.CostL2().fit(signal_3d).sum_of_costs(bkps)
    assert ours == pytest.approx(theirs, rel=1e-12, abs=1e-11)


def test_cost_l2_simd_tail_parity():
    rng = np.random.default_rng(41)
    signal = rng.normal(size=(257, 7))
    ours = mojo.CostL2().fit(signal)
    theirs = upstream.costs.CostL2().fit(signal)
    assert ours.error(13, 241) == pytest.approx(
        theirs.error(13, 241), rel=1e-12, abs=1e-11
    )


@pytest.mark.parametrize("samples", [4032, 4033])
def test_cost_l2_large_input_parity(samples):
    rng = np.random.default_rng(samples)
    signal = rng.normal(size=(samples, 65))
    ours = mojo.CostL2().fit(signal)
    theirs = upstream.costs.CostL2().fit(signal)
    assert ours.error(19, samples - 17) == pytest.approx(
        theirs.error(19, samples - 17), rel=1e-12, abs=1e-10
    )


def test_cost_l2_parallel_threshold_parity():
    rng = np.random.default_rng(43)
    signal = rng.normal(size=(250_000, 64))
    ours = mojo.CostL2().fit(signal)
    theirs = upstream.costs.CostL2().fit(signal)
    assert ours.error(113, 249_731) == pytest.approx(
        theirs.error(113, 249_731), rel=1e-12, abs=1e-8
    )


def test_cost_l2_large_offset_stability():
    rng = np.random.default_rng(4)
    signal = 1e12 + rng.normal(size=(500, 2))
    ours = mojo.CostL2().fit(signal)
    theirs = upstream.costs.CostL2().fit(signal)
    assert ours.error(33, 477) == pytest.approx(
        theirs.error(33, 477), rel=2e-7
    )


def test_cost_l2_short_segment_raises(signal_1d):
    cost = mojo.CostL2().fit(signal_1d)
    with pytest.raises(NotEnoughPoints):
        cost.error(5, 5)


@pytest.mark.parametrize(
    "signal",
    [
        np.empty(0),
        np.empty((3, 0)),
        np.array([1.0, np.nan]),
        np.array([1.0, np.inf]),
        np.array([1.0 + 2.0j]),
        np.array([2**53 + 1], dtype=np.int64),
    ],
)
def test_cost_l2_rejects_unsafe_native_inputs(signal):
    with pytest.raises((TypeError, ValueError)):
        mojo.CostL2().fit(signal)


def test_error_many_rejects_silent_index_narrowing(signal_1d):
    cost = mojo.CostL2().fit(signal_1d)
    with pytest.raises(TypeError):
        cost.error(0.5, 10)
    with pytest.raises(TypeError):
        cost.error_many([0.5], [10.5])
    with pytest.raises(OverflowError):
        cost.error_many(
            np.array([2**63], dtype=np.uint64),
            np.array([2**63], dtype=np.uint64),
        )


@pytest.mark.parametrize("jump", [1, 4, 7])
@pytest.mark.parametrize("n_bkps", [1, 2, 3])
def test_dynp_parity(signal_3d, jump, n_bkps):
    ours = mojo.Dynp(model="l2", min_size=3, jump=jump).fit_predict(
        signal_3d, n_bkps
    )
    theirs = upstream.Dynp(model="l2", min_size=3, jump=jump).fit_predict(
        signal_3d, n_bkps
    )
    assert ours == theirs


@pytest.mark.parametrize("jump", [1, 5, 8])
@pytest.mark.parametrize("penalty", [2.0, 12.0, 35.0])
def test_pelt_parity(signal_1d, jump, penalty):
    ours = mojo.Pelt(model="l2", min_size=2, jump=jump).fit_predict(
        signal_1d, penalty
    )
    theirs = upstream.Pelt(model="l2", min_size=2, jump=jump).fit_predict(
        signal_1d, penalty
    )
    assert ours == theirs


@pytest.mark.parametrize("jump", [1, 5])
@pytest.mark.parametrize("n_bkps", [1, 2, 4])
def test_binseg_breakpoint_count_parity(signal_3d, jump, n_bkps):
    ours = mojo.Binseg(model="l2", min_size=2, jump=jump).fit_predict(
        signal_3d, n_bkps=n_bkps
    )
    theirs = upstream.Binseg(model="l2", min_size=2, jump=jump).fit_predict(
        signal_3d, n_bkps=n_bkps
    )
    assert ours == theirs


@pytest.mark.parametrize(
    "argument,value",
    [("pen", 10.0), ("pen", 40.0), ("epsilon", 80.0), ("epsilon", 250.0)],
)
def test_binseg_other_stopping_rules(signal_3d, argument, value):
    kwargs = {argument: value}
    ours = mojo.Binseg(jump=4).fit_predict(signal_3d, **kwargs)
    theirs = upstream.Binseg(jump=4).fit_predict(signal_3d, **kwargs)
    assert ours == theirs


@pytest.mark.parametrize("estimator", [mojo.Dynp, mojo.Pelt, mojo.Binseg])
def test_custom_cost_fallback(signal_1d, estimator):
    upstream_cost = upstream.costs.CostL2()
    if estimator is mojo.Dynp:
        ours = estimator(custom_cost=upstream_cost, jump=5).fit_predict(signal_1d, 2)
        theirs = upstream.Dynp(custom_cost=upstream.costs.CostL2(), jump=5).fit_predict(
            signal_1d, 2
        )
    elif estimator is mojo.Pelt:
        ours = estimator(custom_cost=upstream_cost, jump=5).fit_predict(signal_1d, 12)
        theirs = upstream.Pelt(custom_cost=upstream.costs.CostL2(), jump=5).fit_predict(
            signal_1d, 12
        )
    else:
        ours = estimator(custom_cost=upstream_cost, jump=5).fit_predict(
            signal_1d, n_bkps=2
        )
        theirs = upstream.Binseg(
            custom_cost=upstream.costs.CostL2(), jump=5
        ).fit_predict(signal_1d, n_bkps=2)
    assert ours == theirs


def test_impossible_segmentation(signal_1d):
    with pytest.raises(BadSegmentationParameters):
        mojo.Dynp(min_size=30, jump=5).fit(signal_1d).predict(10)


@pytest.mark.parametrize("estimator", [mojo.Dynp, mojo.Pelt, mojo.Binseg])
def test_detectors_reject_nonpositive_native_loop_parameters(estimator):
    with pytest.raises(ValueError):
        estimator(jump=0)
    with pytest.raises(ValueError):
        estimator(min_size=0)


def test_detectors_reject_narrowed_or_nonfinite_arguments(signal_1d):
    with pytest.raises(TypeError):
        mojo.Dynp().fit(signal_1d).predict(1.5)
    with pytest.raises(ValueError):
        mojo.Pelt().fit(signal_1d).predict(np.nan)
    with pytest.raises(ValueError):
        mojo.Binseg().fit(signal_1d).predict(pen=np.inf)


def test_constructor_signatures_match_upstream():
    for ours, theirs in [
        (mojo.CostL2, upstream.costs.CostL2),
        (mojo.Dynp, upstream.Dynp),
        (mojo.Pelt, upstream.Pelt),
        (mojo.Binseg, upstream.Binseg),
    ]:
        assert inspect.signature(ours) == inspect.signature(theirs)


@pytest.mark.parametrize(
    "name,kwargs",
    [
        ("pw_constant", {"n_features": 3, "noise_std": 0.5}),
        ("pw_linear", {"n_features": 2, "noise_std": 0.2}),
        ("pw_normal", {}),
        ("pw_wavy", {"noise_std": 0.3}),
    ],
)
def test_dataset_generator_parity(name, kwargs):
    ours_signal, ours_bkps = getattr(mojo, name)(
        n_samples=120, n_bkps=3, seed=9, **kwargs
    )
    their_signal, their_bkps = getattr(upstream, name)(
        n_samples=120, n_bkps=3, seed=9, **kwargs
    )
    assert ours_bkps == their_bkps
    assert np.array_equal(ours_signal, their_signal)


def test_draw_bkps_parity():
    assert mojo.draw_bkps(1000, 8, seed=73) == upstream.utils.draw_bkps(
        1000, 8, seed=73
    )


@pytest.mark.parametrize(
    "name,args",
    [
        ("hausdorff", ([20, 48, 100], [18, 53, 100])),
        ("meantime", ([20, 48, 100], [18, 53, 100])),
        ("precision_recall", ([20, 48, 100], [18, 53, 100])),
        ("randindex", ([20, 48, 100], [18, 53, 100])),
    ],
)
def test_metric_parity(name, args):
    ours = getattr(mojo, name)(*args)
    theirs = getattr(upstream_metrics, name)(*args)
    assert ours == pytest.approx(theirs)
