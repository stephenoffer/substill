"""Loss schedule activates the right objective at each step fraction."""

from __future__ import annotations

from fasd.losses.subspace import Schedule, ScheduleStage, default_schedule


def test_default_schedule_stages():
    s = default_schedule()
    assert list(s.objective_weights(0.0)) == ["gram"]
    assert list(s.objective_weights(0.05)) == ["gram"]
    assert list(s.objective_weights(0.20)) == ["cka"]
    assert list(s.objective_weights(0.50)) == ["procrustes"]
    assert list(s.objective_weights(0.99)) == ["procrustes"]


def test_feature_weight_interpolation():
    s = default_schedule()
    # Linear fade: (0.8, 1.0) → (1.0, 0.0)
    assert abs(s.feature_weight(0.80) - 1.0) < 1e-6
    assert abs(s.feature_weight(1.00) - 0.0) < 1e-6
    mid = s.feature_weight(0.90)
    assert 0.0 < mid < 1.0


def test_custom_schedule_missing_objective_returns_last():
    s = Schedule(
        stages=[ScheduleStage(0.0, 1.0, {"gram": 1.0})],
    )
    assert s.objective_weights(1.5) == {"gram": 1.0}
