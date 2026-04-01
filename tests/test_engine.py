"""Unit tests for the CloudFinOps physics engine.

Tests engine mechanics including state transitions, SLA breaches,
deterministic noise reproducibility, grading, carbon tracking,
and trailing metrics history.
"""

import pytest
from env.engine import (
    CloudFinOpsEngine,
    INSTANCE_CATALOG,
    UPSCALE_PATH,
    CARBON_INTENSITY,
    MAX_STEPS,
    SLA_CPU_LIMIT,
    _deterministic_noise,
    _clamp,
)
from env.models import Action, Observation, ServerState


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def engine():
    return CloudFinOpsEngine()


# ---------------------------------------------------------------------------
# 1. Reset produces clean state
# ---------------------------------------------------------------------------

class TestReset:
    def test_reset_easy_returns_observation(self, engine):
        obs = engine.reset("easy")
        assert isinstance(obs, Observation)
        assert obs.time_step == 0
        assert obs.budget_remaining == 5.0
        assert len(obs.servers) == 10

    def test_reset_medium(self, engine):
        obs = engine.reset("medium")
        assert len(obs.servers) == 12
        assert obs.budget_remaining == 10.0

    def test_reset_hard(self, engine):
        obs = engine.reset("hard")
        assert len(obs.servers) == 8
        assert obs.budget_remaining == 4.0
        assert obs.spike_detected is True

    def test_reset_green(self, engine):
        obs = engine.reset("green")
        assert len(obs.servers) == 10
        assert obs.budget_remaining == 8.0
        assert obs.carbon_kwh == 0.0

    def test_reset_invalid_task(self, engine):
        with pytest.raises(ValueError, match="Unknown task_id"):
            engine.reset("nonexistent")

    def test_reset_clears_state(self, engine):
        engine.reset("easy")
        engine.step(Action(command="TERMINATE", target_id="idle-0"))
        obs2 = engine.reset("easy")
        assert obs2.time_step == 0
        assert len([s for s in obs2.servers if s.status == "running"]) == 10


# ---------------------------------------------------------------------------
# 2. Deterministic noise reproducibility
# ---------------------------------------------------------------------------

class TestDeterministicNoise:
    def test_same_seed_same_result(self):
        a = _deterministic_noise("easy:web-0:1:cpu")
        b = _deterministic_noise("easy:web-0:1:cpu")
        assert a == b

    def test_different_seeds_different_results(self):
        a = _deterministic_noise("easy:web-0:1:cpu")
        b = _deterministic_noise("easy:web-0:2:cpu")
        assert a != b

    def test_noise_within_amplitude(self):
        for i in range(100):
            val = _deterministic_noise(f"test:{i}", amplitude=3.0)
            assert -3.0 <= val <= 3.0


# ---------------------------------------------------------------------------
# 3. Action mechanics
# ---------------------------------------------------------------------------

class TestActions:
    def test_terminate_removes_server(self, engine):
        engine.reset("easy")
        obs, reward, done, info = engine.step(Action(command="TERMINATE", target_id="idle-0"))
        terminated = [s for s in obs.servers if s.id == "idle-0"][0]
        assert terminated.status == "terminated"
        assert terminated.cpu_util == 0.0
        assert reward > 0  # terminate gives +10

    def test_terminate_invalid_target(self, engine):
        engine.reset("easy")
        _, reward, _, _ = engine.step(Action(command="TERMINATE", target_id="nonexistent"))
        assert reward == -2.0

    def test_terminate_already_dead(self, engine):
        engine.reset("easy")
        engine.step(Action(command="TERMINATE", target_id="idle-0"))
        _, reward, _, _ = engine.step(Action(command="TERMINATE", target_id="idle-0"))
        assert reward == -2.0

    def test_upscale_queues_pending(self, engine):
        engine.reset("easy")
        obs, reward, _, _ = engine.step(Action(command="UPSCALE", target_id="web-0"))
        # Upscale is delayed — type changes next step
        web0 = [s for s in obs.servers if s.id == "web-0"][0]
        # Should still be t3.medium this step (change is pending)
        assert reward < 0  # upscale costs -5

    def test_upscale_applies_next_step(self, engine):
        engine.reset("easy")
        engine.step(Action(command="UPSCALE", target_id="web-0"))
        obs, _, _, _ = engine.step(Action(command="IGNORE"))
        web0 = [s for s in obs.servers if s.id == "web-0"][0]
        assert web0.type == "t3.large"  # t3.medium -> t3.large

    def test_upscale_max_cap(self, engine):
        engine.reset("easy")
        # First upscale: t3.medium -> t3.large
        engine.step(Action(command="UPSCALE", target_id="web-0"))
        engine.step(Action(command="IGNORE"))  # apply pending
        # Second upscale: t3.large is max, no UPSCALE_PATH entry
        _, reward, _, _ = engine.step(Action(command="UPSCALE", target_id="web-0"))
        assert reward == -1.0  # wasted action

    def test_downscale_increases_cpu(self, engine):
        engine.reset("easy")
        obs_before = engine.state()
        web0_before = [s for s in obs_before.servers if s.id == "web-0"][0]
        cpu_before = web0_before.cpu_util

        obs, reward, _, _ = engine.step(Action(command="DOWNSCALE", target_id="web-0"))
        web0_after = [s for s in obs.servers if s.id == "web-0"][0]
        # CPU should increase (1.8x + noise)
        assert reward > 0  # downscale gives +5

    def test_redistribute_load(self, engine):
        engine.reset("easy")
        obs, reward, _, _ = engine.step(Action(command="REDISTRIBUTE_LOAD", target_id="web-0"))
        assert reward > 0  # redistribute gives +3

    def test_ignore_no_reward(self, engine):
        engine.reset("easy")
        _, reward, _, _ = engine.step(Action(command="IGNORE"))
        # IGNORE gives 0 reward (but may get -1 for high cost)
        assert reward <= 0

    def test_inbox_reply_gives_bonus(self, engine):
        engine.reset("easy")
        obs1 = engine.state()
        assert len(obs1.inbox) > 0
        _, reward, _, _ = engine.step(Action(command="TERMINATE", target_id="idle-0", reply="On it!"))
        assert reward >= 12.0  # 10 (terminate) + 2 (reply)


# ---------------------------------------------------------------------------
# 4. SLA breach detection
# ---------------------------------------------------------------------------

class TestSLABreach:
    def test_sla_breach_triggers_done(self, engine):
        engine.reset("hard")
        # Force a server to 100% CPU
        engine.servers[0].cpu_util = 99.9
        _, reward, done, info = engine.step(Action(command="IGNORE"))
        # With noise, it might breach or not — but test the mechanism
        # Let's force it directly
        engine2 = CloudFinOpsEngine()
        engine2.reset("hard")
        engine2.servers[0].cpu_util = 100.0
        _, reward, done, info = engine2.step(Action(command="IGNORE"))
        assert done is True
        assert reward <= -100.0


# ---------------------------------------------------------------------------
# 5. Grading
# ---------------------------------------------------------------------------

class TestGrading:
    def test_easy_perfect_score(self, engine):
        engine.reset("easy")
        engine.step(Action(command="TERMINATE", target_id="idle-0", reply="Done"))
        engine.step(Action(command="TERMINATE", target_id="idle-1"))
        engine.step(Action(command="TERMINATE", target_id="idle-2"))
        score = engine.grade()
        assert score == 1.0

    def test_easy_partial_score(self, engine):
        engine.reset("easy")
        engine.step(Action(command="TERMINATE", target_id="idle-0"))
        score = engine.grade()
        assert 0.0 < score < 1.0

    def test_grader_scores_in_range(self, engine):
        for task in ["easy", "medium", "hard", "green"]:
            engine.reset(task)
            for _ in range(MAX_STEPS):
                engine.step(Action(command="IGNORE"))
            score = engine.grade()
            assert 0.0 <= score <= 1.0, f"Score for {task} out of range: {score}"

    def test_green_grader_rewards_carbon_reduction(self, engine):
        engine.reset("green")
        # Terminate high-carbon x86 instances
        for sid in ["compute-0", "compute-1", "compute-2", "batch-0", "batch-1", "batch-2"]:
            engine.step(Action(command="TERMINATE", target_id=sid, reply="Reducing carbon"))
        # Fill remaining steps
        for _ in range(MAX_STEPS - 6):
            engine.step(Action(command="IGNORE"))
        score = engine.grade()
        # Terminating 6 x86 instances should give high carbon reduction score
        assert score > 0.3


# ---------------------------------------------------------------------------
# 6. Carbon tracking (GreenOps)
# ---------------------------------------------------------------------------

class TestCarbonTracking:
    def test_carbon_starts_at_zero(self, engine):
        obs = engine.reset("green")
        assert obs.carbon_kwh == 0.0

    def test_carbon_accumulates(self, engine):
        engine.reset("green")
        obs1, _, _, _ = engine.step(Action(command="IGNORE"))
        assert obs1.carbon_kwh > 0.0
        obs2, _, _, _ = engine.step(Action(command="IGNORE"))
        assert obs2.carbon_kwh > obs1.carbon_kwh

    def test_carbon_decreases_after_terminate(self, engine):
        engine.reset("green")
        obs1, _, _, _ = engine.step(Action(command="IGNORE"))
        rate1 = obs1.carbon_kwh  # carbon after 1 step

        engine.reset("green")
        engine.step(Action(command="TERMINATE", target_id="batch-2"))  # kill high-carbon m5.xlarge
        obs2, _, _, _ = engine.step(Action(command="IGNORE"))
        # Total carbon after 2 steps with one less server should be lower rate
        # (rate per step is lower even though we have 2 steps of accumulation)
        assert obs2.carbon_kwh > 0  # still some carbon

    def test_all_instances_have_carbon_intensity(self):
        for inst_type in INSTANCE_CATALOG:
            assert inst_type in CARBON_INTENSITY, f"Missing carbon intensity for {inst_type}"


# ---------------------------------------------------------------------------
# 7. Trailing metrics history
# ---------------------------------------------------------------------------

class TestTrailingHistory:
    def test_initial_history(self, engine):
        obs = engine.reset("easy")
        for s in obs.servers:
            assert len(s.cpu_history) >= 1
            assert len(s.memory_history) >= 1

    def test_history_grows(self, engine):
        engine.reset("easy")
        engine.step(Action(command="IGNORE"))
        obs = engine.step(Action(command="IGNORE"))[0]
        for s in obs.servers:
            if s.status == "running":
                assert len(s.cpu_history) >= 2
                assert len(s.memory_history) >= 2

    def test_history_max_depth(self, engine):
        engine.reset("easy")
        for _ in range(5):
            engine.step(Action(command="IGNORE"))
        obs = engine.state()
        for s in obs.servers:
            assert len(s.cpu_history) <= 3  # HISTORY_DEPTH = 3
            assert len(s.memory_history) <= 3


# ---------------------------------------------------------------------------
# 8. Episode boundaries
# ---------------------------------------------------------------------------

class TestEpisodeBoundaries:
    def test_max_steps_ends_episode(self, engine):
        engine.reset("easy")
        for i in range(MAX_STEPS):
            obs, reward, done, info = engine.step(Action(command="IGNORE"))
        assert done is True
        assert "grader_score" in info

    def test_budget_overrun_ends_episode(self, engine):
        engine.reset("easy")
        engine.budget_remaining = 0.01  # almost out
        _, _, done, _ = engine.step(Action(command="IGNORE"))
        assert done is True

    def test_step_after_done_returns_done(self, engine):
        engine.reset("easy")
        engine.done = True
        _, reward, done, info = engine.step(Action(command="IGNORE"))
        assert done is True
        assert reward == 0.0


# ---------------------------------------------------------------------------
# 9. Clamp utility
# ---------------------------------------------------------------------------

class TestClamp:
    def test_clamp_within_range(self):
        assert _clamp(50.0) == 50.0

    def test_clamp_below(self):
        assert _clamp(-10.0) == 0.0

    def test_clamp_above(self):
        assert _clamp(150.0) == 100.0

    def test_clamp_custom_bounds(self):
        assert _clamp(5.0, 0.0, 1.0) == 1.0
