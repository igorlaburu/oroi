from oroi import DynamicsConfig


def test_defaults_are_sane():
    config = DynamicsConfig()
    assert 0 < config.decay < 1
    assert 0 < config.damping <= 1
    assert config.activation_floor < config.coact_threshold < config.retrieval_threshold
    assert config.window_keep_turns < config.window_max_turns
