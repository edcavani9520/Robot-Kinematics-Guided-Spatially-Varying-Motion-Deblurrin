import argparse
from dataclasses import FrozenInstanceError

import pytest

import pipeline

from cli_config import (
    RLConfig,
    TVConfig,
    WienerConfig,
    add_common_arguments,
    add_method_subcommands,
    config_from_args,
    finite_nonnegative_float,
    finite_positive_float,
    nonnegative_int,
    positive_int,
    resolve_psf_parameters,
    require_bool,
    require_nonnegative_int_value,
    require_optional_positive_int_value,
    estimate_focal_length_from_diagonal_fov,
)


def make_parser(*, h5_required=True, default_h5=None):
    parser = argparse.ArgumentParser()
    add_common_arguments(
        parser, h5_required=h5_required, default_h5=default_h5
    )
    add_method_subcommands(parser)
    return parser


def test_global_arguments_before_wiener_subcommand_build_wiener_config():
    args = make_parser().parse_args(
        ["--h5", "episode.h5", "wiener", "--K", ".03", "--adaptive-k"]
    )

    assert config_from_args(args) == WienerConfig(K=0.03, adaptive_k=True)


def test_common_argument_abbreviations_are_rejected():
    with pytest.raises(SystemExit) as exc_info:
        make_parser().parse_args(["--h5", "episode.h5", "--dep", ".5", "wiener"])

    assert exc_info.value.code == 2


def test_method_argument_abbreviations_are_rejected():
    with pytest.raises(SystemExit) as exc_info:
        make_parser(default_h5="episode.h5").parse_args(["rl", "--rl-i", "5"])

    assert exc_info.value.code == 2


def test_common_arguments_must_precede_method_subcommand():
    with pytest.raises(SystemExit) as exc_info:
        make_parser().parse_args(["wiener", "--h5", "episode.h5"])

    assert exc_info.value.code == 2


def test_rl_subcommand_builds_rl_config_and_rejects_wiener_option():
    parser = make_parser(default_h5="episode.h5")

    args = parser.parse_args(["rl", "--rl-iters", "5"])
    assert config_from_args(args) == RLConfig(iterations=5)

    with pytest.raises(SystemExit) as exc_info:
        parser.parse_args(["rl", "--K", ".03"])
    assert exc_info.value.code == 2


def test_tv_subcommand_builds_tv_config_and_rejects_rl_option():
    parser = make_parser(default_h5="episode.h5")

    args = parser.parse_args(["tv", "--tv-lam", ".002"])
    assert config_from_args(args) == TVConfig(lam=0.002)

    with pytest.raises(SystemExit) as exc_info:
        parser.parse_args(["tv", "--rl-iters", "5"])
    assert exc_info.value.code == 2


def test_method_subcommand_is_required():
    with pytest.raises(SystemExit) as exc_info:
        make_parser(default_h5="episode.h5").parse_args([])

    assert exc_info.value.code == 2


def test_common_arguments_have_strict_types_and_expected_defaults():
    parser = make_parser(default_h5="default.h5")
    args = parser.parse_args(["wiener"])

    assert args.h5 == "default.h5"
    assert args.depth == 0.5
    assert args.exposure is None
    assert args.fx is None
    assert args.fy is None
    assert args.psf_sigma == 0.0

    with pytest.raises(SystemExit) as exc_info:
        parser.parse_args(["--depth", "0", "wiener"])
    assert exc_info.value.code == 2


def test_camera_focal_lengths_must_be_provided_together():
    parser = make_parser(default_h5="default.h5")
    args = parser.parse_args(["--fx", "300", "wiener"])

    with pytest.raises(ValueError, match="together"):
        config_from_args(args)


def test_pipeline_reports_unpaired_focal_length_as_cli_error():
    with pytest.raises(SystemExit) as exc_info:
        pipeline._parse_args(["--h5", "episode.h5", "--fx", "300", "wiener"])

    assert exc_info.value.code == 2


def test_psf_parameters_use_h5_intrinsics_and_confirmed_exposure_default():
    config = resolve_psf_parameters(
        {
            "camera_intrinsics": {"fx": 301.0, "fy": 302.0, "cx": 159.5, "cy": 119.5},
            "exposure_seconds": None,
        },
        depth=0.5,
        fx=None,
        fy=None,
        exposure=None,
        psf_sigma=0.0,
    )

    assert config.fx == 301.0
    assert config.fy == 302.0
    assert config.exposure == 0.01


def test_psf_parameters_estimate_missing_focal_length_from_75_degree_diagonal_fov():
    config = resolve_psf_parameters(
        {
            "W": 320,
            "H": 240,
            "camera_intrinsics": None,
            "exposure_seconds": None,
        },
        depth=0.5,
        fx=None,
        fy=None,
        exposure=None,
        psf_sigma=0.0,
    )

    expected = estimate_focal_length_from_diagonal_fov(320, 240, 75.0)
    assert config.fx == pytest.approx(expected)
    assert config.fy == pytest.approx(expected)
    assert config.intrinsics_source == "estimated_diagonal_fov_75deg"


@pytest.mark.parametrize("value", [0, -1, 1.5, True])
def test_direct_optional_frame_limit_rejects_cli_bypasses(value):
    with pytest.raises((TypeError, ValueError), match="max_frames"):
        require_optional_positive_int_value("max_frames", value)


@pytest.mark.parametrize("value", [-1, 1.5, True])
def test_direct_frame_index_rejects_cli_bypasses(value):
    with pytest.raises((TypeError, ValueError), match="frame_idx"):
        require_nonnegative_int_value("frame_idx", value)


@pytest.mark.parametrize("value", [0, 1, "true", None])
def test_direct_boolean_option_requires_real_bool(value):
    with pytest.raises(TypeError, match="overwrite"):
        require_bool("overwrite", value)


def test_h5_can_be_required_or_provided_by_default():
    required_parser = make_parser(h5_required=True)
    with pytest.raises(SystemExit) as exc_info:
        required_parser.parse_args(["wiener"])
    assert exc_info.value.code == 2

    default_parser = make_parser(h5_required=False, default_h5="fallback.h5")
    assert default_parser.parse_args(["wiener"]).h5 == "fallback.h5"


def test_config_from_args_rejects_a_missing_or_unknown_method():
    with pytest.raises(ValueError):
        config_from_args(argparse.Namespace())

    with pytest.raises(ValueError):
        config_from_args(argparse.Namespace(method="unknown"))


@pytest.mark.parametrize("value", ["0", "-1", "nan", "inf", "-inf"])
def test_finite_positive_float_rejects_invalid_domain(value):
    with pytest.raises(argparse.ArgumentTypeError):
        finite_positive_float(value)


@pytest.mark.parametrize("value", ["0.01", "3.5"])
def test_finite_positive_float_accepts_positive_finite_values(value):
    assert finite_positive_float(value) == float(value)


@pytest.mark.parametrize("value", ["-1", "nan", "inf", "-inf"])
def test_finite_nonnegative_float_rejects_invalid_domain(value):
    with pytest.raises(argparse.ArgumentTypeError):
        finite_nonnegative_float(value)


def test_finite_nonnegative_float_accepts_zero_and_positive_finite_values():
    assert finite_nonnegative_float("0") == 0.0
    assert finite_nonnegative_float("3.5") == 3.5


@pytest.mark.parametrize("value", ["0", "-1", "1.5"])
def test_positive_int_rejects_nonpositive_or_noninteger(value):
    with pytest.raises(argparse.ArgumentTypeError):
        positive_int(value)


@pytest.mark.parametrize("value", ["1", "12"])
def test_positive_int_accepts_positive_integers(value):
    assert positive_int(value) == int(value)


@pytest.mark.parametrize("value", ["-1", "1.5"])
def test_nonnegative_int_rejects_negative_or_noninteger(value):
    with pytest.raises(argparse.ArgumentTypeError):
        nonnegative_int(value)


def test_nonnegative_int_accepts_zero_and_positive_integers():
    assert nonnegative_int("0") == 0
    assert nonnegative_int("12") == 12


def test_wiener_config_identifies_its_method():
    assert WienerConfig(K=0.03, adaptive_k=True).method == "wiener"


def test_rl_config_identifies_its_method():
    assert RLConfig(iterations=5).method == "rl"


def test_tv_config_identifies_its_method():
    assert TVConfig(lam=0.002).method == "tv"


def test_method_configs_expose_only_relevant_algorithm_fields():
    assert not hasattr(WienerConfig(), "iterations")
    assert not hasattr(WienerConfig(), "lam")
    assert not hasattr(RLConfig(), "K")
    assert not hasattr(RLConfig(), "lam")
    assert not hasattr(TVConfig(), "K")
    assert not hasattr(TVConfig(), "iterations")


@pytest.mark.parametrize(
    ("config_type", "kwargs"),
    [
        (WienerConfig, {"method": "rl"}),
        (RLConfig, {"method": "tv"}),
        (TVConfig, {"method": "wiener"}),
    ],
)
def test_method_discriminator_cannot_be_overridden(config_type, kwargs):
    with pytest.raises(TypeError):
        config_type(**kwargs)


@pytest.mark.parametrize("K", [0, -0.01, float("nan"), float("inf")])
def test_wiener_config_rejects_invalid_k(K):
    with pytest.raises(ValueError):
        WienerConfig(K=K)


@pytest.mark.parametrize("adaptive_k", ["true", 1, 0, None])
def test_wiener_config_requires_boolean_adaptive_k(adaptive_k):
    with pytest.raises(TypeError):
        WienerConfig(adaptive_k=adaptive_k)


@pytest.mark.parametrize("iterations", [0, -1, 1.5, True])
def test_rl_config_rejects_invalid_iterations(iterations):
    with pytest.raises((TypeError, ValueError)):
        RLConfig(iterations=iterations)


@pytest.mark.parametrize("lam", [0, -0.002, float("nan"), float("inf")])
def test_tv_config_rejects_invalid_lambda(lam):
    with pytest.raises(ValueError):
        TVConfig(lam=lam)


def test_method_configs_are_immutable():
    config = WienerConfig()

    with pytest.raises(FrozenInstanceError):
        config.K = 0.03
