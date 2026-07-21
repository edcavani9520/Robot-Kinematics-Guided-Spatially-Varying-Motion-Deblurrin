import inspect

import pytest

import process_one_frame


def test_reverse_psf_is_not_part_of_python_api():
    assert "reverse_psf" not in inspect.signature(process_one_frame.process_frame).parameters


def test_reverse_psf_cli_option_is_rejected():
    with pytest.raises(SystemExit) as exc_info:
        process_one_frame._parse_args(["--reverse-psf", "wiener"])

    assert exc_info.value.code == 2
