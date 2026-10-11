import pytest

from nir.cli import main


def test_help_is_available_without_data_or_weights(capsys):
    with pytest.raises(SystemExit) as error:
        main(["--help"])
    assert error.value.code == 0
    assert "audit" in capsys.readouterr().out


def test_predict_requires_input_folder(capsys):
    with pytest.raises(SystemExit) as error:
        main(["predict"])
    assert error.value.code == 2
    assert "predict requires --folder" in capsys.readouterr().err


def test_negative_workers_are_rejected():
    with pytest.raises(SystemExit) as error:
        main(["train", "--workers", "-1"])
    assert error.value.code == 2
