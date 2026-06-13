from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "utils" / "gpu_memory_hold.sh"


def test_gpu_memory_hold_script_has_expected_cli():
    result = subprocess.run(
        ["bash", str(SCRIPT), "--help"],
        check=True,
        capture_output=True,
        text=True,
    )

    help_text = result.stdout
    assert "Usage:" in help_text
    assert "--gpu ID" in help_text
    assert "--gb GB" in help_text
    assert "Default: --gb 12" in help_text
    assert "--force" in help_text
    assert "Ctrl-C" in help_text


def test_gpu_memory_hold_script_is_valid_bash():
    subprocess.run(["bash", "-n", str(SCRIPT)], check=True)
