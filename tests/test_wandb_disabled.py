"""Test that wandb_mode=disabled doesn't require network or wandb login."""
from hydroda.utils.logger import WandbLogger


def test_wandb_disabled_no_internet():
    """Test wandb disabled mode works without internet."""
    logger = WandbLogger(mode="disabled", project="hydroda-ood", run_name="test_run")
    assert logger.enabled == False
    # Should not make any network requests on init


def test_wandb_disabled_log_is_noop():
    """Test that logging is a no-op when disabled."""
    logger = WandbLogger(mode="disabled")
    # Should not raise
    logger.log({"loss": 1.0})
    logger.log_step({"step": 1, "total_loss": 0.5})
    logger.log_epoch({"epoch": 0, "loss": 0.4})
    logger.log_eval({"rmse": 0.1, "skill": 0.8})
    # finish should also be a no-op
    logger.finish()


def test_wandb_disabled_run_id():
    """Test disabled wandb logger returns None for run_id."""
    logger = WandbLogger(mode="disabled")
    assert logger.run_id is None