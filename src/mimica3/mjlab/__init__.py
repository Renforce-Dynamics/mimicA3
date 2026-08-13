"""MJLab task registration boundary."""

TASK_ID = "MimicA3-MultiMotion-Tracking-v1"


def _register() -> None:
    from mimica3.mjlab.task import multi_motion_env_cfg, multi_motion_runner_cfg
    from mjlab.tasks.registry import register_mjlab_task

    try:
        register_mjlab_task(
            task_id=TASK_ID,
            env_cfg=multi_motion_env_cfg(shard_across_ranks=True),
            play_env_cfg=multi_motion_env_cfg(play=True, shard_across_ranks=False),
            rl_cfg=multi_motion_runner_cfg(),
        )
    except ValueError as exc:
        if "already registered" not in str(exc):
            raise


_register()

__all__ = ["TASK_ID"]
