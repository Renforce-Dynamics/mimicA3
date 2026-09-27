"""MJLab task registration boundary."""

TASK_ID = "MimicA3-MultiMotion-Tracking-v1"


def _register() -> None:
    from beyondamp.integrations.mjlab.runner import MjlabOnPolicyRunner
    from mimica3.mjlab.task import multi_motion_env_cfg, multi_motion_runner_cfg
    from mimica3.randomization import DR_PROFILES
    from mjlab.tasks.registry import register_mjlab_task

    for profile in (None, *DR_PROFILES):
        task_id = TASK_ID if profile is None else f"{TASK_ID}-{profile}"
        runner = multi_motion_runner_cfg()
        runner.run_name = f"mlp_{profile or 'basic_dr'}"
        try:
            register_mjlab_task(
                task_id=task_id,
                env_cfg=multi_motion_env_cfg(shard_across_ranks=True, dr_profile=profile),
                play_env_cfg=multi_motion_env_cfg(play=True, dr_profile=profile),
                rl_cfg=runner,
                runner_cls=MjlabOnPolicyRunner,
            )
        except ValueError as exc:
            if "already registered" not in str(exc):
                raise


_register()

__all__ = ["TASK_ID"]
