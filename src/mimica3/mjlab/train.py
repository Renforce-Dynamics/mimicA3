"""Task-owned DR selection followed by the unchanged BeyondAMP PPO launcher."""

import argparse
import sys

import tyro

import mjlab
from beyondamp.integrations.mjlab.train import TrainConfig, launch_training
from mimica3.mjlab import TASK_ID
from mimica3.randomization import DR_PROFILES
from mjlab.tasks.registry import list_tasks


def parse_train_args(argv: list[str]):
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--dr-profile", choices=DR_PROFILES)
    selection, remaining = parser.parse_known_args(argv)
    task, remaining = tyro.cli(
        tyro.extras.literal_type_from_choices(list_tasks()),
        args=remaining,
        add_help=False,
        return_unknown_args=True,
        config=mjlab.TYRO_FLAGS,
    )
    if selection.dr_profile is not None:
        if task != TASK_ID:
            parser.error("--dr-profile is used with the base task ID; variants already select DR")
        task = f"{TASK_ID}-{selection.dr_profile}"
    cfg = tyro.cli(
        TrainConfig,
        args=remaining,
        default=TrainConfig.from_task(task),
        config=mjlab.TYRO_FLAGS,
        prog=f"train-mimica3 {task}",
    )
    return task, cfg


def main():
    argv = sys.argv[1:]
    if not argv or argv == ["--help"] or argv == ["-h"]:
        print(f"Usage: train-mimica3 {TASK_ID} [--dr-profile PROFILE] [TrainConfig options]")
        print("DR profiles: " + ", ".join(DR_PROFILES))
        print("Default: basic_dr. Use TASK --help for PPO/environment options.")
        return
    task, cfg = parse_train_args(argv)
    launch_training(task_id=task, args=cfg)


if __name__ == "__main__":
    main()
