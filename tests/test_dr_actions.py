"""CPU contracts for delayed encoder commands, including asynchronous RSI resets."""

from types import SimpleNamespace

import torch

from mimica3.mjlab.actions import A3JointPositionAction, A3JointPositionActionCfg


def make_action(delay=1):
    data = SimpleNamespace(
        default_joint_pos=torch.zeros(2, 1),
        joint_pos=torch.tensor([[0.4], [-0.3]]),
        joint_pos_limits=torch.tensor([[[-1.0, 1.0]], [[-1.0, 1.0]]]),
        encoder_bias=torch.tensor([[0.01], [-0.01]]),
    )
    robot = SimpleNamespace(data=data, find_joints=lambda *a, **k: ([0], ["joint"]))
    robot.set_joint_position_target = lambda value, **k: setattr(robot, "target", value.clone())
    env = SimpleNamespace(scene={"robot": robot}, device="cpu", num_envs=2)
    action = A3JointPositionAction(
        A3JointPositionActionCfg(
            entity_name="robot",
            actuator_names=("joint",),
            scale=0.5,
            delay_steps=(delay, delay),
        ),
        env,
    )
    return action, robot


def test_delay_holds_fresh_rsi_pose_and_history_is_encoder_command():
    action, robot = make_action()
    action.reset()
    # The command manager resets AFTER the action manager.
    robot.data.joint_pos[:] = torch.tensor([[0.7], [-0.6]])
    action.process_actions(torch.tensor([[8.0], [-8.0]]))
    action.apply_actions()
    torch.testing.assert_close(robot.target, robot.data.joint_pos)
    torch.testing.assert_close(
        action.executed_action, (robot.target + robot.data.encoder_bias) / 0.5
    )
    action.process_actions(torch.zeros(2, 1))
    action.apply_actions()
    torch.testing.assert_close(robot.target, torch.tensor([[1.0], [-1.0]]))


def test_partial_reset_clears_only_reset_world_delay_queue():
    action, robot = make_action()
    action.process_actions(torch.tensor([[1.0], [-1.0]]))
    action.reset(torch.tensor([0]))
    robot.data.joint_pos[0] = 0.8
    action.process_actions(torch.zeros(2, 1))
    action.apply_actions()
    torch.testing.assert_close(robot.target, torch.tensor([[0.8], [-0.49]]))


def test_no_delay_clips_physical_target_after_bias_correction():
    action, robot = make_action(delay=0)
    action.process_actions(torch.tensor([[-8.0], [8.0]]))
    action.apply_actions()
    torch.testing.assert_close(robot.target, torch.tensor([[-1.0], [1.0]]))
    torch.testing.assert_close(
        action.executed_action, (robot.target + robot.data.encoder_bias) / 0.5
    )
