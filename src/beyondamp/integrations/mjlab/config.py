"""BeyondAMP configuration."""

from dataclasses import dataclass, field
from typing import Any, Literal, Tuple


@dataclass
class BeyondAmpModelCfg:
    """Config for a single neural network model (Actor or Critic)."""

    hidden_dims: Tuple[int, ...] = (128, 128, 128)
    """The hidden dimensions of the network."""
    activation: str = "elu"
    """The activation function."""
    obs_normalization: bool = False
    """Whether to normalize the observations. Default is False."""
    cnn_cfg: dict[str, Any] | None = None
    """CNN encoder config. When set, class_name should be "CNNModel".

  Passed to ``BeyondAMP.modules.CNN``. Common keys: output_channels,
  kernel_size, stride, padding, activation, global_pool, max_pool.
  """
    distribution_cfg: dict[str, Any] | None = None
    """Distribution config dict passed to BeyondAMP. Example::

    {"class_name": "GaussianDistribution",
     "init_std": 1.0, "std_type": "scalar"}

  ``None`` means deterministic output (use for critic).
  """
    rnn_type: str | None = None
    """RNN type ("lstm" or "gru"). When set, class_name should be "RNNModel"."""
    rnn_hidden_dim: int = 256
    """Hidden state dimension for the RNN."""
    rnn_num_layers: int = 1
    """Number of stacked RNN layers."""
    num_experts: int | None = None
    """Number of MoE experts. Used by ``MoEMLPModel`` and ``EstMoEModel``."""
    expert_hidden_dims: Tuple[int, ...] | None = None
    """Hidden dimensions for each MoE expert head."""
    router_hidden_dims: Tuple[int, ...] | None = None
    """Hidden dimensions for the MoE router."""
    router_temperature: float | None = None
    """Softmax temperature for MoE routing."""
    residual_experts: bool | None = None
    """Whether experts predict residual deltas on top of a base policy head."""
    prop_group: str | None = None
    """Observation group containing proprioceptive history for EstMLP/EstMoE models."""
    context_groups: Tuple[str, ...] | None = None
    """Observation groups concatenated with encoded proprioception for EstMLP/EstMoE."""
    prop_latent_dim: int | None = None
    """Latent dimension produced by the EstMLP/EstMoE proprioceptive encoder."""
    prop_encoder_hidden_dims: Tuple[int, ...] | None = None
    """Hidden dimensions for the EstMLP/EstMoE proprioceptive encoder."""
    aux_prop_reconstruction_dim: int | None = None
    """Current proprioceptive target dimension for EstMLP/EstMoE reconstruction loss."""
    aux_base_lin_vel_dim: int | None = None
    """Base linear velocity target dimension for EstMLP/EstMoE prediction loss."""
    task_latent_dim: int | None = None
    """Latent dimension for task semantic encoding in EstMoE."""
    task_encoder_hidden_dims: Tuple[int, ...] | None = None
    """Hidden dimensions for the EstMoE task encoder."""
    cmd_latent_dim: int | None = None
    """Latent dimension for command encoding in EstMoE."""
    cmd_encoder_hidden_dims: Tuple[int, ...] | None = None
    """Hidden dimensions for the EstMoE command encoder."""
    task_dim: int | None = None
    """Number of leading dimensions in actor_context used as task signal."""
    prefer_head_context_group: str | None = None
    """Observation group used to derive weak preferred-expert targets."""
    prefer_head_move_radius: float | None = None
    """Base-command norm used to blend hold-static and hold-move expert priors."""
    prefer_head_stroke_axis: int | None = None
    """Racket target-position axis used to split strike expert priors."""
    temporal_channels: Tuple[int, ...] | None = None
    """Causal H16 encoder channels for SharedSystemAsymmetricMoEActor."""
    temporal_encoder_type: str | None = None
    """Temporal encoder implementation used by SharedSystemAsymmetricMoEActor."""
    temporal_dilations: Tuple[int, ...] | None = None
    """Causal dilation schedule for the residual temporal encoder."""
    system_latent_dim: int | None = None
    """Shared system-state latent width."""
    history_frame_dim: int | None = None
    """Per-frame proprioceptive width when the environment exposes flattened history."""
    branch_hidden_dims: Tuple[int, ...] | None = None
    """Hidden widths after the shared fusion and before each body router."""
    lower_num_experts: int | None = None
    """Number of lower-body experts in the asymmetric shared actor."""
    upper_num_experts: int | None = None
    """Number of upper-body experts in the asymmetric shared actor."""
    privileged_group: str | None = None
    """Teacher-only observation reconstructed by a transfer student."""
    estimator_hidden_dims: Tuple[int, ...] | None = None
    """Hidden widths of the transfer student's privileged-state estimator."""
    skill_public_groups: Tuple[str, ...] | None = None
    """Public groups consumed by the copied skill, excluding estimator-only context."""
    class_name: str = "MLPModel"
    """Model class name resolved by BeyondAMP."""


@dataclass
class BeyondAmpPpoAlgorithmCfg:
    """Config for the PPO algorithm."""

    num_learning_epochs: int = 5
    """The number of learning epochs per update."""
    num_mini_batches: int = 4
    """The number of mini-batches per update.
  mini batch size = num_envs * num_steps / num_mini_batches
  """
    learning_rate: float = 1e-3
    """The learning rate."""
    actor_learning_rate_scale: float = 1.0
    """Actor learning-rate multiplier; useful for conservative policy fine-tuning."""
    schedule: Literal["adaptive", "fixed"] = "adaptive"
    """The learning rate schedule."""
    gamma: float = 0.99
    """The discount factor."""
    lam: float = 0.95
    """The lambda parameter for Generalized Advantage Estimation (GAE)."""
    entropy_coef: float = 0.005
    """The coefficient for the entropy loss."""
    desired_kl: float = 0.01
    """The desired KL divergence between the new and old policies."""
    max_grad_norm: float = 1.0
    """The maximum gradient norm for the policy."""
    value_loss_coef: float = 1.0
    """The coefficient for the value loss."""
    use_clipped_value_loss: bool = True
    """Whether to use clipped value loss."""
    clip_param: float = 0.2
    """The clipping parameter for the policy."""
    normalize_advantage_per_mini_batch: bool = False
    """Whether to normalize the advantage per mini-batch. Default is False. If True, the
  advantage is normalized over the mini-batches only. Otherwise, the advantage is
  normalized over the entire collected trajectories.
  """
    optimizer: Literal["adam", "adamw", "sgd", "rmsprop"] = "adam"
    """The optimizer to use."""
    share_cnn_encoders: bool = False
    """Share CNN encoders between actor and critic."""
    auxiliary_cfg: dict[str, Any] | None = None
    """Optional model-provided auxiliary losses added to PPO actor updates."""
    actor_freeze_updates: int = 0
    """Number of initial PPO updates that train only the critic."""
    cts_cfg: dict[str, Any] | None = None
    """Concurrent teacher/student role weights, schedules, and EMA settings."""
    class_name: str = "PPO"
    """Algorithm class name resolved by BeyondAMP."""


@dataclass
class BeyondAmpMultiObjectivePpoAlgorithmCfg(BeyondAmpPpoAlgorithmCfg):
    """Shared-actor multi-objective PPO configuration."""

    class_name: str = "MultiObjectivePPO"


@dataclass
class BeyondAmpBaseRunnerCfg:
    seed: int = 42
    """The seed for the experiment. Default is 42."""
    num_steps_per_env: int = 24
    """The number of steps per environment update."""
    max_iterations: int = 300
    """The maximum number of iterations."""
    obs_groups: dict[str, tuple[str, ...]] = field(
        default_factory=lambda: {"actor": ("actor",), "critic": ("critic",)},
    )
    save_interval: int = 50
    """The number of iterations between saves."""
    experiment_name: str = "exp1"
    """Directory name used to group runs under ``{log_root}/{experiment_name}/``.
  The log root defaults to ``logs/beyondamp`` and can be overridden with
  ``--log-root`` on the CLI."""
    run_name: str = ""
    """Optional label appended to the timestamped run directory
  (e.g. ``2025-01-27_14-30-00_{run_name}``). Also becomes the
  display name for the run in wandb."""
    logger: Literal["wandb", "tensorboard"] = "wandb"
    """The logger to use. Default is wandb."""
    wandb_project: str = "mjlab"
    """The wandb project name."""
    wandb_tags: Tuple[str, ...] = ()
    """Tags for the wandb run. Default is empty tuple."""
    resume: bool = False
    """Whether to resume the experiment. Default is False."""
    load_run: str = ".*"
    """The run directory to load. Default is ".*" which means all runs. If regex
  expression, the latest (alphabetical order) matching run will be loaded.
  """
    load_checkpoint: str = "model_.*.pt"
    """The checkpoint file to load. Default is "model_.*.pt" (all). If regex expression,
  the latest (alphabetical order) matching file will be loaded.
  """
    clip_actions: float | None = None
    """The clipping range for action values. If None (default), no clipping is applied."""
    upload_model: bool = True
    """Whether to upload model files (.pt, .onnx) to W&B on save. Set to
  False to keep metric logging but avoid storage usage. Default is True."""
    reward_groups: dict[str, Any] | None = None
    """Optional term-to-objective grouping emitted with each rollout step.

  This instrumentation does not change scalar PPO behavior. Multi-objective
  algorithms consume the grouped tensors from the environment extras.
  """
    agent_reward_groups: dict[str, Any] | None = None
    """Optional cooperative MAPPO reward-component configuration.

  The canonical contract starts from the exact scalar environment reward, then
  adds a body-local residual and an optional nested AMP component for each
  action group. This is separate from ``reward_groups``, which remains
  diagnostic or multi-objective instrumentation.
  """
    use_agent_reward_groups: bool = True
    """Whether MAPPO consumes ``agent_reward_groups``.

  Set to ``False`` to run the same grouped actor with the environment's scalar
  reward, which is the scalar-equivalent Shared-Reward MAPPO ablation.
  """
    mappo: dict[str, Any] | None = None
    """Optional grouped-actor MAPPO configuration.

  ``action_groups`` split the flat environment action into non-overlapping
  slices owned by separate actor heads while preserving one scalar cooperative
  PPO objective.
  """


@dataclass
class BeyondAmpOnPolicyRunnerCfg(BeyondAmpBaseRunnerCfg):
    class_name: str = "OnPolicyRunner"
    """The runner class name. Default is OnPolicyRunner."""
    actor: BeyondAmpModelCfg = field(
        default_factory=lambda: BeyondAmpModelCfg(
            distribution_cfg={
                "class_name": "GaussianDistribution",
                "init_std": 1.0,
                "std_type": "scalar",
            }
        )
    )
    """The actor configuration."""
    critic: BeyondAmpModelCfg = field(default_factory=BeyondAmpModelCfg)
    """The critic configuration."""
    algorithm: BeyondAmpPpoAlgorithmCfg = field(default_factory=BeyondAmpPpoAlgorithmCfg)
    """The algorithm configuration."""
    amp: dict[str, Any] | None = None
    """Optional AMP discriminator and expert-dataset configuration."""


@dataclass
class BeyondAmpDistillationAlgorithmCfg:
    """Frozen-teacher online distillation operating point."""

    class_name: str = "GroupedTeacherStudentDistillation"
    num_learning_epochs: int = 2
    gradient_length: int = 1
    learning_rate: float = 3.0e-4
    max_grad_norm: float = 1.0
    loss_type: Literal["mse", "huber"] = "huber"
    optimizer: Literal["adam", "adamw", "sgd", "rmsprop"] = "adam"
    teacher_rollout_start: float = 1.0
    teacher_rollout_end: float = 0.0
    teacher_rollout_decay_updates: int = 750
    teacher_rollout_hold_updates: int = 0
    """Updates that execute only the Teacher before DAgger decay begins."""
    action_group_weights: dict[str, float] = field(
        default_factory=lambda: {"lower": 1.0, "upper": 2.0}
    )
    privileged_reconstruction_coef: float = 0.0
    """Direct supervision for the Student estimate of Teacher-only input."""
    skill_feature_coef: float = 0.0
    """Feature alignment coefficient inside the transferred skill network."""
    skill_freeze_updates: int = 0
    """Updates that train only the Student estimator before skill fine-tuning."""
    skill_learning_rate_scale: float = 1.0
    """Learning-rate multiplier for copied Teacher skill parameters."""


@dataclass
class BeyondAmpDistillationRunnerCfg(BeyondAmpBaseRunnerCfg):
    """Configuration for grouped Teacher→Student online distillation."""

    class_name: str = "DistillationRunner"
    student: BeyondAmpModelCfg = field(
        default_factory=lambda: BeyondAmpModelCfg(
            distribution_cfg={
                "class_name": "GaussianDistribution",
                "init_std": 0.35,
                "std_type": "log",
                "std_range": (0.05, 1.0),
            }
        )
    )
    teacher: BeyondAmpModelCfg = field(
        default_factory=lambda: BeyondAmpModelCfg(
            distribution_cfg={
                "class_name": "GaussianDistribution",
                "init_std": 0.70,
                "std_type": "log",
                "std_range": (0.05, 2.0),
            }
        )
    )
    algorithm: BeyondAmpDistillationAlgorithmCfg = field(
        default_factory=BeyondAmpDistillationAlgorithmCfg
    )
    action_groups: tuple[dict[str, Any], ...] = ()
