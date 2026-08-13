import os
from pathlib import Path

import torch

from beyondamp.env import VecEnv
from beyondamp.integrations.mjlab.vecenv_wrapper import BeyondAmpVecEnvWrapper
from beyondamp.runners import OnPolicyRunner


def _promote_distilled_student_to_actor(
    loaded_dict: dict, load_cfg: dict | None
) -> bool:
    """Expose a distilled Student under the PPO actor checkpoint key.

    Returns whether a migration was performed.  Strict model loading remains
    responsible for verifying that the destination actor has the identical
    architecture and observation ABI.
    """

    if (
        load_cfg is None
        or not load_cfg.get("actor", False)
        or "actor_state_dict" in loaded_dict
        or "student_state_dict" not in loaded_dict
    ):
        return False
    loaded_dict["actor_state_dict"] = loaded_dict["student_state_dict"]
    return True


class MjlabOnPolicyRunner(OnPolicyRunner):
    """Base runner that persists environment state across checkpoints."""

    env: BeyondAmpVecEnvWrapper

    def __init__(
        self,
        env: VecEnv,
        train_cfg: dict,
        log_dir: str | None = None,
        device: str = "cpu",
    ) -> None:
        # Strip None-valued optional configs so MLPModel doesn't receive them.
        for key in ("actor", "critic"):
            if key in train_cfg:
                for opt in ("cnn_cfg", "distribution_cfg"):
                    if train_cfg[key].get(opt) is None:
                        train_cfg[key].pop(opt, None)
                if train_cfg[key].get("rnn_type") is None:
                    for opt in ("rnn_type", "rnn_hidden_dim", "rnn_num_layers"):
                        train_cfg[key].pop(opt, None)
        # Dataclass serialization retains optional algorithm extensions as
        # explicit None values.  Plain PPO does not accept extension-specific
        # kwargs such as cts_cfg, while MAPPO variants still receive them when
        # configured with an actual mapping.
        if "algorithm" in train_cfg:
            train_cfg["algorithm"] = {
                key: value
                for key, value in train_cfg["algorithm"].items()
                if value is not None
            }
        super().__init__(env, train_cfg, log_dir, device)

    def export_policy_to_onnx(
        self, path: str, filename: str = "policy.onnx", verbose: bool = False
    ) -> None:
        """Export policy to ONNX format using legacy export path.

        Overrides the base implementation to set dynamo=False, avoiding warnings about
        dynamic_axes being deprecated with the new TorchDynamo export path
        (torch>=2.9 default).
        """
        onnx_model = self.alg.get_policy().as_onnx(verbose=verbose)
        onnx_model.to("cpu")
        onnx_model.eval()
        os.makedirs(path, exist_ok=True)
        torch.onnx.export(
            onnx_model,
            onnx_model.get_dummy_inputs(),  # type: ignore[operator]
            os.path.join(path, filename),
            export_params=True,
            opset_version=18,
            verbose=verbose,
            input_names=onnx_model.input_names,  # type: ignore[arg-type]
            output_names=onnx_model.output_names,  # type: ignore[arg-type]
            dynamic_axes={},
            dynamo=False,
        )

    @staticmethod
    def _get_export_paths(checkpoint_path: str) -> tuple[Path, str, Path]:
        """Resolve ONNX export paths from a checkpoint path."""
        export_dir = Path(checkpoint_path).parent
        filename = f"{export_dir.name}.onnx"
        return export_dir, filename, export_dir / filename

    def load(
        self,
        path: str,
        load_cfg: dict | None = None,
        strict: bool = True,
        map_location: str | None = None,
    ) -> dict:
        """Load PPO state, including an exact distilled-Student actor warm-start."""

        loaded_dict = torch.load(path, map_location=map_location, weights_only=False)
        if not _promote_distilled_student_to_actor(loaded_dict, load_cfg):
            return super().load(path, load_cfg, strict, map_location)
        print(f"Detected Student distillation checkpoint at {path}; loading Student as actor.")
        load_iteration = self.alg.load(loaded_dict, load_cfg, strict)
        if load_iteration:
            self.current_learning_iteration = loaded_dict["iter"]
        infos = loaded_dict.get("infos") or {}
        if load_iteration and "env_state" in infos:
            self.env.unwrapped.common_step_counter = infos["env_state"]["common_step_counter"]
        return infos


class MjlabDistillationRunner(MjlabOnPolicyRunner):
    """MJLab checkpoint semantics plus frozen-teacher validation."""

    def __init__(self, env, train_cfg, log_dir=None, device="cpu") -> None:
        for key in ("student", "teacher"):
            if key in train_cfg:
                for opt in ("cnn_cfg", "distribution_cfg"):
                    if train_cfg[key].get(opt) is None:
                        train_cfg[key].pop(opt, None)
                if train_cfg[key].get("rnn_type") is None:
                    for opt in ("rnn_type", "rnn_hidden_dim", "rnn_num_layers"):
                        train_cfg[key].pop(opt, None)
        # Avoid the actor/critic-only cleanup in MjlabOnPolicyRunner.
        OnPolicyRunner.__init__(self, env, train_cfg, log_dir, device)

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False) -> None:
        if not getattr(self.alg, "teacher_loaded", False):
            raise ValueError("Strong Teacher checkpoint must be loaded before distillation")
        super().learn(num_learning_iterations, init_at_random_ep_len)

    def save(self, path: str, infos=None) -> None:
        """Save checkpoint.

        Extends the base implementation to persist the environment's
        common_step_counter and to respect the ``upload_model`` config flag.
        """
        env_state = {"common_step_counter": self.env.unwrapped.common_step_counter}
        infos = {**(infos or {}), "env_state": env_state}
        # Inline base OnPolicyRunner.save() to conditionally gate W&B upload.
        saved_dict = self.alg.save()
        saved_dict["iter"] = self.current_learning_iteration
        saved_dict["infos"] = infos
        torch.save(saved_dict, path)
        if self.cfg["upload_model"]:
            self.logger.save_model(path, self.current_learning_iteration)

    def load(
        self,
        path: str,
        load_cfg: dict | None = None,
        strict: bool = True,
        map_location: str | None = None,
    ) -> dict:
        """Load checkpoint.

        Extends the base implementation to:
        1. Restore common_step_counter to preserve curricula state.
        2. Migrate legacy checkpoints (actor.* -> mlp.*, actor_obs_normalizer.*
          -> obs_normalizer.*) to the current format (BeyondAMP>=4.0).
        """
        loaded_dict = torch.load(path, map_location=map_location, weights_only=False)

        # CTS-Transfer checkpoints persist the deployable grouped policy as
        # ``student_state_dict``.  A PPO actor warm-start is an exact state-dict
        # migration when the selected profile has the same grouped transfer
        # architecture; strict loading below remains the ABI guard.
        if _promote_distilled_student_to_actor(loaded_dict, load_cfg):
            print(f"Detected Student distillation checkpoint at {path}; loading Student as actor.")

        if "model_state_dict" in loaded_dict:
            print(f"Detected legacy checkpoint at {path}. Migrating to new format...")
            model_state_dict = loaded_dict.pop("model_state_dict")
            actor_state_dict = {}
            critic_state_dict = {}

            for key, value in model_state_dict.items():
                # Migrate actor keys.
                if key.startswith("actor."):
                    new_key = key.replace("actor.", "mlp.")
                    actor_state_dict[new_key] = value
                elif key.startswith("actor_obs_normalizer."):
                    new_key = key.replace("actor_obs_normalizer.", "obs_normalizer.")
                    actor_state_dict[new_key] = value
                elif key in ["std", "log_std"]:
                    actor_state_dict[key] = value

                # Migrate critic keys.
                if key.startswith("critic."):
                    new_key = key.replace("critic.", "mlp.")
                    critic_state_dict[new_key] = value
                elif key.startswith("critic_obs_normalizer."):
                    new_key = key.replace("critic_obs_normalizer.", "obs_normalizer.")
                    critic_state_dict[new_key] = value

            loaded_dict["actor_state_dict"] = actor_state_dict
            loaded_dict["critic_state_dict"] = critic_state_dict

        # Migrate BeyondAMP 4.x actor keys to 5.x distribution keys.
        actor_sd = loaded_dict.get("actor_state_dict", {})
        if "std" in actor_sd:
            actor_sd["distribution.std_param"] = actor_sd.pop("std")
        if "log_std" in actor_sd:
            actor_sd["distribution.log_std_param"] = actor_sd.pop("log_std")

        load_iteration = self.alg.load(loaded_dict, load_cfg, strict)
        if load_iteration:
            self.current_learning_iteration = loaded_dict["iter"]

        infos = loaded_dict["infos"]
        load_env_state = load_cfg is None or bool(load_cfg.get("iteration", False))
        if load_env_state and infos and "env_state" in infos:
            self.env.unwrapped.common_step_counter = infos["env_state"]["common_step_counter"]
        return infos
