"""Named, immutable robot DR building blocks; no simulator imports.

Full robot priors follow iKobe Mimic; Sonic is the six-channel overlay from
iKobe velocity/finetune. These are simulation priors, not hardware calibration.
"""

from dataclasses import dataclass
from typing import Literal, get_args

DrProfile = Literal["no_dr", "basic_dr", "full_dr", "sonic_dr", "full_plus_sonic_dr"]
DR_PROFILES: tuple[str, ...] = get_args(DrProfile)
DR_REVISION = "mimica3.dr.v1"


@dataclass(frozen=True)
class RandomizationConfig:
    profile: DrProfile
    revision: str = DR_REVISION
    observation_noise: bool = False
    reset_noise: bool = False
    density_scale: tuple[float, float] | None = None
    com_half_width: tuple[float, float, float] | None = None
    foot_friction: tuple[float, float] | None = None
    kp_scale: tuple[float, float] | None = None
    kd_scale: tuple[float, float] | None = None
    encoder_bias: float = 0.0
    delay_steps: tuple[int, int] = (0, 0)
    push_interval_s: tuple[float, float] | None = None
    payload_mass_add: tuple[float, float] | None = None
    armature_scale: tuple[float, float] | None = None
    gravity_z_offset: tuple[float, float] | None = None


def resolve_dr_profile(profile: DrProfile) -> RandomizationConfig:
    if profile not in DR_PROFILES:
        raise ValueError(f"Unknown DR profile {profile!r}; choose one of {DR_PROFILES}")
    values = dict(observation_noise=profile != "no_dr", reset_noise=profile != "no_dr")
    if profile in ("full_dr", "full_plus_sonic_dr"):
        values.update(
            density_scale=(0.8, 1.2),
            com_half_width=(0.025, 0.05, 0.05),
            foot_friction=(0.2, 1.5),
            kp_scale=(0.8, 1.2),
            kd_scale=(0.7, 1.3),
            encoder_bias=0.01,
            delay_steps=(0, 1),
            push_interval_s=(1.0, 3.0),
        )
    if profile in ("sonic_dr", "full_plus_sonic_dr"):
        # Replace overlapping ranges, never stack two independent samples.
        values.update(
            kp_scale=(0.8, 1.3),
            kd_scale=(0.8, 1.3),
            com_half_width=(0.075, 0.075, 0.125),
            payload_mass_add=(-0.2, 2.0),
            armature_scale=(0.3, 1.3),
            gravity_z_offset=(-0.1, 0.1),
        )
    return RandomizationConfig(profile=profile, **values)
