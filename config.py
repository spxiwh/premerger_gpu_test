from pathlib import Path
from pydantic import BaseModel

class AnalysisConfig(BaseModel):
    cutoff_days: float
    parameters: list[str] = []
    batch_size: int = 200

class SamplerConfig(BaseModel):
    n_particles: int = 500
    target_ess_ratio: float = 0.8
    mutation_n_steps: int = 100
    n_final_particles: int | None = None


class NessaiConfig(BaseModel):
    nlive: int = 1000
    reset_flow: int = 0

class InferenceConfig(BaseModel):
    outdir: Path
    label_suffix: str = ""
    sampler: SamplerConfig
    analysis: AnalysisConfig
    skip_sampling: bool = False

class NessaiInferenceConfig(BaseModel):
    outdir: Path
    label_suffix: str = ""
    sampler: NessaiConfig
    analysis: AnalysisConfig
    skip_sampling: bool = False


import yaml

def load_config(path: str, cls: type[InferenceConfig] = InferenceConfig) -> InferenceConfig:
    with open(path, "r") as f:
        raw = yaml.safe_load(f)
    return cls.model_validate(raw)