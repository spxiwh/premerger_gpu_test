from pathlib import Path
from pydantic import BaseModel

class AnalysisConfig(BaseModel):
    cutoff_days: float
    parameters: list[str] = []

class SamplerConfig(BaseModel):
    n_particles: int = 500
    target_ess_ratio: float = 0.8
    mutation_n_steps: int = 100
    n_final_particles: int | None = None

class InferenceConfig(BaseModel):
    outdir: Path
    label_suffix: str = ""
    sampler: SamplerConfig
    analysis: AnalysisConfig


import yaml

def load_config(path: str) -> InferenceConfig:
    with open(path, "r") as f:
        raw = yaml.safe_load(f)
    return InferenceConfig.model_validate(raw)