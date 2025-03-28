from omegaconf import DictConfig, OmegaConf
from pathlib import Path
from typing import Union


def parse_config(
    *config_paths: Union[str, Path],
) -> DictConfig:
    """Parse a list of config files and merge them into a single config."""
    yml_config = OmegaConf.merge(*[OmegaConf.load(path) for path in config_paths])
    cli_config = OmegaConf.from_cli()
    return OmegaConf.merge(yml_config, cli_config)  # type: ignore
