import os
from pathlib import Path

import yaml
from dotmap import DotMap


_DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.yaml"


class Config:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Config, cls).__new__(cls)
            cls._instance._load_config()
        return cls._instance

    def _resolve_path(self, path=None):
        if path is not None:
            return Path(path).expanduser().resolve()
        env_path = os.environ.get("SNN_CONFIG_PATH")
        if env_path:
            return Path(env_path).expanduser().resolve()
        return _DEFAULT_CONFIG_PATH

    def _load_config(self, path=None):
        self.config_path = self._resolve_path(path)
        with open(self.config_path, "r") as f:
            config_data = yaml.safe_load(f)
        self._config = DotMap(config_data)

    def reload(self, path=None):
        self._load_config(path)
        return self

    def __getattr__(self, name):
        return getattr(self._config, name)


cfg = Config()
