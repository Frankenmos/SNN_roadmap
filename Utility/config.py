import yaml
from dotmap import DotMap

class Config:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Config, cls).__new__(cls)
            cls._instance._load_config()
        return cls._instance

    def _load_config(self):
        with open("config.yaml", "r") as f:
            config_data = yaml.safe_load(f)
        self._config = DotMap(config_data)

    def __getattr__(self, name):
        return getattr(self._config, name)

cfg = Config()
