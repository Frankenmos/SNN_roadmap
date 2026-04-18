import sys
import unittest.mock as mock
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

class SpecMock(mock.MagicMock):
    @property
    def __spec__(self):
        return mock.MagicMock()

def mock_dependencies():
    # Add to sys.modules
    sys.modules['numpy'] = SpecMock()
    sys.modules['torch'] = SpecMock()
    sys.modules['torch.nn'] = SpecMock()
    sys.modules['torch.nn.functional'] = SpecMock()
    sys.modules['torch.optim'] = SpecMock()
    sys.modules['torch.amp'] = SpecMock()
    sys.modules['torch.distributions'] = SpecMock()
    sys.modules['torchvision'] = SpecMock()
    sys.modules['pysc2'] = SpecMock()
    sys.modules['pysc2.lib'] = SpecMock()
    sys.modules['pysc2.agents'] = SpecMock()
    sys.modules['pysc2.agents.base_agent'] = SpecMock()
    sys.modules['pysc2.env'] = SpecMock()
    sys.modules['ray'] = SpecMock()
    sys.modules['dotmap'] = SpecMock()
    sys.modules['absl'] = SpecMock()
    sys.modules['snntorch'] = SpecMock()
    sys.modules['snntorch.surrogate'] = SpecMock()
    sys.modules['yaml'] = SpecMock()
    sys.modules['yaml.safe_load'] = SpecMock()

mock_dependencies()
