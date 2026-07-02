import unittest
import sys
from types import SimpleNamespace

import numpy as np

if "torch" not in sys.modules:
    sys.modules["torch"] = SimpleNamespace(
        Tensor=object,
        float32="float32",
        zeros=lambda *args, **kwargs: None,
    )

from agent_core.policy_protocol import SMART_SCREEN_FUNCTION_ID


class SmartOutcomeDiagnosticsWrapperParsingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            from Utility.smart_outcome_diagnostics_wrapper import (  # noqa: PLC0415
                SmartOutcomeDiagnosticsWrapper,
                _normalize_actions,
            )
        except Exception as exc:  # PySC2 may be unavailable in lightweight test envs.
            raise unittest.SkipTest(f"Smart outcome wrapper unavailable: {exc}") from exc
        cls.Wrapper = SmartOutcomeDiagnosticsWrapper
        cls.normalize_actions = staticmethod(_normalize_actions)

    def wrapper(self):
        wrapper = self.Wrapper.__new__(self.Wrapper)
        wrapper._action_spec = None
        return wrapper

    def test_extracts_target_from_realish_pysc2_function_call(self):
        action = SimpleNamespace(
            function=SMART_SCREEN_FUNCTION_ID,
            arguments=[[0], [np.int32(40), np.int32(50)]],
            name="Smart_screen",
        )

        self.assertEqual(self.wrapper()._extract_smart_target(action), (40.0, 50.0))

    def test_rejects_non_smart_action(self):
        action = SimpleNamespace(function=0, arguments=[])

        self.assertIsNone(self.wrapper()._extract_smart_target(action))

    def test_mapping_action_shape_is_supported_for_unit_tests(self):
        action = {
            "function_id": SMART_SCREEN_FUNCTION_ID,
            "arguments": [[0], [12, 34]],
        }

        self.assertEqual(self.wrapper()._extract_smart_target(action), (12.0, 34.0))

    def test_normalize_actions_handles_missing_args(self):
        self.assertEqual(self.normalize_actions((), {}), [])


if __name__ == "__main__":
    unittest.main()
