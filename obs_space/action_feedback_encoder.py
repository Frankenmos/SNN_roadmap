import numpy as np

from agent_core.policy_protocol import (
    ACTION_FEEDBACK_ANY_EXECUTED_OFFSET,
    ACTION_FEEDBACK_BRIDGE_TYPE_OFFSET,
    ACTION_FEEDBACK_EXECUTED_SMART_OFFSET,
    ACTION_FEEDBACK_KILL_DELTA_OFFSET,
    ACTION_FEEDBACK_PENALTY_BIT_OFFSET,
    ACTION_FEEDBACK_RESERVED_OFFSET,
    ACTION_FEEDBACK_SCORE_DELTA_OFFSET,
    ACTION_FEEDBACK_TOKEN_DIM,
    ACTION_FEEDBACK_X_NORM_OFFSET,
    ACTION_FEEDBACK_Y_NORM_OFFSET,
    AGENT_ACTION_TOKEN_DIM,
    BRIDGE_ACTION_NO_OP,
    SMART_SCREEN_FUNCTION_ID,
    SPATIAL_OBS_SHAPE,
)


class ActionFeedbackEncoder:
    def __init__(self, screen_size=None):
        self.screen_size = int(screen_size or SPATIAL_OBS_SHAPE[-1])

    def encode_feedback(
        self,
        *,
        last_action_token=None,
        last_action_ids=None,
        score_delta=None,
    ):
        action_token = self._normalize_last_action_token(last_action_token)
        action_ids = self._as_int_set(last_action_ids)
        delta = self._normalize_score_delta(score_delta)

        feedback = np.zeros(ACTION_FEEDBACK_TOKEN_DIM, dtype=np.float32)
        feedback[ACTION_FEEDBACK_BRIDGE_TYPE_OFFSET] = action_token[0]
        feedback[ACTION_FEEDBACK_X_NORM_OFFSET] = action_token[1]
        feedback[ACTION_FEEDBACK_Y_NORM_OFFSET] = action_token[2]
        feedback[ACTION_FEEDBACK_EXECUTED_SMART_OFFSET] = (
            1.0 if SMART_SCREEN_FUNCTION_ID in action_ids else 0.0
        )
        feedback[ACTION_FEEDBACK_ANY_EXECUTED_OFFSET] = 1.0 if action_ids else 0.0
        feedback[ACTION_FEEDBACK_SCORE_DELTA_OFFSET] = float(
            np.clip(delta[0], -10.0, 10.0) / 10.0,
        )
        feedback[ACTION_FEEDBACK_KILL_DELTA_OFFSET] = float(
            np.clip(delta[5], 0.0, 100.0) / 100.0,
        )
        feedback[ACTION_FEEDBACK_PENALTY_BIT_OFFSET] = (
            1.0 if float(delta[0]) < 0.0 else 0.0
        )
        feedback[ACTION_FEEDBACK_RESERVED_OFFSET] = 0.0
        return feedback

    def _normalize_last_action_token(self, token):
        if token is None:
            token = np.asarray(
                [BRIDGE_ACTION_NO_OP, 0, 0, 0],
                dtype=np.float32,
            )
        else:
            token = np.asarray(token, dtype=np.float32).reshape(-1)
        if token.size < AGENT_ACTION_TOKEN_DIM:
            token = np.pad(
                token,
                (0, AGENT_ACTION_TOKEN_DIM - token.size),
                mode="constant",
            )
        else:
            token = token[:AGENT_ACTION_TOKEN_DIM]

        max_coord = float(max(1, self.screen_size - 1))
        out = token.astype(np.float32, copy=True)
        out[0] = float(max(0.0, out[0]))
        out[1] = float(np.clip(out[1], 0.0, max_coord) / max_coord)
        out[2] = float(np.clip(out[2], 0.0, max_coord) / max_coord)
        out[3] = float(out[3])
        return out

    @staticmethod
    def _as_int_set(value):
        if value is None:
            return set()
        try:
            arr = np.asarray(value).reshape(-1)
            return {int(item) for item in arr.tolist()}
        except Exception:
            try:
                return {int(item) for item in list(value)}
            except Exception:
                return set()

    @staticmethod
    def _normalize_score_delta(score_delta):
        if score_delta is None:
            return np.zeros(13, dtype=np.float32)
        delta = np.asarray(score_delta, dtype=np.float32).reshape(-1)
        if delta.size < 13:
            delta = np.pad(delta, (0, 13 - delta.size), mode="constant")
        else:
            delta = delta[:13]
        return delta
