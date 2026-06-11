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

from obs_space.action_effects import FrameSnapshot, UnitSnapshot
from obs_space.smart_outcome_detector import (
    CooldownSnapshot,
    PendingSmartClick,
    SmartOutcome,
    SmartOutcomeDetector,
    check_cooldowns_fired_production,
    extract_cooldown_snapshots_from_feature_units,
    nearest_enemy_distance,
    _safe_float,
)


FRIENDLY = 1
ENEMY = 4


def raw_unit(*, alliance, health, cooldown, x, y, tag):
    row = np.zeros(30, dtype=np.float32)
    row[1] = alliance
    row[2] = health
    row[8] = cooldown
    row[12] = x
    row[13] = y
    row[29] = tag
    return row


def raw_units(*, enemy_health=80.0, friendly_cooldowns=(0.0, 0.0)):
    return np.asarray(
        [
            raw_unit(
                alliance=FRIENDLY,
                health=50.0,
                cooldown=friendly_cooldowns[0],
                x=20.0,
                y=30.0,
                tag=101,
            ),
            raw_unit(
                alliance=FRIENDLY,
                health=45.0,
                cooldown=friendly_cooldowns[1],
                x=22.0,
                y=32.0,
                tag=102,
            ),
            raw_unit(
                alliance=ENEMY,
                health=enemy_health,
                cooldown=0.0,
                x=40.0,
                y=50.0,
                tag=201,
            ),
        ],
        dtype=np.float32,
    )


def frame(*, enemy_health=80.0, friendlies=None, enemies=None):
    if friendlies is None:
        friendlies = (
            UnitSnapshot(FRIENDLY, 50.0, 20.0, 30.0, 101),
            UnitSnapshot(FRIENDLY, 45.0, 22.0, 32.0, 102),
        )
    if enemies is None:
        enemies = (UnitSnapshot(ENEMY, enemy_health, 40.0, 50.0, 201),)
    return FrameSnapshot(friendlies=tuple(friendlies), enemies=tuple(enemies))


class SmartOutcomeDetectorTests(unittest.TestCase):
    def test_reset_clears_pending_clicks(self):
        detector = SmartOutcomeDetector()
        detector._pending_clicks.append(
            PendingSmartClick(
                click_step=0,
                target_x=10.0,
                target_y=20.0,
                enemy_health_snapshot=100.0,
                enemy_snapshots=(),
                friendly_cooldowns=(),
                friendly_positions=(),
            ),
        )

        detector.reset()

        self.assertEqual(detector.pending_count, 0)

    def test_attack_likely_uses_previous_frame_baseline(self):
        detector = SmartOutcomeDetector(outcome_window=5)
        outcomes = detector.process_transition(
            previous_frame=frame(enemy_health=100.0),
            current_frame=frame(enemy_health=70.0),
            click_step=0,
            resolution_step=1,
            smart_target=(40.0, 50.0),
            previous_feature_units=raw_units(enemy_health=100.0),
            current_feature_units=raw_units(enemy_health=70.0),
        )

        self.assertEqual(len(outcomes), 1)
        self.assertEqual(outcomes[0].outcome_class, "attack_likely")
        self.assertEqual(outcomes[0].resolution_reason, "enemy_health_drop")
        self.assertEqual(outcomes[0].enemy_health_delta, 30.0)
        self.assertEqual(outcomes[0].window_steps, 1)

    def test_attack_intent_when_click_near_enemy_without_damage(self):
        detector = SmartOutcomeDetector(outcome_window=5)
        detector.observe_smart_click(
            previous_frame=frame(enemy_health=80.0),
            target=(40.0, 50.0),
            click_step=0,
            previous_feature_units=raw_units(enemy_health=80.0),
        )

        outcomes = detector.resolve(
            current_frame=frame(enemy_health=80.0),
            resolution_step=5,
            current_feature_units=raw_units(enemy_health=80.0),
        )

        self.assertEqual(len(outcomes), 1)
        self.assertEqual(outcomes[0].outcome_class, "attack_intent")
        self.assertTrue(outcomes[0].target_was_near_enemy)

    def test_fired_likely_uses_real_cooldown_change(self):
        detector = SmartOutcomeDetector(outcome_window=5)
        outcomes = detector.process_transition(
            previous_frame=frame(enemy_health=80.0),
            current_frame=frame(enemy_health=80.0),
            click_step=0,
            resolution_step=1,
            smart_target=(40.0, 50.0),
            previous_feature_units=raw_units(enemy_health=80.0, friendly_cooldowns=(0.0, 1.0)),
            current_feature_units=raw_units(enemy_health=80.0, friendly_cooldowns=(15.0, 1.0)),
        )

        self.assertEqual(len(outcomes), 1)
        self.assertEqual(outcomes[0].outcome_class, "fired_likely")
        self.assertTrue(outcomes[0].any_cooldown_fired)
        self.assertEqual(outcomes[0].resolution_reason, "cooldown_fired")

    def test_move_like_when_units_move_toward_non_enemy_target(self):
        detector = SmartOutcomeDetector(outcome_window=2)
        previous = frame(
            friendlies=(UnitSnapshot(FRIENDLY, 50.0, 10.0, 10.0, 101),),
            enemies=(),
        )
        current = frame(
            friendlies=(UnitSnapshot(FRIENDLY, 50.0, 15.0, 10.0, 101),),
            enemies=(),
        )
        detector.observe_smart_click(
            previous_frame=previous,
            target=(20.0, 10.0),
            click_step=0,
        )

        outcomes = detector.resolve(current_frame=current, resolution_step=2)

        self.assertEqual(len(outcomes), 1)
        self.assertEqual(outcomes[0].outcome_class, "move_like")
        self.assertTrue(outcomes[0].friendly_moved_toward)

    def test_repeated_smart_spam_credits_one_click_for_one_damage_event(self):
        detector = SmartOutcomeDetector(outcome_window=5)
        previous = frame(enemy_health=100.0)
        detector.observe_smart_click(
            previous_frame=previous,
            target=(40.0, 50.0),
            click_step=0,
            previous_feature_units=raw_units(enemy_health=100.0),
        )
        detector.observe_smart_click(
            previous_frame=previous,
            target=(40.0, 50.0),
            click_step=1,
            previous_feature_units=raw_units(enemy_health=100.0),
        )

        outcomes = detector.resolve(
            current_frame=frame(enemy_health=70.0),
            resolution_step=2,
            current_feature_units=raw_units(
                enemy_health=70.0,
                friendly_cooldowns=(15.0, 1.0),
            ),
        )

        attack_outcomes = [outcome for outcome in outcomes if outcome.outcome_class == "attack_likely"]
        self.assertEqual(len(attack_outcomes), 1)
        self.assertEqual([outcome.outcome_class for outcome in outcomes], ["attack_likely"])
        self.assertEqual(detector.pending_count, 1)

        later = detector.resolve(
            current_frame=frame(enemy_health=70.0),
            resolution_step=5,
            current_feature_units=raw_units(enemy_health=70.0),
        )
        self.assertNotIn("attack_likely", [outcome.outcome_class for outcome in later])


class CooldownHelperTests(unittest.TestCase):
    def test_extract_cooldown_snapshots_uses_raw_tag_index_29(self):
        snapshots = extract_cooldown_snapshots_from_feature_units(raw_units())

        self.assertEqual([snapshot.unit_tag for snapshot in snapshots], [101, 102])
        self.assertEqual(snapshots[0].cooldown, 0.0)
        self.assertEqual(snapshots[0].x, 20.0)
        self.assertEqual(snapshots[0].y, 30.0)

    def test_extract_cooldown_snapshots_supports_object_units(self):
        units = [
            SimpleNamespace(alliance=1, health=50, weapon_cooldown=7, x=20, y=30, tag=101),
            SimpleNamespace(alliance=4, health=80, weapon_cooldown=0, x=40, y=50, tag=201),
        ]

        snapshots = extract_cooldown_snapshots_from_feature_units(units)

        self.assertEqual(len(snapshots), 1)
        self.assertEqual(snapshots[0].unit_tag, 101)
        self.assertEqual(snapshots[0].cooldown, 7.0)

    def test_check_cooldowns_fired_detects_ready_to_fired_pattern(self):
        previous = (
            CooldownSnapshot(unit_tag=101, cooldown=0.0, health=50.0, x=20.0, y=30.0),
            CooldownSnapshot(unit_tag=102, cooldown=1.0, health=45.0, x=22.0, y=32.0),
        )
        current = raw_units(friendly_cooldowns=(15.0, 1.0))

        self.assertTrue(check_cooldowns_fired_production(previous, current))

    def test_check_cooldowns_fired_ignores_already_cooling_unit(self):
        previous = (
            CooldownSnapshot(unit_tag=101, cooldown=10.0, health=50.0, x=20.0, y=30.0),
        )
        current = np.asarray(
            [raw_unit(alliance=FRIENDLY, health=50.0, cooldown=12.0, x=20.0, y=30.0, tag=101)],
            dtype=np.float32,
        )

        self.assertFalse(check_cooldowns_fired_production(previous, current))


class MiscTests(unittest.TestCase):
    def test_nearest_enemy_distance(self):
        enemies = (
            UnitSnapshot(ENEMY, 80.0, 40.0, 50.0, 201),
            UnitSnapshot(ENEMY, 60.0, 10.0, 10.0, 202),
        )

        self.assertLess(nearest_enemy_distance(enemies, (42.0, 51.0)), 3.0)
        self.assertEqual(nearest_enemy_distance((), (42.0, 51.0)), float("inf"))

    def test_outcome_to_dict(self):
        outcome = SmartOutcome(
            outcome_class="attack_likely",
            click_step=100,
            resolution_step=102,
            window_steps=2,
            target_x=40.0,
            target_y=50.0,
            enemy_health_delta=15.0,
            nearest_enemy_distance=5.0,
            any_cooldown_fired=True,
            friendly_moved_toward=False,
            target_was_near_enemy=True,
            resolution_reason="enemy_health_drop",
        )

        result = outcome.to_dict()

        self.assertEqual(result["outcome_class"], "attack_likely")
        self.assertEqual(result["target_x"], 40.0)
        self.assertEqual(result["resolution_reason"], "enemy_health_drop")

    def test_safe_float(self):
        self.assertEqual(_safe_float(3.14), 3.14)
        self.assertEqual(_safe_float(None), 0.0)
        self.assertEqual(_safe_float("invalid"), 0.0)


if __name__ == "__main__":
    unittest.main()
