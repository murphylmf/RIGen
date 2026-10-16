''
                                                                           
                                                                              
                                     
from __future__ import annotations

import random
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from typing import Any

MOTIONS = ("hold", "approach", "retreat", "left", "right")
DISTANCE_TRENDS = ("stable", "approach", "retreat")
AXIS_DIM = 4
BUTTON_DIM = 6
FIRE_BUTTON = 0
RELOAD_BUTTON = 4


def _as_bool(value: Any, default: bool = False) -> bool:
    ''
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "y", "open", "clear", "available"}:
            return True
        if normalized in {"0", "false", "no", "n", "closed", "blocked", "none", "unknown"}:
            return False
    return default


def _feedback(value: Feedback | Mapping[str, Any] | None) -> Feedback:
    if value is None:
        return Feedback()
    if isinstance(value, Feedback):
        return value
    if isinstance(value, Mapping):
        fields = {key: value[key] for key in Feedback.__dataclass_fields__ if key in value}
        if "player_hits" in fields:
            fields["player_hits"] = tuple(int(offset) for offset in fields["player_hits"])
        return Feedback(**fields)
    raise TypeError(f"Expected Feedback, mapping or None; got {type(value).__name__}")


@dataclass(frozen=True)
class CombatState:
    player_hp: int = 100
    enemy_hp: int = 100
    ammo: int = 30
    last_player_alive_hp: int = 100
    last_enemy_alive_hp: int = 100
    last_fire: bool = False
    last_reload: bool = False
    firing_allowed: bool = False
    revision: int = 0

    @property
    def player_alive(self) -> bool:
        return self.player_hp > 0

    @property
    def enemy_alive(self) -> bool:
        return self.enemy_hp > 0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class SceneObservation:
    enemy_visible: bool = False
    enemy_facing_player: bool = False
    path_to_player: bool = False
    path_away: bool = False
    path_left: bool = False
    path_right: bool = False
    cover_route: str = "none"

    def __post_init__(self) -> None:
        for field in ("enemy_visible", "enemy_facing_player", "path_to_player", "path_away",
                      "path_left", "path_right"):
            object.__setattr__(self, field, _as_bool(getattr(self, field)))
        route = str(self.cover_route).strip().lower()
        object.__setattr__(self, "cover_route", route if route in {"none", "retreat", "left", "right"} else "none")

    @classmethod
    def from_value(cls, value: SceneObservation | Mapping[str, Any] | None) -> SceneObservation:
        if value is None:
            return cls()
        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping):
            raise TypeError(f"Expected SceneObservation, mapping or None; got {type(value).__name__}")
        fields = set(cls.__dataclass_fields__)
        data = {key: value[key] for key in fields if key in value}
        if data.get("cover_route", "none") == "none" and value.get("cover_available", False):
            data["cover_route"] = "retreat"
        lateral = value.get("lateral_path")
        if lateral is not None:
            data.setdefault("path_left", lateral)
            data.setdefault("path_right", lateral)
        return cls(**data)


@dataclass(frozen=True)
class EnemyPlan:
    motion: str = "hold"
    attack_intent: bool = False
    distance_trend: str = "stable"

    def __post_init__(self) -> None:
        motion = str(self.motion).strip().lower()
        if motion not in MOTIONS:
            raise ValueError(f"Unknown enemy motion {self.motion!r}; expected one of {MOTIONS}")
        object.__setattr__(self, "motion", motion)
        object.__setattr__(self, "attack_intent", _as_bool(self.attack_intent))
        trend = str(self.distance_trend).strip().lower()
        if trend not in DISTANCE_TRENDS:
            raise ValueError(
                f"Unknown enemy distance trend {self.distance_trend!r}; "
                f"expected one of {DISTANCE_TRENDS}"
            )
        object.__setattr__(self, "distance_trend", trend)


class VLMEnemyPlanner:
    ''

    def __init__(self, observer: Any, *, allow_attack: bool = True) -> None:
        self.observer = observer
        self.allow_attack = allow_attack

    def propose(self, state: CombatState, obs: SceneObservation, image: Any = None,
                strategy: str | None = None) -> EnemyPlan:
        strategy = str(strategy or ("retreat" if state.enemy_hp <= 50 else "engage")).strip().lower()
        if strategy not in {"engage", "retreat", "dead"}:
            strategy = "engage"
        fallback = EnemyPlan("hold", False, "retreat" if strategy == "retreat" else "stable")
        if image is None:
            return fallback
        from ltx_trainer.vlm_video_authoritative_v9 import image_url, request_images

                                                                               
                                                                               
                                                                              
                                                                             
        context = {
            "enemy_hp": state.enemy_hp,
            "player_hp": state.player_hp,
            "enemy_visible": obs.enemy_visible,
            "enemy_facing_player": obs.enemy_facing_player,
            "path_to_player": obs.path_to_player,
            "path_away": obs.path_away,
            "path_left": obs.path_left,
            "path_right": obs.path_right,
            "cover_route": obs.cover_route,
            "enemy_strategy": strategy,
        }
        if strategy == "retreat":
            strategy_instruction = (
                "The rule strategy is retreat: choose distance_trend=retreat. "
                "Prefer retreat when a clear path away or cover exists; otherwise choose a visible "
                "left or right route while still moving away. Never choose approach."
            )
        elif strategy == "engage":
            strategy_instruction = (
                "The rule strategy is engage: choose distance_trend=approach or stable, never retreat. "
                "Choose approach when the visible path to the player is usable; otherwise choose a "
                "visible left or right route or hold."
            )
        else:
            strategy_instruction = "The rule strategy is dead: choose hold with distance_trend=stable."
        attack_instruction = (
            "The opponent may attack only when visible and facing the player."
            if self.allow_attack
            else "Do not plan an opponent attack in this chain; return false for attack_intent."
        )
        vocabulary_instruction = (
            "Use the training motion vocabulary exactly: motion=approach means "
            "'moves toward the player'; motion=retreat means 'moves away from the player'; "
            "motion=left/right means 'moves toward the player's left/right'. For left/right, "
            "distance_trend=approach adds 'while approaching the player', "
            "distance_trend=retreat adds 'while moving away from the player', and stable "
            "has no distance clause. motion=hold means 'remains stationary'."
        )
        result = request_images(
            self.observer,
            "Choose the next opponent behaviour from CURRENT. Observe only visible "
            "paths, cover and facing. " + strategy_instruction + " " + vocabulary_instruction + " "
            + attack_instruction + " "
            "Do not plan player actions. The structured state/observation below is "
            "a planning constraint, not evidence to override the image: "
            + json.dumps(context, sort_keys=True)
            + ". Return enum values only.",
            [("CURRENT", image_url(image))],
            {
                "motion": list(MOTIONS),
                "distance_trend": list(DISTANCE_TRENDS),
                "attack_intent": ["true", "false"] if self.allow_attack else ["false"],
            },
        )
        if not isinstance(result, Mapping):
            return fallback
        raw_attack = result.get("attack_intent", "false")
        motion = str(result.get("motion", "hold")).strip().lower()
        if motion not in MOTIONS:
            motion = "hold"
        trend = str(result.get("distance_trend", "retreat" if strategy == "retreat" else "stable"))
        trend = trend.strip().lower()
        if trend not in DISTANCE_TRENDS:
            trend = "retreat" if strategy == "retreat" else "stable"
        return EnemyPlan(
            motion=motion,
            attack_intent=(raw_attack is True or str(raw_attack).lower() == "true")
            if self.allow_attack else False,
            distance_trend=trend,
        )


class VLMSceneObserver:
    ''

    def __init__(self, observer: Any) -> None:
        self.observer = observer

    def observe(self, image: Any) -> SceneObservation:
        if image is None:
            return SceneObservation()
        from ltx_trainer.vlm_video_authoritative_v9 import image_url, request_images

        result = request_images(
            self.observer,
            "Inspect CURRENT only. Report visible enemy state and traversable paths. "
            "Do not infer an enemy from the prompt and do not plan player actions. "
            "Return only the requested enum and boolean fields.",
            [("CURRENT", image_url(image))],
            {
                "enemy_visible": ["true", "false"],
                "enemy_facing_player": ["true", "false"],
                "path_to_player": ["true", "false"],
                "path_away": ["true", "false"],
                "path_left": ["true", "false"],
                "path_right": ["true", "false"],
                "cover_route": ["none", "retreat", "left", "right"],
            },
        )
        if not isinstance(result, Mapping):
            return SceneObservation()
        return SceneObservation(
            enemy_visible=result.get("enemy_visible", False),
            enemy_facing_player=result.get("enemy_facing_player", False),
            path_to_player=result.get("path_to_player", False),
            path_away=result.get("path_away", False),
            path_left=result.get("path_left", False),
            path_right=result.get("path_right", False),
            cover_route=result.get("cover_route", "none"),
        )


@dataclass(frozen=True)
class RuleConfig:
    enemy_retreat_threshold: int = 50
    enemy_attack_rate: float = 0.5                                    
    enemy_hit_probability: float = 0.5
    enemy_shot_damage: int = 25
    player_hit_damage: int = 50
    magazine_size: int = 30
    allow_enemy_approach: bool = True

    def __post_init__(self) -> None:
        if not all(0 <= p <= 1 for p in (self.enemy_attack_rate, self.enemy_hit_probability)):
            raise ValueError("Attack and hit probabilities must lie in [0, 1]")
        if min(self.enemy_shot_damage, self.player_hit_damage, self.enemy_retreat_threshold) < 0:
            raise ValueError("Damage and retreat threshold must be nonnegative")
        if self.magazine_size < 1:
            raise ValueError("Magazine size must be positive")


@dataclass(frozen=True)
class ActionGate:
    state: CombatState
    axes: tuple[tuple[float, ...], ...]
    buttons: tuple[tuple[float, ...], ...]
    fire_offsets: tuple[int, ...]
    reload_offsets: tuple[int, ...]


@dataclass(frozen=True)
class Proposal:
    block: int
    prior: CombatState
    gate: ActionGate
    observation: SceneObservation
    plan: EnemyPlan
    player_hits: tuple[int, ...]
    enemy_attack: bool
    enemy_hit: bool
    planned: CombatState
    guidance: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class Feedback:
                                                                                     
    enemy_status: str = "unknown"
    player_status: str = "unknown"
    enemy_attack: str = "unknown"
    player_hit: str = "unknown"
    reload_completed: str = "unknown"
    player_hits: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if self.enemy_status not in {"alive", "dead", "not_visible", "unknown"}:
            raise ValueError("Invalid enemy status")
        if self.player_status not in {"alive", "dead", "unknown"}:
            raise ValueError("Invalid player status")
        if any(v not in {"yes", "no", "unknown"} for v in
               (self.enemy_attack, self.player_hit, self.reload_completed)):
            raise ValueError("Invalid event feedback")

    @classmethod
    def from_value(cls, value: Feedback | Mapping[str, Any] | None) -> Feedback:
        return _feedback(value)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class CombatStateEngine:
    def __init__(self, config: RuleConfig | None = None, *, seed: int = 0) -> None:
        self.config = config or RuleConfig()
        self.random = random.Random(seed)
        self.committed: dict[int, tuple[Proposal, Feedback, CombatState]] = {}

    def initial_state(self, *, player_hp: int = 100, enemy_hp: int = 100, ammo: int = 30) -> CombatState:
        if min(player_hp, enemy_hp, ammo) < 0 or ammo > self.config.magazine_size:
            raise ValueError("Initial HP/ammunition outside valid range")
        return CombatState(player_hp=player_hp, enemy_hp=enemy_hp, ammo=ammo,
                           last_player_alive_hp=max(1, player_hp), last_enemy_alive_hp=max(1, enemy_hp))

    def strategy(self, state: CombatState) -> str:
        if not state.enemy_alive:
            return "dead"
        return "retreat" if state.enemy_hp <= self.config.enemy_retreat_threshold else "engage"

    def gate_actions(
        self,
        state: CombatState,
        axes: Sequence[Sequence[float]],
        buttons: Sequence[Sequence[float]],
        *,
        allow_fire: bool = True,
    ) -> ActionGate:
        axes_rows = [list(row) for row in axes]
        button_rows = [list(row) for row in buttons]
        if (
            not axes_rows
            or len(axes_rows) != len(button_rows)
            or any(len(x) != AXIS_DIM for x in axes_rows)
            or any(len(b) != BUTTON_DIM for b in button_rows)
        ):
            raise ValueError("Expected matching nonempty [F,4] axes and [F,6] buttons")
        actual = button_rows
        motion = axes_rows
        ammo, last_fire, last_reload = state.ammo, state.last_fire, state.last_reload
        allowed = state.firing_allowed
        shots, reloads = [], []
        for i, row in enumerate(actual):
            requested_fire, reload = row[FIRE_BUTTON] > 0, row[RELOAD_BUTTON] > 0
            fire = requested_fire and allow_fire
            if not allow_fire:
                row[FIRE_BUTTON] = 0.0
            if not state.player_alive:
                row[:] = [0.] * 6
                motion[i] = [0.] * 4
            else:
                if reload and not last_reload:
                    reloads.append(i)
                                                                            
                pulse = fire and not last_fire
                if pulse:
                    allowed = ammo > 0 and not reload
                    if allowed:
                        ammo -= 1
                        shots.append(i)
                keep_fire = pulse and allowed and not reload
                if not keep_fire:
                    row[FIRE_BUTTON] = 0.0
                else:
                    row[FIRE_BUTTON] = 1.0
            last_fire, last_reload = fire, reload
        gated = replace(state, ammo=ammo, last_fire=last_fire, last_reload=last_reload, firing_allowed=allowed)
        return ActionGate(gated, tuple(map(tuple, motion)), tuple(map(tuple, actual)), tuple(shots), tuple(reloads))

    def legal_plan(self, state: CombatState, obs: SceneObservation, requested: EnemyPlan) -> EnemyPlan:
                                                                            
                                                                           
                                                                             
                                     
        legal = {
            "hold": True,
            "approach": obs.path_to_player and self.config.allow_enemy_approach,
            "retreat": obs.path_away or obs.cover_route == "retreat",
            "left": obs.path_left or obs.cover_route == "left",
            "right": obs.path_right or obs.cover_route == "right",
        }
        if not state.enemy_alive or not obs.enemy_visible or not state.player_alive:
            return EnemyPlan()
        strategy = self.strategy(state)
        if strategy == "retreat":
                                                                           
                                                                            
                                                                            
                                                                     
                                                                            
                           
            requested_motion = requested.motion
            candidates = []
            if requested_motion in {"retreat", "left", "right"}:
                candidates.append((requested_motion, "retreat"))
            if obs.cover_route in {"retreat", "left", "right"}:
                candidates.append((obs.cover_route, "retreat"))
            candidates.extend(
                (
                    ("retreat", "retreat"),
                    ("left", "retreat"),
                    ("right", "retreat"),
                    ("hold", "stable"),
                )
            )
            motion, trend = next(
                (candidate_motion, candidate_trend)
                for candidate_motion, candidate_trend in candidates
                if legal.get(candidate_motion, False)
            )
            return EnemyPlan(motion, False, trend)

                                                                           
                                                                           
        motion = requested.motion
        if motion == "approach":
            if legal["approach"]:
                return EnemyPlan("approach", requested.attack_intent, "approach")
            motion = "left" if legal["left"] else "right" if legal["right"] else "hold"
        elif motion == "retreat":
            motion = "left" if legal["left"] else "right" if legal["right"] else "approach" if legal["approach"] else "hold"
        elif motion not in {"hold", "left", "right"} or not legal.get(motion, False):
            motion = "approach" if legal["approach"] else "left" if legal["left"] else "right" if legal["right"] else "hold"

        if motion == "hold":
            return EnemyPlan("hold", requested.attack_intent, "stable")
        if motion == "approach":
            return EnemyPlan("approach", requested.attack_intent, "approach")
        trend = requested.distance_trend if requested.distance_trend in {"stable", "approach"} else "stable"
        if not self.config.allow_enemy_approach and trend == "approach":
            trend = "stable"
        return EnemyPlan(motion, requested.attack_intent, trend)

    def propose(self, *, state: CombatState, observation: SceneObservation, plan: EnemyPlan,
                gate: ActionGate, player_hits: tuple[int, ...] = (), block: int) -> Proposal:
        player_hits = tuple(int(offset) for offset in player_hits)
        if block in self.committed:
            raise ValueError("Block already committed")
        if any(i not in gate.fire_offsets for i in player_hits):
            raise ValueError("Hit without an executed shot")
        plan = self.legal_plan(state, observation, plan)
        can_attack = (plan.attack_intent and state.enemy_alive and state.player_alive
                      and observation.enemy_visible and observation.enemy_facing_player
                      and self.strategy(state) == "engage")
        attempt = can_attack and self.random.random() < self.config.enemy_attack_rate
        hit = attempt and self.random.random() < self.config.enemy_hit_probability
        predicted = replace(gate.state,
                            player_hp=max(0, state.player_hp - self.config.enemy_shot_damage * hit),
                            enemy_hp=max(0, state.enemy_hp - self.config.player_hit_damage * len(set(player_hits))))
        guidance = self.guidance(state, predicted, plan, attempt, hit, bool(player_hits))
        return Proposal(block, state, gate, observation, plan, player_hits, attempt, hit, predicted, guidance)

    @staticmethod
    def guidance(prior: CombatState, planned: CombatState, plan: EnemyPlan,
                 attack: bool, enemy_hit: bool, player_hit: bool) -> str:
        if plan.motion == "hold":
            motion = "The opponent remains stationary."
        elif plan.motion == "approach":
            motion = "The opponent moves toward the player."
        elif plan.motion == "retreat":
            motion = "The opponent moves away from the player."
        else:
            side = "left" if plan.motion == "left" else "right"
            motion = f"The opponent moves toward the player's {side}"
            if plan.distance_trend == "approach":
                motion += " while approaching the player"
            elif plan.distance_trend == "retreat":
                motion += " while moving away from the player"
            motion += "."
        if not prior.enemy_alive:
            motion = "The opponent starts this block dead and remains dead throughout the block."
        elif not planned.enemy_alive:
            motion = "The opponent is killed during this block, falls to the ground and remains lying on the ground."
        elif player_hit:
            motion += " The opponent is hit during this block and remains alive."
        if attack:
            motion += " The opponent fires the machine gun toward the player during this block."
        if not prior.player_alive:
            motion += " The player remains dead. The first-person view stays still."
        elif not planned.player_alive:
            motion += " The opponent kills the player during this block."
        elif enemy_hit:
            motion += " The player is hit by the opponent during this block and remains alive."
        return motion

    def reconcile(self, proposal: Proposal, feedback: Feedback | Mapping[str, Any] | None) -> CombatState:
        feedback = Feedback.from_value(feedback)
        previous = self.committed.get(proposal.block)
        if previous is not None:
            if previous[:2] != (proposal, feedback):
                raise ValueError("Conflicting duplicate block audit")
            return previous[2]
        if proposal.prior.revision != len(self.committed):
            raise ValueError("Stale or out-of-order state revision")
        if any(i not in proposal.gate.fire_offsets for i in feedback.player_hits):
            raise ValueError("Observed hit without an executed shot")
        prior = proposal.prior
        confirmed_hits = len(set(feedback.player_hits))
        hit_damage = self.config.player_hit_damage
        hp_enemy = max(0, prior.enemy_hp - hit_damage * confirmed_hits)
                                                                                
                                                                                
                                                                            
                                                                          
                                                                           
                                                       
        visual_attack = (
            proposal.plan.attack_intent
            and feedback.enemy_attack == "yes"
            and feedback.player_hit == "yes"
        )
        sampled_attack = (
            proposal.enemy_attack
            and feedback.enemy_attack == "yes"
            and (feedback.player_hit == "yes"
                 or (proposal.enemy_hit and feedback.player_hit == "unknown"))
        )
        accepted_enemy_hit = visual_attack or sampled_attack
        hp_player = max(0, prior.player_hp - self.config.enemy_shot_damage * accepted_enemy_hit)
                                                                             
                                                                             
                                                                     
                                                                              
                                                                               
                                                                                 
                                                         
        if prior.enemy_hp == 0:
            hp_enemy = 0
        elif feedback.enemy_status == "dead":
            # The END-state audit is lifecycle telemetry only.  It cannot spend
            # damage by itself. This keeps a false corpse/death label from
            # turning a missed point shot into a kill.
            hp_enemy = hp_enemy
        elif feedback.enemy_status == "alive" and hp_enemy == 0:
                                                                              
                                                                              
                                                                  
            hp_enemy = prior.enemy_hp
        elif feedback.enemy_status == "not_visible" and hp_enemy == 0:
                                                                             
                                                                                 
            hp_enemy = prior.enemy_hp
        player_death_evidence = (
            feedback.enemy_attack == "yes" and feedback.player_hit == "yes"
        ) or prior.player_hp == 0
        if player_death_evidence and feedback.player_status == "dead":
            hp_player = 0
        elif prior.player_hp == 0:
            hp_player = 0
        elif feedback.player_status == "alive" and hp_player == 0:
            hp_player = prior.player_hp
        ammo = proposal.gate.state.ammo
        if proposal.gate.reload_offsets and feedback.reload_completed == "yes" and prior.player_alive:
                                                                                        
            last_reload = proposal.gate.reload_offsets[-1]
            ammo = max(0, self.config.magazine_size - sum(i > last_reload for i in proposal.gate.fire_offsets))
        state = replace(proposal.gate.state, enemy_hp=hp_enemy, player_hp=hp_player, ammo=ammo,
                        last_enemy_alive_hp=hp_enemy or prior.last_enemy_alive_hp,
                        last_player_alive_hp=hp_player or prior.last_player_alive_hp,
                        revision=prior.revision + 1)
        self.committed[proposal.block] = proposal, feedback, state
        return state


