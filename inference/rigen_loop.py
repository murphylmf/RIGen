''
                                                                         
                    
from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any

from rigen_semantic_core import (
    CombatState,
    EnemyPlan,
    Feedback,
    CombatStateEngine,
    Proposal,
    SceneObservation,
    VLMEnemyPlanner,
)


class RIGenAREnemyPlanner:
    ''

    def __init__(self, *, allow_attack: bool = False, force_hold_low_health: bool = False) -> None:
        self.allow_attack = allow_attack
        self.force_hold_low_health = force_hold_low_health

    def propose(self, state: CombatState, obs: SceneObservation) -> EnemyPlan:
        if not state.enemy_alive:
            return EnemyPlan("hold", False, "stable")
        if state.enemy_hp <= 50:
            if self.force_hold_low_health:
                                                                          
                                                                   
                return EnemyPlan("hold", False, "stable")
                                                                            
                                                                       
                                                            
            if obs.cover_route in {"left", "right", "retreat"}:
                return EnemyPlan(obs.cover_route, False, "retreat")
            if obs.path_away:
                return EnemyPlan("retreat", False, "retreat")
            if obs.path_left:
                return EnemyPlan("left", False, "retreat")
            if obs.path_right:
                return EnemyPlan("right", False, "retreat")
            return EnemyPlan("hold", False, "stable")

                                                                          
                                                                     
        if obs.path_to_player:
            return EnemyPlan("approach", self.allow_attack, "approach")
        if obs.path_left:
            return EnemyPlan("left", self.allow_attack, "stable")
        if obs.path_right:
            return EnemyPlan("right", self.allow_attack, "stable")
        return EnemyPlan("hold", False, "stable")


class RIGenVLMEnemyPlanner:
    ''

    requires_image = True

    def __init__(self, observer: Any, *, fallback: RIGenAREnemyPlanner | None = None,
                 force_hold_low_health: bool = False) -> None:
        self.observer = observer
        self.vlm = VLMEnemyPlanner(observer, allow_attack=False)
        self.fallback = fallback or RIGenAREnemyPlanner()
        self.force_hold_low_health = force_hold_low_health

    def propose(self, state: CombatState, obs: SceneObservation, image: Any = None,
                strategy: str | None = None) -> EnemyPlan:
        if image is None:
            return self.fallback.propose(state, obs)
        plan = self.vlm.propose(state, obs, image, strategy=strategy)
        if self.force_hold_low_health and state.enemy_hp <= 50 and state.enemy_alive:
            return EnemyPlan("hold", False, "stable")
                                                                            
                                                                           
        return EnemyPlan(plan.motion, False, plan.distance_trend)


class RIGenAREnemyChain:
    ''

    def __init__(self, *, output_dir: Path | None = None, rules: CombatStateEngine | None = None,
                 planner: Any | None = None, initial_enemy_hp: int = 100,
                 initial_player_hp: int = 100,
                 feedback_provider: Callable[[Proposal], Feedback | Mapping[str, Any] | None] | None = None,
                 respawn_enabled: bool = True) -> None:
        self.output_dir = output_dir
        self.rules = rules or CombatStateEngine(seed=0)
        self.planner = planner or RIGenAREnemyPlanner()
        self.state = self.rules.initial_state(player_hp=initial_player_hp, enemy_hp=initial_enemy_hp)
        self.initial_enemy_hp = int(initial_enemy_hp)
        self.respawn_enabled = bool(respawn_enabled)
        self.enemy_death_block: int | None = None
        self.enemy_phase = "active"
        self.respawn_cycle = 0
        self.phase_by_block: dict[int, str] = {}
        self.respawn_by_block: dict[int, int] = {}
        self.proposals: list[Proposal] = []
        self.feedback_provider = feedback_provider
        self.feedback: dict[int, Feedback] = {}
        self.states_after: dict[int, CombatState] = {}
        self.trace: list[dict[str, Any]] = []

    def _advance_enemy_lifecycle(self, block: int) -> None:
        if self.enemy_death_block is None:
            self.enemy_phase = "active"
            return
        if not self.respawn_enabled:
            self.enemy_phase = "dead"
            self.state = replace(self.state, enemy_hp=0)
            return
        elapsed = block - self.enemy_death_block
        if elapsed < 3:
            self.enemy_phase = "corpse"
            self.state = replace(self.state, enemy_hp=0)
        elif elapsed == 3:
            self.enemy_phase = "despawn"
            self.state = replace(self.state, enemy_hp=0)
        elif elapsed == 4:
            self.enemy_phase = "reentry"
            self.state = replace(
                self.state,
                enemy_hp=self.initial_enemy_hp,
                last_enemy_alive_hp=self.initial_enemy_hp,
            )
            self.respawn_cycle += 1
        else:
            self.enemy_phase = "active"
            self.enemy_death_block = None

    def _lifecycle_guidance(self, guidance: str) -> str:
        if self.enemy_phase == "dead":
            return "The opponent was killed in an earlier block and remains dead for the rest of this rollout. The opponent does not fire and does not re-enter."
        if self.enemy_phase == "corpse":
            return "The opponent starts this block dead and remains a motionless corpse throughout the block. The opponent does not fire."
        if self.enemy_phase == "despawn":
            return "The previous opponent's corpse disappears at the start of this block. No opponent remains visible and the opponent does not fire."
        if self.enemy_phase == "reentry":
            return "A new opponent re-enters the player's view during this block. The opponent does not fire during re-entry."
        return guidance

    def configure_vlm_planner(self, observer: Any, *, force_hold_low_health: bool = False) -> None:
        ''
        fallback = self.planner if not getattr(self.planner, "requires_image", False) else None
        self.planner = RIGenVLMEnemyPlanner(
            observer,
            fallback=fallback,
            force_hold_low_health=force_hold_low_health,
        )

    def _propose_enemy(self, state: CombatState, observation: SceneObservation,
                       image: Any = None) -> EnemyPlan:
        if getattr(self.planner, "requires_image", False):
            return self.planner.propose(
                state, observation, image=image, strategy=self.rules.strategy(state)
            )
        return self.planner.propose(state, observation)

    def before_block(self, *, block: int, observation: SceneObservation, axes: list,
                     buttons: list, image: Any = None) -> Proposal:
        if self.proposals and self.proposals[-1].block not in self.feedback:
            raise ValueError("Previous RIGen block must be reconciled before proposing the next block")
        if self.proposals and block <= self.proposals[-1].block:
            raise ValueError("RIGen blocks must be strictly increasing")
        self._advance_enemy_lifecycle(block)
        self.phase_by_block[block] = self.enemy_phase
        self.respawn_by_block[block] = self.respawn_cycle
        gate = self.rules.gate_actions(
            self.state,
            axes,
            buttons,
            allow_fire=self.enemy_phase == "active" and self.state.player_alive,
        )
        requested_plan = self._propose_enemy(self.state, observation, image=image)
                                                                         
                                                                       
                                                                
        plan = EnemyPlan(requested_plan.motion, False, requested_plan.distance_trend)
                                                                            
                                                                           
                                                                            
                                                
        forecast_hits = gate.fire_offsets if observation.enemy_visible else ()
        proposal = self.rules.propose(state=self.state, observation=observation,
                                      plan=plan, gate=gate,
                                      player_hits=forecast_hits, block=block)
        proposal = replace(proposal, guidance=self._lifecycle_guidance(proposal.guidance))
        self.proposals.append(proposal)
        self._write()
        return proposal

    def after_block(self, proposal: Proposal, feedback: Any = None) -> CombatState:
        if not self.proposals or proposal is not self.proposals[-1]:
            raise ValueError("RIGen blocks must be reconciled in proposal order")
        if proposal.block in self.feedback:
            raise ValueError(f"Block {proposal.block} has already been reconciled")
        if feedback is None and self.feedback_provider is not None:
            feedback = self.feedback_provider(proposal)
        feedback = Feedback.from_value(feedback)
        self.state = self.rules.reconcile(proposal, feedback)
        if self.enemy_phase == "active" and proposal.prior.enemy_alive and not self.state.enemy_alive:
            self.enemy_death_block = proposal.block
        elif self.enemy_phase == "active" and self.state.enemy_alive:
            self.enemy_death_block = None
        self.feedback[proposal.block] = feedback
        self.states_after[proposal.block] = self.state
        self._write()
        return self.state

    def _write(self) -> None:
        self.trace = []
        for proposal in self.proposals:
            audit = self.feedback.get(proposal.block)
            after = self.states_after.get(proposal.block)
            self.trace.append({
                "block": proposal.block,
                "enemy_phase": self.phase_by_block.get(proposal.block, "unknown"),
                "respawn_cycle": self.respawn_by_block.get(proposal.block, 0),
                "proposal": proposal.to_dict(),
                "feedback": audit.to_dict() if audit is not None else None,
                "state_after": after.to_dict() if after is not None else None,
            })
        if self.output_dir is None:
            return
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "semantic_rigen_trace.json").write_text(
            json.dumps(self.trace, indent=2) + "\n", encoding="utf-8")
