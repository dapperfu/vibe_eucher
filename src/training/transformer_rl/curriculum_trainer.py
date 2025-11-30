"""Training curriculum system for progressive learning.

Implements a 7-stage curriculum that gradually introduces complexity:
1. Learn card legality and basic play
2. Learn trick strategy using self-play
3. Add full deduction logic and perfect memory
4. Introduce bidding and risk tuning
5. Full hand reinforcement learning
6. Multi-agent self-play with varying risk
7. Evaluation against fixed bots
"""

from enum import Enum
from typing import Dict, List, Optional

from src.ai_players.transformer_rl.actor_critic_agent import ActorCriticAgent


class TrainingStage(Enum):
    """Training stages in the curriculum."""

    STAGE_1_BASIC_PLAY = 1
    STAGE_2_TRICK_STRATEGY = 2
    STAGE_3_DEDUCTION = 3
    STAGE_4_BIDDING = 4
    STAGE_5_FULL_RL = 5
    STAGE_6_MULTI_AGENT = 6
    STAGE_7_EVALUATION = 7


class CurriculumTrainer:
    """Manages training curriculum with automatic stage progression.

    Parameters
    ----------
    agent : ActorCriticAgent
        The RL agent to train.
    stage_progression_threshold : float
        Performance threshold for stage progression (0.0-1.0).
    """

    def __init__(
        self,
        agent: ActorCriticAgent,
        stage_progression_threshold: float = 0.7,
    ) -> None:
        """Initialize curriculum trainer.

        Parameters
        ----------
        agent : ActorCriticAgent
            RL agent to train.
        stage_progression_threshold : float
            Performance threshold for progression.
        """
        self.agent = agent
        self.current_stage = TrainingStage.STAGE_1_BASIC_PLAY
        self.stage_progression_threshold = stage_progression_threshold

        # Stage-specific configurations
        self.stage_configs: Dict[TrainingStage, Dict] = {
            TrainingStage.STAGE_1_BASIC_PLAY: {
                "use_deduction": False,
                "use_bidding": False,
                "use_risk_tuning": False,
                "focus": "card_legality",
                "opponent_type": "random",
            },
            TrainingStage.STAGE_2_TRICK_STRATEGY: {
                "use_deduction": False,
                "use_bidding": False,
                "use_risk_tuning": False,
                "focus": "trick_strategy",
                "opponent_type": "self_play",
            },
            TrainingStage.STAGE_3_DEDUCTION: {
                "use_deduction": True,
                "use_bidding": False,
                "use_risk_tuning": False,
                "focus": "deduction",
                "opponent_type": "self_play",
            },
            TrainingStage.STAGE_4_BIDDING: {
                "use_deduction": True,
                "use_bidding": True,
                "use_risk_tuning": True,
                "focus": "bidding",
                "opponent_type": "self_play",
            },
            TrainingStage.STAGE_5_FULL_RL: {
                "use_deduction": True,
                "use_bidding": True,
                "use_risk_tuning": True,
                "focus": "full_game",
                "opponent_type": "self_play",
            },
            TrainingStage.STAGE_6_MULTI_AGENT: {
                "use_deduction": True,
                "use_bidding": True,
                "use_risk_tuning": True,
                "focus": "multi_agent",
                "opponent_type": "self_play",
                "varying_risk": True,
            },
            TrainingStage.STAGE_7_EVALUATION: {
                "use_deduction": True,
                "use_bidding": True,
                "use_risk_tuning": True,
                "focus": "evaluation",
                "opponent_type": "fixed_bots",
            },
        }

        # Performance tracking
        self.stage_performance: Dict[TrainingStage, List[float]] = {
            stage: [] for stage in TrainingStage
        }

    def get_stage_config(self) -> Dict:
        """Get configuration for current stage.

        Returns
        -------
        Dict
            Stage configuration dictionary.
        """
        return self.stage_configs[self.current_stage].copy()

    def record_performance(self, performance_metric: float) -> None:
        """Record performance metric for current stage.

        Parameters
        ----------
        performance_metric : float
            Performance metric (e.g., win rate, average tricks).
        """
        self.stage_performance[self.current_stage].append(performance_metric)

    def should_progress_stage(self) -> bool:
        """Check if should progress to next stage.

        Returns
        -------
        bool
            True if should progress, False otherwise.
        """
        if self.current_stage == TrainingStage.STAGE_7_EVALUATION:
            return False  # Final stage

        performance_history = self.stage_performance[self.current_stage]
        if len(performance_history) < 10:
            return False  # Need minimum samples

        # Check if recent performance meets threshold
        recent_performance = sum(performance_history[-10:]) / 10.0
        return recent_performance >= self.stage_progression_threshold

    def progress_stage(self) -> bool:
        """Progress to next stage if criteria met.

        Returns
        -------
        bool
            True if progressed, False otherwise.
        """
        if not self.should_progress_stage():
            return False

        # Move to next stage
        current_idx = self.current_stage.value
        if current_idx < 7:
            self.current_stage = TrainingStage(current_idx + 1)
            return True

        return False

    def get_current_stage(self) -> TrainingStage:
        """Get current training stage.

        Returns
        -------
        TrainingStage
            Current stage.
        """
        return self.current_stage

    def get_stage_name(self) -> str:
        """Get human-readable stage name.

        Returns
        -------
        str
            Stage name.
        """
        stage_names = {
            TrainingStage.STAGE_1_BASIC_PLAY: "Stage 1: Basic Play",
            TrainingStage.STAGE_2_TRICK_STRATEGY: "Stage 2: Trick Strategy",
            TrainingStage.STAGE_3_DEDUCTION: "Stage 3: Deduction",
            TrainingStage.STAGE_4_BIDDING: "Stage 4: Bidding",
            TrainingStage.STAGE_5_FULL_RL: "Stage 5: Full RL",
            TrainingStage.STAGE_6_MULTI_AGENT: "Stage 6: Multi-Agent",
            TrainingStage.STAGE_7_EVALUATION: "Stage 7: Evaluation",
        }
        return stage_names.get(self.current_stage, "Unknown Stage")

    def get_stage_stats(self) -> Dict:
        """Get statistics for current stage.

        Returns
        -------
        Dict
            Stage statistics.
        """
        performance_history = self.stage_performance[self.current_stage]
        if not performance_history:
            return {
                "stage": self.get_stage_name(),
                "samples": 0,
                "avg_performance": 0.0,
                "recent_performance": 0.0,
            }

        return {
            "stage": self.get_stage_name(),
            "samples": len(performance_history),
            "avg_performance": sum(performance_history) / len(performance_history),
            "recent_performance": sum(performance_history[-10:]) / min(10, len(performance_history)),
        }





