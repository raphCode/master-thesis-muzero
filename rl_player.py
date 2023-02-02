from collections.abc import Sequence

from attrs import frozen

from trajectory import InfoType


@frozen(kw_only=True)
class TrainingInfo:
    """
    Information recorded by the RLPlayers to enable training.
    """

    info: InfoType
    target_policy: Sequence[float]
    mcts_value: float
