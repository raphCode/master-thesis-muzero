from typing import Optional

from attrs import define

from networks.bases import Networks


@define
class Globals:
    """
    This class contains data that is meant to change while training, like the network
    parameters or the number of epochs and games.
    This is in contrast to the global config C which is completely static.
    """

    epoch_num: int = 0
    game_num: int = 0
    nets: Optional[Networks] = None


G = Globals()
