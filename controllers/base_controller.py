class Controller:
    """
    Base Controller Interface for BiVAD environment.
    All controllers must implement get_action(observation).
    """
    def __init__(self):
        pass

    def get_action(self, observation, info=None):
        """
        Takes in the normalized observation (length 9 array) and an optional
        info dictionary, and returns the action (delta_lvad, delta_rvad).
        """
        raise NotImplementedError("Must implement get_action in subclasses.")
