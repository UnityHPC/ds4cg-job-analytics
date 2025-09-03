class NodeNotFoundWarning(UserWarning):
    """Warning raised when a node is not found in the provided configuration file."""

    def __init__(self, node_name: str, message: str) -> None:
        """
        Initialize the warning with the missing node's name and message.
        """
        super().__init__(message)
        self.node_name = node_name
