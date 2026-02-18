from typing import List, Tuple, Dict, Any


class Adapter:

    def __init__(self, config: Dict[str, Any]):
        self.config = config

    def handle(
        self, event_name: str, payload: Dict[str, Any]
    ) -> List[Tuple[str, Dict[str, Any]]]:
        raise NotImplementedError
