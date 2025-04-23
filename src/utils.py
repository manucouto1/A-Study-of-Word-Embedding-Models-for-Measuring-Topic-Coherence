from typing import Set


def overwrite_first(model: str, seen_models: Set[str], apply: bool = False) -> bool:
    if model in seen_models or not apply:
        return False
    else:
        seen_models.add(model)
        return True
