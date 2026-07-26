from model.sets.states import State

NODE_POSITIONS: dict[State, dict[str, int]] = {
    State.START: {"x": 0, "y": -600},
    State.SITE_A: {"x": 100, "y": -200},
    State.SITE_B: {"x": 400, "y": -200},
    State.BASE: {"x": 500, "y": -600},
    State.IMMOBILE: {"x": 250, "y": -400},
    State.DESTROYED: {"x": 250, "y": -800}
}