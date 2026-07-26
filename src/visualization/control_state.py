from threading import Event, Lock


class VisualizationControlState:
    def __init__(self, step_delay_seconds: float = 0.05) -> None:
        self._lock = Lock()
        self._start_requested = Event()
        self._step_delay_seconds = step_delay_seconds

    def get_step_delay_seconds(self) -> float:
        with self._lock:
            return self._step_delay_seconds

    def set_step_delay_seconds(self, value: float) -> None:
        if value < 0:
            raise ValueError("Step delay must be non-negative.")
        with self._lock:
            self._step_delay_seconds = value

    def request_simulation_start(self) -> None:
        self._start_requested.set()

    def wait_for_simulation_start(self) -> None:
        self._start_requested.wait()

    def has_simulation_started(self) -> bool:
        return self._start_requested.is_set()
