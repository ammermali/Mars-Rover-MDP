from enum import Enum

class State(Enum):
    START = 0
    SITE_A = 1
    SITE_B = 2
    IMMOBILE = 3
    BASE = 4
    DESTROYED = 5

    def is_final(self) -> bool:
        return self in [self.IMMOBILE, self.BASE, self.DESTROYED]