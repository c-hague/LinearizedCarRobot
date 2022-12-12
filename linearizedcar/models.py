from dataclasses import dataclass
from enum import IntEnum, auto
import numpy as np


class DubinsPathType(IntEnum):
    """Different Dubins Paths"""
    UNKNOWN = auto()
    LSL = auto()
    LSR = auto()
    RSL = auto()
    RSR = auto()
    LRL = auto()
    RLR = auto()


@dataclass
class Vertex:
    """class for vertex of a graph"""
    x: float = 0
    y: float = 0
    psi: float = 0
    id: int = -1

    @staticmethod
    def fromList(a):
        return Vertex(*a)

    def __eq__(self, other):
        if not isinstance(other, Vertex):
            return False
        return self.id == other.id
    
    def toarray(self):
        return np.array([self.x, self.y, self.psi])


@dataclass
class DubinsPath:
    """class for dubins path data"""
    start: 'Vertex' = None
    end: 'Vertex' = None
    a: float = 0
    b: float = 0
    c: float = 0
    r: float = 0
    type: DubinsPathType = 0
    cost: float = 0
    n: int = 0
