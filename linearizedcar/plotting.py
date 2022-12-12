from .models import DubinsPathType
import numpy as np

def dubinsCurve2d(s: np.array, a, b, c, r, type: DubinsPathType):
    """
    return a path function for a dubins curve where the input t is in [0, 1]
    Parameters
    ----------
    s: np.ndarray
        starting point for the curve
    a: float
        dubins path a parameter
    b: float
        dubins path b parameter
    c: float
        dubins path c paramters
    r: float
        dubins path radius
    type: DubinsPathType
        dubins path type
    
    Returns
    -------
    (float) -> np.ndarray
        path function where input goes from 0 to 1
    """
    i = maneuverToDir(type.name[0])
    j = maneuverToDir(type.name[1])
    k = maneuverToDir(type.name[2])
    s1 = r * i * np.array([-np.sin(s[2]), np.cos(s[2])]) + s[:2] + r * np.array([np.cos(i * a / r + s[2] - i * np.pi / 2), np.sin(i * a / r + s[2] - i * np.pi / 2)])
    h1 = i * a / r + s[2]
    if j == 0:
        s2 = s1 + b * np.array([np.cos(h1), np.sin(h1)])
        h2 = h1
    else:
        s2 = s1 + r * j * np.array([-np.sin(h1), np.cos(h1)]) + r * np.array([np.cos(j * b / r + h1 - j * np.pi / 2), np.sin(j * b / r + h1 - j * np.pi / 2)])
        h2 = h1 + j * b / r

    def f(t):
        t = t * (a + b + c)
        if t < a:
            p0 = r * i * np.array([-np.sin(s[2]), np.cos(s[2])])
            p1 = s[:2]
            p2 = r * np.array([np.cos(i * t / r + s[2] - i * np.pi / 2), np.sin(i * t / r + s[2] - i * np.pi / 2)])
            return p0 + p1 + p2
        if t >= a and t < a + b and j == 0:
            u = t - a
            return s1 + u * np.array([np.cos(h1), np.sin(h1)])
        if t >= a and t < a + b:
            u = t - a
            p0 = r * j * np.array([-np.sin(h1), np.cos(h1)])
            p1 = s1
            p2 = r * np.array([np.cos(j * u / r + h1 - j * np.pi / 2), np.sin(j * u / r + h1 - j * np.pi / 2)])
            return p0 + p1 + p2
        u = t - b - a
        p0 = r * k * np.array([-np.sin(h2), np.cos(h2)])
        p1 = s2
        p2 = r * np.array([np.cos(k * u / r + h2 - k * np.pi / 2), np.sin(k * u / r + h2 - k * np.pi / 2)])
        return p0 + p1 + p2
        
    return f
        
def maneuverToDir(str: str) -> int:
    """
    maps dubins curve action to an interger {-1, 0, 1}
    Paramters
    ---------
    str: str
        dubins curve action
    Returns
    -------
    int
        L -> 1, R -> -1, S -> 0
    """
    if str == 'L':
        return 1
    if str == 'R':
        return -1
    return 0