from typing import Tuple
import numpy as np
from linearizedcar.models import Vertex, DubinsPath, DubinsPathType
"""
solves 3d path planning with obsticles for dubins airplane
Authors
-------
Collin Hague : chague@uncc.edu
References
----------
https://github.com/robotics-uncc/RobustDubins
"""


PATH_COMPARE_TOLERANCE = 1e-6
POSSIBLE_A_ZERO = 1e-6
HEADING_ALMOST_ZERO = 1e-6
DEFAULT_DUBINS = (np.nan, np.nan, np.nan, np.Infinity, DubinsPathType.UNKNOWN)


class DubinsCar(object):
    """
    Calculates 2D Dubins car paths
    Methods
    -------
    calculatePath(q0: Vertex, q1: Vertex, r: float): DubinsPath
    calculatePathType(q0: Vertex, q1: Vertex, r: float, pathType: DubinsPathType): DubinsPath
    """
    def calculatePath(self, q0: Vertex, q1: Vertex, r: float):
        """
        calculates the optimal dubins path between two points
        Parameters
        ----------
        q0: Vertex
            initial position
        q1: Vertex
            final position
        r: float
            minimum turn radius
        
        Returns
        -------
        DubinsPath
            the optimal dubins path
        """
        x, y, h = self._normalize(q0.x, q0.y, q0.psi, q1.x, q1.y, q1.psi, r)
        m_cth = np.cos(h)
        m_sth = np.sin(h)

        results = [
            list(self._solveLSL(x, y, h, m_cth, m_sth)),
            list(self._solveLSR(x, y, h, m_cth, m_sth)),
            list(self._solveRSL(x, y, h, m_cth, m_sth)),
            list(self._solveRSR(x, y, h, m_cth, m_sth)),
            list(self._solveLRL(x, y, h)),
            list(self._solveRLR(x, y, h))
        ]
        
        # transform back to r = r
        for result in results:
            result[0] *= r
            result[1] *= r
            result[2] *= r
            result[3] *= r
        results.sort(key=lambda x: x[3])
        return DubinsPath(
            start=q0, 
            end=q1,
            a=results[0][0],
            b=results[0][1],
            c=results[0][2],
            cost=results[0][3],
            type=results[0][4],
            r=r,
            n=2
        )
    
    def solveType(self, q0: Vertex, q1: Vertex, r, pathType: DubinsPathType):
        """
        calculates one type of dubins paths between two points
        Parameters
        ----------
        q0: Vertex
            initial position
        q1: Vertex
            final position
        r: float
            minimum turn radius
        
        Returns
        -------
        DubinsPath
            the dubins path with type pathType
        """
        x, y, h = self._normalize(q0.x, q0.y, q0.psi, q1.x, q1.y, q1.psi, r)
        m_cth = np.cos(h)
        m_sth = np.sin(h)
        result = DEFAULT_DUBINS
        if pathType == DubinsPathType.LSL:
            result = self._solveLSL(x, y, h ,m_cth, m_sth)
        elif pathType == DubinsPathType.LSR:
            result = self._solveLSR(x, y, h, m_cth, m_sth)
        elif pathType == DubinsPathType.RSL:
            result = self._solveRSL(x, y, h, m_cth, m_sth)
        elif pathType == DubinsPathType.RSR:
            result = self._solveRSR(x, y, h, m_cth, m_sth)
        elif pathType == DubinsPathType.LRL:
            result = self._solveLRL(x, y, h, m_cth, m_sth)
        elif pathType == DubinsPathType.RLR:
            result = self._solveRLR(x, y, h, m_cth, m_sth)
        result = list(result)
        result[0] *= r
        result[1] *= r
        result[2] *= r
        result[3] *= r
        return DubinsPath(
            start=q0, 
            end=q1,
            a=result[0],
            b=result[1],
            c=result[2],
            cost=result[3],
            type=result[4],
            r=r,
            n=2
        )
    
    def _normalize(self, x0, y0, h0, x1, y1, h1, r):
        # transform to (0, 0, 0) with r = 1
        x = (x1 - x0) / r
        y = (y1 - y0) / r
        x, y = x * np.cos(h0) + y * np.sin(h0), -x * np.sin(h0) + y * np.cos(h0)
        h = (h1 - h0) % (2 * np.pi)
        return x, y ,h
        

    def _solveLSL(self, x: float, y: float, h: float, m_cth: float, m_sth: float) -> 'Tuple[float, float, float, float, int]':
        b = np.sqrt((x - m_sth)*(x-m_sth)+(y+m_cth-1.0)*(y+m_cth-1.0))
        a = np.arctan2(y + m_cth - 1.0,  x - m_sth)
        r = DEFAULT_DUBINS

        #check to see if a is nan
        if np.isnan(a):
            return r

        # make sure a > 0
        while a < 0:
            a += np.pi
        
        # 2 solutions on 0, 2pi
        possibleA = [a, a + np.pi]
        for a in possibleA:
            c = (h-a) % (2.0*np.pi)

            #ensure endpoint is going to correct point
            newEnd = self._getEndpointBSB(a, b, c, 1, 1, 1)
            if self._compareVector(newEnd, [x, y, h]) == 0:
                r = (a, b, c, a + b + c, DubinsPathType.LSL)
    
        # no correct points return default
        return r
        
    def _solveLSR(self, x: float, y: float, h: float, m_cth: float, m_sth: float) -> 'Tuple[float, float, float, float, int]':
        b = np.sqrt((x + m_sth) * (x + m_sth) + (y - m_cth - 1) * (y - m_cth - 1) - 4)
        a = np.arctan2(2 * (x + m_sth) + b * (y - m_cth - 1), b * (x + m_sth) - 2 * (y - m_cth - 1))
        r = DEFAULT_DUBINS

        #check to see if a is nan
        if np.isnan(a):
            return r

        # make sure a > 0
        while a < 0:
            a += np.pi
        
        # 2 solutions on 0, 2pi
        possibleA = [a, a + np.pi]
        for a in possibleA:
            c = (a - h) % (2.0 * np.pi)

            #ensure endpoint is going to correct point
            newEnd = self._getEndpointBSB(a, b, c, 1, 1, -1)
            if self._compareVector(newEnd, [x, y, h]) == 0:
                r = (a, b, c, a + b + c, DubinsPathType.LSR)
    
        return r
    
    def _solveRSL(self, x: float, y: float, h: float, m_cth: float, m_sth: float) -> 'Tuple[float, float, float, float, int]':
        b = np.sqrt((x - m_sth) * (x - m_sth) + (y + m_cth + 1) * (y + m_cth + 1) - 4)
        a = np.arctan2(2 * (x - m_sth) - b * (y + m_cth + 1), b * (x - m_sth) + 2 * (y + m_cth + 1))
        r = (np.nan, np.nan, np.nan, np.Infinity, DubinsPathType.UNKNOWN)

        #check to see if a is nan
        if np.isnan(a):
            return r

        # make sure a > 0
        while a < 0:
            a += np.pi
        
        # 2 solutions on 0, 2pi
        possibleA = [a, a + np.pi]
        for a in possibleA:
            c = (a + h) % (2.0 * np.pi)

            #ensure endpoint is going to correct point
            newEnd = self._getEndpointBSB(a, b, c, -1, 1, 1)
            if self._compareVector(newEnd, [x, y, h]) == 0:
                r = (a, b, c, a + b + c, DubinsPathType.RSL)
    
        # no correct points return default
        return r
    
    def _solveRSR(self, x: float, y: float, h: float, m_cth: float, m_sth: float) -> 'Tuple[float, float, float, float, int]':
        b = np.sqrt((x + m_sth) * (x + m_sth) + (y - m_cth + 1.0) * (y - m_cth + 1.0))
        a = np.arctan2(m_cth - y - 1, x + m_sth)
        r = DEFAULT_DUBINS

        #check to see if a is nan
        if np.isnan(a):
            return r

        # make sure a > 0
        while a < 0:
            a += np.pi
        
        # 2 solutions on 0, 2pi
        possibleA = [a, a + np.pi]
        for a in possibleA:
            if abs(a) < POSSIBLE_A_ZERO:
                if abs(h) < HEADING_ALMOST_ZERO:
                    c = 0 # straight line
                else:
                    c = 2 * np.pi - h
            else:
                # heading angle > 2pi
                # check if turn 1 has good heading angle
                h_cw = 2 * np.pi - (h % (2 * np.pi))
                if a >= h_cw:
                    c = h_cw + 2 * np.pi - a
                else:
                    c = 2 * np.pi - h - a

            #ensure endpoint is going to correct point
            newEnd = self._getEndpointBSB(a, b, c, -1, 1, -1)
            if self._compareVector(newEnd, [x, y, h]) == 0:
                r = a, b, c, a + b + c, DubinsPathType.RSR
    
        # no correct points return default
        return r
    
    def _solveRLR(self, x, y, h):
        possibleB = []
        v = (x + np.sin(h)) / 2
        w = (-y - 1  + np.cos(h)) / 2
        possibleB.append(np.arccos(1 - (v ** 2 + w ** 2) / 2))
        r = DEFAULT_DUBINS

        # if b is nan can't continue calc
        if np.isnan(possibleB[0]):
            return r
        
        possibleB[0] = possibleB[0] if possibleB[0] >= 0 else -possibleB[0]
        possibleB.append(2 * np.pi - possibleB[0])
        for b in possibleB:
            A = (v ** 2 - w ** 2) / (2 * (1 - np.cos(b)))
            B = v * w / (1 - np.cos(b))
            a = .5 * np.arctan2(B * np.cos(b) + A * np.sin(b), A * np.cos(b) - B * np.sin(b))
            
            # a is nan skip current b value
            if np.isnan(a):
                continue

            while a < 0:
                a += np.pi / 2
            
            # 4 possible a values
            possibleA = [
                a % (2 * np.pi),
                (a + np.pi / 2) % (2 * np.pi),
                (a + np.pi) % (2 * np.pi),
                (a + 3 * np.pi / 2) % (2 * np.pi)
            ]
            for a in possibleA:
                c = (b - a - h) % (2 * np.pi)
                # check for valid point
                newEnd = self._getEndpointBBB(a, b, c, -1, 1, -1)
                # check for valid a b and c values
                if self._compareVector(newEnd, [x, y, h]) == 0 and max(a, c) < b and min(a, c) < b + np.pi:
                    r = a, b, c, a + b + c, DubinsPathType.RLR

        return r

    
    def _solveLRL(self, x, y, h):
        possibleB = []
        v = (x - np.sin(h)) / 2
        w = (y -1  + np.cos(h)) / 2
        possibleB.append(np.arccos(1 - (v ** 2 + w ** 2) / 2))
        r = DEFAULT_DUBINS

        # if b is nan can't continue calc
        if np.isnan(possibleB[0]):
            return r
         
        possibleB[0] = possibleB[0] if possibleB[0] >= 0 else -possibleB[0]
        possibleB.append(2 * np.pi - possibleB[0])
        for b in possibleB:
            A = (v ** 2 - w ** 2) / (2 * (1 - np.cos(b)))
            B = v * w / (1 - np.cos(b))
            a = .5 * np.arctan2(B * np.cos(b) + A * np.sin(b), A * np.cos(b) - B * np.sin(b))
            
            # a is nan skip current b value
            if np.isnan(a):
                continue

            while a < 0:
                a += np.pi / 2
            
            # 4 possible a values
            possibleA = [
                a % (2 * np.pi),
                (a + np.pi / 2) % (2 * np.pi),
                (a + np.pi) % (2 * np.pi),
                (a + 3 * np.pi / 2) % (2 * np.pi)
            ]
            for a in possibleA:
                c = (b - a + h) % (2 * np.pi)
                # check for valid point
                newEnd = self._getEndpointBBB(a, b, c, 1, -1, 1)
                # check for valid a b and c values
                if self._compareVector(newEnd, [x, y, h]) == 0 and max(a, c) < b and min(a, c) < b + np.pi:
                    r = a, b, c, a + b + c, DubinsPathType.LRL
        # no solution found
        return r

    def _compareVector(self, a: 'list[float]', b: 'list[float]') -> int:
        result = np.linalg.norm(np.array(a) - np.array(b))
        if result < PATH_COMPARE_TOLERANCE:
            return 0
        return result
    
    def _getEndpointBSB(self, a, b, c, ki, km, kf):
        a *= ki
        c *= kf
        v0 = ki * np.sin(a) + b * np.cos(a) + kf * (np.sin(a + c) - np.sin(a))
        v1 = ki*(-np.cos(a) + 1) + b * np.sin(a) + kf * (-np.cos( a + c) + np.cos(a))
        v2 = (a + c) % (2 * np.pi)
        return [v0, v1, v2]
    
    def _getEndpointBBB(self, a, b, c, ki, km, kf):
        a *= ki
        b *= km
        c *= kf
        v0 =  2 * ki * np.sin(a) - 2 * ki * np.sin(a + b) + ki * np.sin(a + b + c)
        v1 = ki - 2 * ki* np.cos(a)+ 2*ki * np.cos(a + b) - ki * np.cos(a + b + c)
        v2 = (a + b + c) % (2 * np.pi)
        return [v0, v1, v2]