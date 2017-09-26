### File pitcher.py
### Implements the water pitcher puzzle for state space search

from search import *

class PitcherState(ProblemState):
    """
    The water pitcher puzzle: Suppose that you are given a 3 quart
    pitcher and a 4 quart pitcher.  Either pitcher can be filled
    from a faucet.  The contents of either pitcher can be poured
    down a drain.  Water may be poured from one pitcher to the other.
    When pouring, as soon as the pitcher being poured into is full,
    the pouring stops.  There is no additional measuring device and
    and the pitchers have no markings to show partial quantities.

    Each operator returns a new instance of this class representing
    the successor state.  
    """
    def __init__(self, q3, q4):
        self.q3 = q3
        self.q4 = q4
    def __str__(self):
        """
        Required method for use with the Search class.
        Returns a string representation of the state.
        """
        return "("+str(self.q3)+","+str(self.q4)+")"
    def illegal(self):
        """
        Required method for use with the Search class.
        Tests whether the state is illegal.
        """
        if self.q3 < 0 or self.q4 < 0: return 1
        if self.q3 > 3 or self.q4 > 4: return 1
        return 0
    def equals(self, state):
        """
        Required method for use with the Search class.
        Determines whether the state instance and the given
        state are equal.
        """
        return self.q3==state.q3 and self.q4==state.q4
    def fillq3(self):
        return PitcherState(3, self.q4)
    def fillq4(self):
        return PitcherState(self.q3, 4)
    def drainq3(self):
        return PitcherState(0, self.q4)
    def drainq4(self):
        return PitcherState(self.q3, 0)
    def pourq3Toq4(self):
        capacity = 4 - self.q4
        if self.q3 > capacity:
            return PitcherState(self.q3-capacity, 4)
        else:
            return PitcherState(0, self.q4 + self.q3)
    def pourq4Toq3(self):
        capacity = 3 - self.q3
        if self.q4 > capacity:
            return PitcherState(3, self.q4-capacity)
        else:
            return PitcherState(self.q3 + self.q4, 0)
    def operatorNames(self):
        """
        Required method for use with the Search class.
        Returns a list of the operator names in the
        same order as the applyOperators method.
        """
        return ["fillq3", "fillq4", "drainq3", "drainq4",
                "pourq3Toq4", "pourq4Toq3"]
    def applyOperators(self):
        """
        Required method for use with the Search class.
        Returns a list of possible successors to the current
        state, some of which may be illegal.  
        """
        return [self.fillq3(), self.fillq4(),
                self.drainq3(), self.drainq4(),
                self.pourq3Toq4(), self.pourq4Toq3()]

Search(PitcherState(0,0), PitcherState(0,2))

    
