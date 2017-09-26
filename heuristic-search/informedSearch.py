from pq import *
from search import *


class InformedNode(Node):
    """
    Added the goal state as a parameter to the constructor.  Also
    added a new method to be used in conjunction with a priority
    queue.
    """

    def __init__(self, goal, state, parent, operator, depth):
        Node.__init__(self, state, parent, operator, depth)
        self.goal = goal

    def priority(self):
        """
        Needed to determine where the node should be placed in the
        priority queue.  Depends on the current depth of the node as
        well as the estimate of the distance from the current state to
        the goal state.
        """
        return self.depth + self.state.heuristic(self.goal)


class InformedSearch(Search):
    """
    A general informed search class that uses a priority queue and
    traverses a search tree containing instances of the InformedNode
    class.  The problem domain should be based on the
    InformedProblemState class.
    """

    def __init__(self, initialState, goalState):
        self.expansions = 0
        self.clearVisitedStates()
        self.q = PriorityQueue()
        self.goalState = goalState
        self.q.enqueue(InformedNode(goalState, initialState, None, None, 0))
        solution = self.execute()
        if solution == None:
            print("Search failed")
        else:
            self.showPath(solution)
            print("Expanded", self.expansions, "nodes during search")

    def execute(self):
        while not self.q.empty():
            current = self.q.dequeue()
            self.expansions += 1
            if self.goalState.equals(current.state):
                return current
            else:
                successors = current.state.applyOperators()
                operators = current.state.operatorNames()
                for i in range(len(successors)):
                    if not successors[i].illegal():
                        n = InformedNode(self.goalState,
                                         successors[i],
                                         current,
                                         operators[i],
                                         current.depth + 1)
                        if n.repeatedState():
                            del (n)
                        else:
                            self.q.enqueue(n)
        return None


class InformedProblemState(ProblemState):
    """
    An interface class for problem domains used with informed search.
    """

    def heuristic(self, goal):
        """
        For use with informed search.  Returns the estimated
        cost of reaching the goal from this state.
        """
        abstract()