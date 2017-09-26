### File: search.py
### This file includes four class definitions: Queue, Node,
### Search, ProblemState

# from exceptions import *

# This is a global variable that is used to avoid revisiting repeated
# states.  It needs to be reset to an empty dictionary each time
# a search is run.
VisitedStates = {}


class Queue:
    """
    A Queue class to be used in combination with state space
    search. The enqueue method adds new elements to the end. The
    dequeue method removes elements from the front.
    """

    def __init__(self):
        self.queue = []

    def __str__(self):
        result = "Queue contains " + str(len(self.queue)) + " items\n"
        for item in self.queue:
            result += str(item) + "\n"
        return result

    def enqueue(self, node):
        self.queue.append(node)

    def dequeue(self):
        if not self.empty():
            return self.queue.pop(0)
        else:
            raise RunTimeError

    def empty(self):
        return len(self.queue) == 0


class Node:
    """
    A Node class to be used in combination with state space search.
    A node contains a state, a parent node, the name of the operator
    used to reach the current state, and the depth of the node in
    the search tree.  The root node should be at depth 0. The method
    repeatedState can be used to determine if the current state
    is the same as the parent's parent's state. Eliminating such
    repeated states improves search efficiency.
    """

    def __init__(self, state, parent, operator, depth):
        self.state = state
        self.parent = parent
        self.operator = operator
        self.depth = depth

    def __str__(self):
        result = "State: " + str(self.state)
        result += " Depth: " + str(self.depth)
        if self.parent != None:
            result += " Parent: " + str(self.parent.state)
            result += " Operator: " + self.operator
        return result

    def repeatedState(self):
        global VisitedStates
        if str(self.state) in VisitedStates:
            return 1
        else:
            VisitedStates[str(self.state)] = True
            return 0
            # if self.parent == None: return 0
            # if self.parent.state.equals(self.state): return 1
            # if self.parent.parent == None: return 0
            # if self.parent.parent.state.equals(self.state): return 1
            # return 0


class Search:
    """
    A general Search class that can be used for any problem domain.
    Given instances of an initial state and a goal state in the
    problem domain, this class will print the solution or a failure
    message.  The problem domain should be based on the ProblemState
    class.
    """

    def __init__(self, initialState, goalState):
        self.clearVisitedStates()
        self.q = Queue()
        self.q.enqueue(Node(initialState, None, None, 0))
        self.goalState = goalState
        solution = self.execute()
        if solution == None:
            print("Search failed")
        else:
            self.showPath(solution)

    def clearVisitedStates(self):
        global VisitedStates
        VisitedStates = {}

    def execute(self):
        while not self.q.empty():
            current = self.q.dequeue()
            if self.goalState.equals(current.state):
                return current
            else:
                successors = current.state.applyOperators()
                operators = current.state.operatorNames()
                for i in range(len(successors)):
                    if not successors[i].illegal():
                        n = Node(successors[i],
                                 current,
                                 operators[i],
                                 current.depth + 1)
                        if n.repeatedState():
                            del (n)
                        else:
                            self.q.enqueue(n)
                            # Uncomment the line below to see the queue.
                            # print "Enqueuing state: " + str(n)
        return None

    def showPath(self, node):
        path = self.buildPath(node)
        for current in path:
            if current.depth != 0:
                print("Operator:", current.operator)
            print(current.state)
        print("Goal reached in", current.depth, "steps")

    def buildPath(self, node):
        """
        Beginning at the goal node, follow the parent links back
        to the start state.  Create a list of the states traveled
        through during the search from start to finish.
        """
        result = []
        while node != None:
            result.insert(0, node)
            node = node.parent
        return result


class ProblemState:
    """
    An interface class for problem domains.
    """

    def illegal(self):
        """
        Tests the state instance for validity.
        Returns true or false.
        """
        abstract()

    def applyOperators(self):
        """
        Returns a list of successors to the current state,
        some of which may be illegal.
        """
        abstract()

    def operatorNames(self):
        """
        Returns a list of operator names in the same order
        as the successors list is generated.
        """
        abstract()

    def equals(self, state):
        """
        Tests whether the state instance equals the given
        state.
        """
        abstract()