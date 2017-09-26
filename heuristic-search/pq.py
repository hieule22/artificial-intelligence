class PriorityQueue:
    """
    Implements a heap-style priority queue with O(lg n) enqueue and
    dequeue methods.  The priority queue is stored as a list where
    position 0 always contains None.  The first actual item is stored
    in position 1.  This is necessary so that the list can be treated
    as a binary tree and simple calculations can be done to find the
    parent, and left and right sub-trees. The items being stored are
    expected to be instances of a class which has a priority() method.
    """
    def __init__(self):
        self.q = [None]
    def __str__(self):
        result = "Queue contains " + str(len(self.q)-1) + " items"
        if not self.empty():
            result += "-Minimum item has priority: " + \
                      str(self.min().priority())
        return result
    def parent(self, i):
        return i//2
    def right(self, i):
        return (i * 2) + 1
    def left(self, i):
        return i * 2
    def hasLeft(self, i):
        return self.left(i) <= len(self.q)-1
    def hasRight(self, i):
        return self.right(i) <= len(self.q)-1
    def empty(self):
        return len(self.q) == 1
    def swap(self, p1, p2):
        self.q[p1], self.q[p2], = self.q[p2], self.q[p1]
    def bubbleUp(self,i):
        p = self.parent(i)
        if i==1 or self.q[i].priority() >= self.q[p].priority():
            return
        else:
            self.swap(i, p)
            self.bubbleUp(p)
    def bubbleDown(self, i):
        if (not self.hasLeft(i)) and (not self.hasRight(i)):
            return
        elif self.hasLeft(i) and (not self.hasRight(i)):
            l = self.left(i)
            if self.q[i].priority() > self.q[l].priority():
                self.swap(i, l)
                self.bubbleDown(l)
        else:
            l = self.left(i)
            r = self.right(i)
            key = self.q[i].priority()
            if self.q[l].priority() >= key and self.q[r].priority() >= key:
                return
            elif self.q[l].priority() <= self.q[r].priority():
                self.swap(i, l)
                self.bubbleDown(l)
            else:
                self.swap(i, r)
                self.bubbleDown(r)
    def min(self):
        if self.empty():
            raise RunTimeError
        return self.q[1]
    def dequeue(self):
        if self.empty():
            raise RunTimeError
        result = self.q.pop(1)
        self.q.insert(1, self.q.pop(len(self.q)-1))
        self.bubbleDown(1)
        return result
    def enqueue(self, item):
        self.q.append(item)
        self.bubbleUp(len(self.q)-1)

if __name__ == '__main__':
    class Test:
        """
        A simple class created to test the priority queue.
        The PriorityQueue class expects to store instances
        of a class that has a method called priority().
        """
        def __init__(self, v):
            self.value = v
        def __str__(self):
            return str(self.value)
        def priority(self):
            return self.value

    print("Creating a PriorityQueue")
    pq = PriorityQueue()
    print("Check that an empty queue is printable")
    print(pq)
    print("Inserting 10, 5, 2, 12, 25")
    pq.enqueue(Test(10))
    pq.enqueue(Test(5))
    pq.enqueue(Test(2))
    pq.enqueue(Test(12))
    pq.enqueue(Test(25))
    print(pq)
    print("Removing the minimum until empty")
    while (not pq.empty()):
        print(pq.dequeue())
