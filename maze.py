import sys
import heapq


class Node():
    def __init__(self, state, parent, action, cost=1):  # tk added cost field
        self.state = state
        self.parent = parent
        self.action = action
        self.cost = cost

    def __lt__(self, other): return id(self) < id(other)
    def __le__(self, other): return id(self) <= id(other)

class StackFrontier():
    def __init__(self):
        self.frontier = []

    def add(self, node):
        self.frontier.append(node)

    def contains_state(self, state):
        return any(node.state == state for node in self.frontier)

    def empty(self):
        return len(self.frontier) == 0

    def remove(self):
        if self.empty():
            raise Exception("empty frontier")
        else:
            node = self.frontier[-1]
            self.frontier = self.frontier[:-1]
            return node


class QueueFrontier(StackFrontier):  # Only need to implement remove

    def remove(self):

        if self.empty():
            raise Exception("empty frontier")
        else:
            node = self.frontier[0]
            self.frontier = self.frontier[1:]
            return node







class PriorityQueue:
    """
      Modified https://courses.cs.washington.edu/courses/csep573/14sp/pacman/tracking/docs/util.html
      Implements a priority queue data structure. Each inserted item
      has a priority associated with it and the client is usually interested
      in quick retrieval of the lowest-priority item in the queue. This
      data structure allows O(1) access to the lowest-priority item.

      Note that this PriorityQueue does not allow you to change the priority
      of an item.  However, you may insert the same item multiple times with
      different priorities.
    """

    def __init__(self):
        self.heap = []
        self.frontier = []

    def add(self, item, priority):
        self.frontier.append(item)
        pair = (priority, item)
        heapq.heappush(self.heap, pair)

    def remove(self):
        (priority, item) = heapq.heappop(self.heap)
        self.frontier.remove(item)
        return item

    def contains_state(self, state):
        return any(item.state == state for item in self.frontier)

    def empty(self):
        return len(self.heap) == 0



def manhattanDistance(xy1, xy2):
    "Returns the Manhattan distance between points xy1 and xy2"
    ''' Your Code goes here'''
    d_x = abs(xy1[0] - xy2[0])
    d_y = abs(xy1[1] - xy2[1])
    return d_x + d_y 


class Maze():

    def __init__(self, filename):

        # Read file and set height and width of maze
        with open(filename) as f:
            contents = f.read()

        # Validate start and goal
        if contents.count("A") != 1:
            raise Exception("maze must have exactly one start point")
        if contents.count("B") != 1:
            raise Exception("maze must have exactly one goal")

        # Determine height and width of maze
        contents = contents.splitlines()
        self.height = len(contents)
        self.width = max(len(line) for line in contents)

        # Keep track of walls
        self.walls = []
        for i in range(self.height):
            row = []
            for j in range(self.width):
                try:
                    if contents[i][j] == "A":
                        self.start = (i, j)
                        row.append(False)
                    elif contents[i][j] == "B":
                        self.goal = (i, j)
                        row.append(False)
                    elif contents[i][j] == " ":
                        row.append(False)
                    else:
                        row.append(True)
                except IndexError:
                    row.append(False)
            self.walls.append(row)

        self.solution = None

    def print(self):
        solution = self.solution[1] if self.solution is not None else None
        print()
        for i, row in enumerate(self.walls):
            for j, col in enumerate(row):
                if col:
                    print("█", end="")
                elif (i, j) == self.start:
                    print("A", end="")
                elif (i, j) == self.goal:
                    print("B", end="")
                elif solution is not None and (i, j) in solution:
                    print("*", end="")
                else:
                    print(" ", end="")
            print()
        print()

    def neighbors(self, state):
        row, col = state
        candidates = [
            ("up", (row - 1, col)),
            ("down", (row + 1, col)),
            ("left", (row, col - 1)),
            ("right", (row, col + 1))
        ]

        result = []
        for action, (r, c) in candidates:
            if 0 <= r < self.height and 0 <= c < self.width and not self.walls[r][c]:
                result.append((action, (r, c)))
        return result


    def solve(self):
        """Finds a solution to maze, if one exists.
           Complete the following code where indicated"""

        # Keep track of number of states explored
        self.num_explored = 0

        # Initialize frontier to just the starting position, priority is not part of state
        start = Node(state=self.start, parent=None, action=None, cost=0)

        # Choose the appropriate data structure for the frontier
        # used for search from the three below

       # frontier = StackFrontier()
        # frontier = QueueFrontier()
        frontier = PriorityQueue()

        # slightly different code if using Priority Queue
        if  isinstance(frontier, PriorityQueue) :
            frontier.add(start, 100)  #  priority
        else:
            frontier.add(start)    # no priority


        # Initialize an empty explored set
        self.explored = set()

        # This is the loop that finds the path
        while True:

            # If nothing left in frontier, then no path
            if frontier.empty():
                raise Exception("no solution")

            # Choose a node from the frontier
            node = frontier.remove()
            print("nextnode is : ", node.state, node.cost)
            self.num_explored += 1

            # If node is the goal, then have a solution
            # and  must reconstruct the path
            if node.state == self.goal:
                actions = []
                cells = []

                ''' your code goes here '''
                


                actions.reverse()
                cells.reverse()
                self.solution = (actions, cells)  # solution is not local
                print("goal found")
                return

            # Mark node as explored
            self.explored.add(node.state)

            ''' your code goes here '''

            # Add neighbors to frontier
            for action, state in self.neighbors(node.state):
                    
                    if(state in self.explored):
                            continue
                    ''' your code goes here '''
                    if not isinstance(frontier, PriorityQueue) :
                        # No Priority
                        
                        child = Node(state=state, parent=node, action=action, cost=node.cost + 1)
                        frontier.add(child)                                                  # add back
                    else:
                        # Priority  - need to add the appropriate code to
                        # in case for using PriorityQueue
                        child = Node(state=state, parent=node, action=action, cost=node.cost + 1)

                        priority = child.cost + manhattanDistance(child.state, self.goal)              # use if PQ
                        frontier.add(child, priority)        # use if PQ

    def output_image(self, filename, show_solution=True, show_explored=False):
        from PIL import Image, ImageDraw
        cell_size = 50
        cell_border = 2

        # Create a blank canvas
        img = Image.new(
            "RGBA",
            (self.width * cell_size, self.height * cell_size),
            "black"
        )
        draw = ImageDraw.Draw(img)

        solution = self.solution[1] if self.solution is not None else None
        for i, row in enumerate(self.walls):
            for j, col in enumerate(row):

                # Walls
                if col:
                    fill = (40, 40, 40)

                # Start
                elif (i, j) == self.start:
                    fill = (255, 0, 0)

                # Goal
                elif (i, j) == self.goal:
                    fill = (0, 171, 28)

                # Solution
                elif solution is not None and show_solution and (i, j) in solution:
                    fill = (220, 235, 113)

                # Explored
                elif solution is not None and show_explored and (i, j) in self.explored:
                    fill = (212, 97, 85)

                # Empty cell
                else:
                    fill = (237, 240, 252)

                # Draw cell
                draw.rectangle(
                    ([(j * cell_size + cell_border, i * cell_size + cell_border),
                      ((j + 1) * cell_size - cell_border, (i + 1) * cell_size - cell_border)]),
                    fill=fill
                )

        img.save(filename)


if len(sys.argv) != 2:
    sys.exit("Usage: python maze.py maze.txt")

m = Maze(sys.argv[1])
print("Maze:")
m.print()
print("Solving...")
m.solve()
print("States Explored:", m.num_explored)
print("Solution:")
m.print()
m.output_image("maze.png", show_explored=True)
