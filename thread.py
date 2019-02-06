class Thread:
    def __init__(self, x, y, g, cost):
        self.x = x
        self.y = y
        self.g = g
        self.cost = cost

class Indiv:
	def __init__(self):
		self.genes = []
		self.fitness = 0