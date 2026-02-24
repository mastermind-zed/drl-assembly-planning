class Robot:
    def __init__(self, id, x, y, z):
        self.id = id
        self.x = x
        self.y = y
        self.z = z
        self.battery = 100.0
        self.load_capacity = 1.0
        self.carried_sco = None
        self.status = 'idle' # idle, moving, carrying, assembling
        self.path = []

    def move(self, dx, dy, dz):
        self.x += dx
        self.y += dy
        self.z += dz
        self.battery -= 0.1 # Energy consumption per move

    def pick_up(self, sco):
        if self.carried_sco is None:
            self.carried_sco = sco
            self.status = 'carrying'
            return True
        return False

    def drop_off(self):
        if self.carried_sco:
            sco = self.carried_sco
            self.carried_sco = None
            self.status = 'idle'
            return sco
        return None
