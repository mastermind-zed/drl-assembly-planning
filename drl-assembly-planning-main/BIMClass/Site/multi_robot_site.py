import copy
from BIMClass.Site.SCO import *
from BIMClass.Site.Robot import Robot
import random

class MultiRobotSite:
    def __init__(self, width, length, height, num_robots=2):
        self.s_wid = width
        self.s_len = length
        self.s_he = height
        
        # Construction area (centralized)
        self.construction_x = range(6, 13)
        self.construction_y = range(8, 15)
        
        # Site 3D representation
        # 0: empty, 10: ground, 50: construction area
        # 1-9: robots, 100+: placed components
        self.site_3D = [[[0 for _ in range(self.s_len)] for _ in range(self.s_wid)] for _ in range(self.s_he + 1)]
        
        # Initialize ground
        for i in range(self.s_wid):
            for j in range(self.s_len):
                self.site_3D[0][i][j] = 10
        
        # Initialize construction area
        for i in self.construction_x:
            for j in self.construction_y:
                self.site_3D[0][i][j] = 50
        
        self.robots = []
        for i in range(num_robots):
            # Place robots near edge
            self.robots.append(Robot(i + 1, random.randint(0, 3), random.randint(0, 3), 1))
            
        self.scos = [] # All required components
        self.arrived_scos = []
        self.init_icf_components()
        
    def init_icf_components(self):
        # Example ICF layout: a 2x2 square
        # Target positions for ICF blocks (x1, y1, z1, x2, y2, z2)
        # Assuming 2x1x1 ICF blocks
        targets = [
            [8, 10, 1, 9, 10, 1],
            [8, 11, 1, 9, 11, 1],
            [8, 10, 2, 9, 10, 2],
            [8, 11, 2, 9, 11, 2]
        ]
        for i, t in enumerate(targets):
            sco = SCO(i, 'icf', 0, 0, 0, 0, 0, 0, t[0], t[1], t[2], t[3], t[4], t[5], 2, [], train_on=True)
            sco.x_1, sco.y_1, sco.z_1 = 0, i, 1 # Supply area
            sco.x_2, sco.y_2, sco.z_2 = 1, i, 1
            self.scos.append(sco)

    def step(self, robot_actions):
        """
        robot_actions: dict {robot_id: action_index}
        Actions: 0: FW, 1: BK, 2: L, 3: R, 4: UP, 5: DN, 6: PickUp, 7: Place
        """
        rewards = {r.id: 0 for r in self.robots}
        dones = {r.id: False for r in self.robots}
        
        for robot in self.robots:
            action = robot_actions.get(robot.id, None)
            if action is None: continue
            
            # Save old position
            old_pos = (robot.x, robot.y, robot.z)
            
            # Basic Movement
            if action == 0: robot.move(1, 0, 0)
            elif action == 1: robot.move(-1, 0, 0)
            elif action == 2: robot.move(0, -1, 0)
            elif action == 3: robot.move(0, 1, 0)
            elif action == 4: robot.move(0, 0, 1)
            elif action == 5: robot.move(0, 0, -1)
            
            # Small penalty per step to encourage efficiency
            rewards[robot.id] -= 0.01
            
            # Boundary & Collision Check
            if not self.is_valid_pos(robot.x, robot.y, robot.z):
                robot.x, robot.y, robot.z = old_pos
                rewards[robot.id] -= 0.5 # Penalty for hitting boundary/obs
                
            # Pick Up Action
            elif action == 6:
                if robot.carried_sco is None:
                    # Check for available SCOs at current position
                    for sco in self.scos:
                        if not sco.working and not sco.arrived:
                            if (sco.x_1 == robot.x and sco.y_1 == robot.y and sco.z_1 == robot.z):
                                robot.pick_up(sco)
                                sco.working = True
                                rewards[robot.id] += 1.0
                                break
            
            # Place Action
            elif action == 7:
                if robot.carried_sco:
                    sco = robot.carried_sco
                    # Check if at target
                    if (robot.x == sco.x_tar_1 and robot.y == sco.y_tar_1 and robot.z == sco.z_tar_1):
                        robot.drop_off()
                        sco.arrived = True
                        sco.working = False
                        self.arrived_scos.append(sco)
                        rewards[robot.id] += 5.0
                        # Update site_3D with placed component
                        self.site_3D[sco.z_tar_1][sco.x_tar_1][sco.y_tar_1] = 100
                        self.site_3D[sco.z_tar_2][sco.x_tar_2][sco.y_tar_2] = 100
                    else:
                        rewards[robot.id] -= 1.0 # Missed target
            
            # Update SCO position if carried
            if robot.carried_sco:
                sco = robot.carried_sco
                # Update relative to robot
                sco.x_1, sco.y_1, sco.z_1 = robot.x, robot.y, robot.z
                # Simplified 2-node update
                sco.x_2, sco.y_2, sco.z_2 = robot.x + 1, robot.y, robot.z # Fixed orientation for now
        
        return rewards, dones

    def is_valid_pos(self, x, y, z):
        if not (0 <= x < self.s_wid and 0 <= y < self.s_len and 0 <= z <= self.s_he):
            return False
        # Check collision with other robots or placed elements
        # (Simplified: just check site_3D for non-zero above ground)
        if z > 0 and self.site_3D[z][x][y] >= 100:
            return False
        return True

    def get_obs(self, robot_id):
        # Global observation for now (Centralized)
        # Flattened site_3D + robot states
        # In actual implementation, this will be handled by the Gym environment
        pass
