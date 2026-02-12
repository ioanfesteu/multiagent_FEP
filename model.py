# model.py
import numpy as np
from mesa import Model
from mesa.space import MultiGrid
from mesa.agent import AgentSet
from agents import (
    AllostaticAgent, 
    GRID_WIDTH, GRID_HEIGHT, NUM_AGENTS, SEED,
    NUM_FOOD_PATCHES, FOOD_PATCH_AMOUNT_MIN, FOOD_PATCH_AMOUNT_MAX,
    SCENT_DECAY, MEMORY_DECAY
)

# ==========================================
# Environment Fields
# ==========================================

def generate_temperature_field(width, height):
    field = np.zeros((width, height))
    for x in range(width):
        for y in range(height):
            # Warm zones (Global Plateau)
            field[x, y] += 28 * np.exp(-((x - width/2)**2 + (y - height/2)**2) / (width*7.5))
            # Local optima (Hot spots)
            field[x, y] += 14 * np.exp(-((x - width*0.2)**2 + (y - height*0.8)**2) / 70)
            field[x, y] += 12 * np.exp(-((x - width*0.75)**2 + (y - height*0.25)**2) / 60)
    return field

def generate_food_field(width, height, n_patches):
    field = np.zeros((width, height))
    for _ in range(n_patches):
        cx, cy = np.random.randint(5, width-5), np.random.randint(5, height-5)
        amp = np.random.uniform(FOOD_PATCH_AMOUNT_MIN, FOOD_PATCH_AMOUNT_MAX) 
        sigma = np.random.uniform(2.0, 4.0) 
        
        for x in range(width):
            for y in range(height):
                dist = (x-cx)**2 + (y-cy)**2
                if dist < 30: 
                    field[x, y] += amp * np.exp(-dist / (2*sigma**2))
    return field

# ==========================================
# Model (OPTIMIZED)
# ==========================================

class DualDriveModel(Model):
    def __init__(self, width=GRID_WIDTH, height=GRID_HEIGHT, num_agents=NUM_AGENTS, seed=SEED):
        super().__init__(seed=seed)
        self.grid = MultiGrid(width, height, torus=False)
        self.agent_set = AgentSet([], self)
        
        # Fields
        self.temperature = generate_temperature_field(width, height)
        self.food = generate_food_field(width, height, n_patches=NUM_FOOD_PATCHES)
        
        # Global Scent
        self.food_scent = np.zeros((width, height)) 
        
        # Global Navigation Memory (Shared Stigmergy)
        self.shared_memory = np.zeros((width, height))
        
        # ********************************************
        # Internal params needed by agents
        # Explained in detail in HOWTO.md
        self.eta = 0.15        
        self.mu_affect = 0.1
        self.sigma = 1.0
        # ********************************************
        
        self.directions = [(-1,0),(1,0),(0,-1),(0,1),(1,1),(-1,1),(1,-1),(-1,-1)]

        # ✅ FIX: Statistics for dead agents
        self.dead_count = 0

        # Spawn Agents
        for i in range(num_agents):
            agent = AllostaticAgent(self)
            rx = self.random.randint(0, width-1)
            ry = self.random.randint(0, height-1)
            self.grid.place_agent(agent, (rx, ry))
            self.agent_set.add(agent)

    def step(self):
        """✅ FIX: Optimized for dead agent cleanup and NumPy operations"""
        # 1. Agents step
        agents = list(self.agent_set)
        self.random.shuffle(agents)  # ✅ FIX: Using self.random (no longer need random_gen)
        
        dead_agents = []
        for agent in agents:
            # Save state before
            was_alive = agent.is_alive
            
            # Execute step
            agent.step()
            
            # ✅ FIX: Death detection in this step
            if was_alive and not agent.is_alive:
                dead_agents.append(agent)
        
        # ✅ FIX: Cleanup dead agents from grid and agent_set
        for agent in dead_agents:
            self.grid.remove_agent(agent)
            self.agent_set.remove(agent)
            self.dead_count += 1
            
        # 2. Global Environment Decay
        # ✅ FIX: Optimized to reduce NumPy temporaries on Windows
        np.multiply(self.food_scent, SCENT_DECAY, out=self.food_scent)
        np.putmask(self.food_scent, self.food_scent < 0.05, 0)
        
        # Decay Shared Memory
        np.multiply(self.shared_memory, MEMORY_DECAY, out=self.shared_memory)
        np.putmask(self.shared_memory, self.shared_memory < 0.05, 0)