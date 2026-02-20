import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
import os

# Import constants from agents.py
try:
    from agents import (
        GRID_WIDTH, GRID_HEIGHT,
        TEMP_BASE_MAX, TEMP_SPOT_1, TEMP_SPOT_2
    )
except ImportError:
    # Fallback if agents.py is not available
    GRID_WIDTH = 80
    GRID_HEIGHT = 40
    TEMP_BASE_MAX = 28.0
    TEMP_SPOT_1 = 14.0
    TEMP_SPOT_2 = 12.0

def generate_temperature_field(width, height):
    field = np.zeros((width, height))
    for x in range(width):
        for y in range(height):
            # Warm zones (Global Plateau)
            field[x, y] += TEMP_BASE_MAX * np.exp(-((x - width/2)**2 + (y - height/2)**2) / (width*7.5))
            # Local optima (Hot spots)
            field[x, y] += TEMP_SPOT_1 * np.exp(-((x - width*0.2)**2 + (y - height*0.8)**2) / 70)
            field[x, y] += TEMP_SPOT_2 * np.exp(-((x - width*0.75)**2 + (y - height*0.25)**2) / 60)
    return field

def create_report(agent_id, df_agent, temp_field):
    steps = df_agent['Step']

    # ── Single Figure Setup ──────────────────────────────────────────────────
    # We calculate a figure height that accommodates the map (keeping aspect ratio)
    # plus the two charts below it.
    # Map aspect ratio = Height / Width.
    map_aspect = GRID_HEIGHT / GRID_WIDTH
    fig_width = 8
    
    # Calculate heights to ensure map fills the width
    map_height_inches = fig_width * map_aspect
    chart_height_inches = 3.0
    
    # Total figure height including charts and some padding
    fig_height = map_height_inches + (2 * chart_height_inches) + 1
    
    fig = plt.figure(figsize=(fig_width, fig_height), constrained_layout=True)
    fig.suptitle(f"Agent {agent_id} — Integrated Report", fontsize=16, fontweight='bold')
    
    # Create a grid: 3 rows, 1 column. 
    # Adjust height ratios so the map slot matches its aspect ratio requirements
    gs = fig.add_gridspec(3, 1, height_ratios=[map_height_inches, chart_height_inches, chart_height_inches])

    # ── Subplot 1: World Map ─────────────────────────────────────────────────
    ax_map = fig.add_subplot(gs[0])
    im = ax_map.imshow(temp_field.T, origin='lower', cmap='plasma', alpha=0.6,
                       extent=[0, GRID_WIDTH, 0, GRID_HEIGHT], aspect='equal')
    ax_map.plot(df_agent['X'], df_agent['Y'], color='white', linewidth=1.5, alpha=0.9, label='Path')
    ax_map.scatter(df_agent['X'].iloc[0], df_agent['Y'].iloc[0],
                   color='green', s=80, label='Start', zorder=5)
    ax_map.scatter(df_agent['X'].iloc[-1], df_agent['Y'].iloc[-1],
                   color='red', s=80, label='End/Death', zorder=5)
    ax_map.set_title("World Map Trajectory (Temperature Field)", fontsize=12)
    ax_map.set_xlabel("X coordinate")
    ax_map.set_ylabel("Y coordinate")
    ax_map.legend(loc='upper right')
    fig.colorbar(im, ax=ax_map, label='Temperature (°C)', fraction=0.02, pad=0.04)

    # ── Subplot 2: Physiological States ──────────────────────────────────────
    ax2 = fig.add_subplot(gs[1])
    ax2.plot(steps, df_agent['Energy'], color='blue', linewidth=2, label='Energy')
    ax2.plot(steps, df_agent['Temp'], color='orange', linewidth=2, label='Agent Temperature')
    ax2.set_title("Physiological States: Energy and internal Temperature", fontsize=13)
    ax2.set_ylabel("Levels")
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.6)

    # ── Subplot 3: Cognitive State ───────────────────────────────────────────
    ax3 = fig.add_subplot(gs[2], sharex=ax2)
    ax3.plot(steps, df_agent['Valence'], color='green', linewidth=2, label='Valence (Mood)')
    ax3.axhline(0, color='black', linewidth=1, linestyle='-', alpha=0.5)
    ax3.set_title("Cognitive State: Integrated Valence (Mood)", fontsize=13)
    ax3.set_xlabel("Step")
    ax3.set_ylabel("Valence Value")
    ax3.legend()
    ax3.grid(True, linestyle='--', alpha=0.6)

def main():
    csv_file = 'fep_swarm_agent_data.csv'
    if not os.path.exists(csv_file):
        print(f"Error: {csv_file} not found.")
        return

    print(f"Loading data from {csv_file}...")
    df = pd.read_csv(csv_file)
    
    # Get unique agent IDs that have data
    unique_agents = df['AgentID'].unique()
    print(f"Found data for agents: {unique_agents}")

    # Generate temperature field
    temp_field = generate_temperature_field(GRID_WIDTH, GRID_HEIGHT)

    # Check command line arguments for specific agent IDs
    if len(sys.argv) > 1:
        target_ids = []
        for arg in sys.argv[1:]:
            try:
                target_ids.append(int(arg))
            except ValueError:
                print(f"Skipping invalid ID: {arg}")
    else:
        # Default: show the first 2 agents if no specific ID provided
        target_ids = unique_agents[:2]
        print(f"No AgentID provided. Showing reports for the first {len(target_ids)} agents.")

    for agent_id in target_ids:
        if agent_id not in unique_agents:
            print(f"Agent {agent_id} not found in data.")
            continue
        
        print(f"Generating report for Agent {agent_id}...")
        df_agent = df[df['AgentID'] == agent_id].sort_values('Step')
        
        # Determine the "lifetime" of the agent
        # Some biological agents might "drop out" before the end of the simulation
        # The filter df['AgentID'] == agent_id already captures only the steps where they were present.
        
        create_report(agent_id, df_agent, temp_field)

    print("Showing plots. Close the windows to exit.")
    plt.show()

if __name__ == "__main__":
    main()
