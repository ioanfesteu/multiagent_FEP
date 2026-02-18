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

    # ── Figure 1: World Map ──────────────────────────────────────────────────
    # With aspect='equal', matplotlib auto-sizes the figure height so the
    # 80x40 grid is rendered proportionally (2:1). We set a fixed width of 12".
    fig_map, ax_map = plt.subplots(1, 1, figsize=(12, 6))
    fig_map.suptitle(f"Agent {agent_id} — World Map Trajectory", fontsize=16, fontweight='bold')

    im = ax_map.imshow(temp_field.T, origin='lower', cmap='plasma', alpha=0.6,
                       extent=[0, GRID_WIDTH, 0, GRID_HEIGHT], aspect='equal')
    ax_map.plot(df_agent['X'], df_agent['Y'], color='white', linewidth=1.5, alpha=0.9, label='Path')
    ax_map.scatter(df_agent['X'].iloc[0], df_agent['Y'].iloc[0],
                   color='green', s=80, label='Start', zorder=5)
    ax_map.scatter(df_agent['X'].iloc[-1], df_agent['Y'].iloc[-1],
                   color='red', s=80, label='End/Death', zorder=5)
    ax_map.set_xlabel("X coordinate")
    ax_map.set_ylabel("Y coordinate")
    ax_map.legend()
    fig_map.colorbar(im, ax=ax_map, label='Temperature (°C)', fraction=0.023, pad=0.04)
    fig_map.tight_layout()

    # ── Figure 2: Parameter Charts ───────────────────────────────────────────
    fig_charts, (ax2, ax3) = plt.subplots(2, 1, figsize=(12, 8))
    fig_charts.suptitle(f"Agent {agent_id} — Parameter Evolution", fontsize=16, fontweight='bold')

    # Energy and Temperature
    ax2.plot(steps, df_agent['Energy'], color='blue', linewidth=2, label='Energy')
    ax2.plot(steps, df_agent['Temp'], color='orange', linewidth=2, label='Agent Temperature')
    ax2.set_title("Physiological States: Energy and internal Temperature", fontsize=13)
    ax2.set_xlabel("Step")
    ax2.set_ylabel("Levels")
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.6)

    # Valence (relative to 0) — primary mood indicator
    # Beta (Precision) is derived from Valence via exp(σ·V) and modulates Softmax internally
    ax3.plot(steps, df_agent['Valence'], color='green', linewidth=2, label='Valence (Mood)')
    ax3.axhline(0, color='black', linewidth=1, linestyle='-', alpha=0.5)
    ax3.set_title("Cognitive State: Integrated Valence (Mood)", fontsize=13)
    ax3.set_xlabel("Step")
    ax3.set_ylabel("Valence Value")
    ax3.legend()
    ax3.grid(True, linestyle='--', alpha=0.6)

    fig_charts.tight_layout(rect=[0, 0, 1, 0.96])

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
