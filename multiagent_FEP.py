# multiagent_FEP.py
# Solms-Friston Swarm: Temperature + Energy + Social Food Trails
# Windows 10 Optimized Version - Memory Leak Fixes Applied
# Date: 2026-02-11
# Improvements: Figure caching, AsyncIO tracking, Dead agent cleanup, NumPy optimization

# ==========================================
# WINDOWS OPTIMIZATIONS
# ==========================================
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for Windows

import gc
gc.set_threshold(700, 10, 10)  # More aggressive GC

# NumPy thread limitation (optional, uncomment if you have issues)
# import os
# os.environ['OMP_NUM_THREADS'] = '1'
# os.environ['OPENBLAS_NUM_THREADS'] = '1'
# os.environ['MKL_NUM_THREADS'] = '1'

# ==========================================
# IMPORTS
# ==========================================
import numpy as np
import os
import matplotlib.pyplot as plt
import solara
import asyncio

# Modular imports
from model import DualDriveModel
from agents import (
    NUM_AGENTS, COLOR_OK, COLOR_HUNGRY, COLOR_COLD, COLOR_HOT, 
    COLOR_FOOD, COLOR_TRAIL, WEIGHT_TEMP, WEIGHT_ENERGY
)

# ==========================================
# Visualization (OPTIMIZED + OVERLAY)
# ==========================================

def get_plot_figure(model, step_number=0):
    """
    Version with Stats Overlay
    """
    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(111)
    
    # 1. Heatmap (Temperature)
    ax.imshow(model.temperature.T, origin='lower', cmap='coolwarm', alpha=0.4, vmin=0, vmax=40)
    
    # 2. Food patches
    fx, fy, fs = [], [], []
    for x in range(model.grid.width):
        for y in range(model.grid.height):
            val = model.food[x, y]
            if val > 1.0:
                fx.append(x)
                fy.append(y)
                fs.append(min(val * 3, 150)) 
    if fx:
        ax.scatter(fx, fy, c=COLOR_FOOD, s=fs, alpha=0.6, edgecolors='green', label='Food Source')

    # 3. Social Scent Trails
    sx, sy, ss = [], [], []
    for x in range(model.grid.width):
        for y in range(model.grid.height):
            val = model.food_scent[x, y]
            if val > 0.1:
                sx.append(x)
                sy.append(y)
                ss.append(min(val * 20, 50))
    if sx:
        ax.scatter(sx, sy, c=COLOR_TRAIL, s=ss, alpha=0.8, marker='.', label='Food Trail')

    # 4. Agents
    # Filter alive agents for accurate stats in overlay
    alive_agents = [a for a in model.agent_set if a.is_alive]
    n_alive = len(alive_agents)

    for agent in model.agent_set:
        
        # A. Visits (Black dots)
        visits_x, visits_y = [], []
        for (vx, vy), val in agent.visits.items():
            if val > 0.1:
                visits_x.append(vx)
                visits_y.append(vy)
        if visits_x:
            ax.scatter(visits_x, visits_y, c='black', s=5, alpha=0.1)

        # B. Body & Color Logic
        x, y = agent.pos
        z = 10
        marker = 'o'
        
        # --- Logic for Red/Blue/Brown ---
        diff_T = agent.T_int - agent.T_pref 
        err_T_weighted = abs(diff_T) * WEIGHT_TEMP
        err_E_weighted = max(0, agent.E_crit - agent.E_int) * WEIGHT_ENERGY
        
        # Determine dominant drive
        if err_E_weighted > err_T_weighted and err_E_weighted > 1.0:
            c = COLOR_HUNGRY
        elif err_T_weighted > err_E_weighted and err_T_weighted > 1.0:
            # Hot or Cold?
            if diff_T > 0:
                c = COLOR_HOT # Too hot
            else:
                c = COLOR_COLD # Too cold
        else:
            c = COLOR_OK
        
        ax.scatter(x, y, c=c, s=120, marker=marker, edgecolors='black', linewidth=1.5, zorder=z)

    # 5. Stats Overlay (Integrated from overlay.py)
    if n_alive > 0:
        avg_E = sum(a.E_int for a in alive_agents) / n_alive
        avg_T = sum(a.T_int for a in alive_agents) / n_alive
        avg_Valence = sum(a.valence_integrated for a in alive_agents) / n_alive
    else:
        avg_E = 0; avg_T = 0; avg_Valence = 0

    textstr = '\n'.join((
        f'Step: {step_number}',
        f'Alive: {n_alive} | Dead: {model.dead_count}',
        f'Avg Energy: {avg_E:.1f}',
        f'Avg Temp: {avg_T:.1f}',
        f'Avg Mood: {avg_Valence:.2f}'
    ))

    # Place a text box in upper left in axes coords
    props = dict(boxstyle='round', facecolor='white', alpha=0.8)
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', bbox=props)

    # Map settings
    ax.set_xlim(-0.5, model.grid.width-0.5)
    ax.set_ylim(-0.5, model.grid.height-0.5)
    ax.axis('off')
    
    # Cleanup
    plt.close(fig)
    
    return fig

# ==========================================
# Solara GUI (OPTIMIZED)
# ==========================================

# ✅ FIX: Helper functions for callbacks (avoids lambda closures)
def create_model():
    """Factory function for model creation"""
    return DualDriveModel()

def increment_tick(t):
    """Increment function for tick counter"""
    return t + 1

@solara.component
def Page():
    reset_ctr, set_reset = solara.use_state(0)
    model = solara.use_memo(create_model, dependencies=[reset_ctr])  # ✅ FIX: No longer using lambda
    
    # Tick counter
    tick, set_tick = solara.use_state(0)
    is_playing, set_playing = solara.use_state(False)
    current_task, set_current_task = solara.use_state(None)  # ✅ FIX: Track task explicitly

    # ==========================================
    # CALLBACK FUNCTIONS (defined BEFORE use)
    # ==========================================
    
    def on_step():
        model.step()
        set_tick(tick + 1)

    def on_reset():
        set_playing(False)
        set_reset(reset_ctr + 1)
        set_tick(0)
        
    def on_play():
        set_playing(not is_playing)

    def run_loop():
        """
        ✅ FIX: Optimized version with task tracking to prevent leaks
        """
        # Cleanup old task if it exists
        if current_task is not None:
            current_task.cancel()
            set_current_task(None)
        
        if not is_playing:
            return
        
        async def loop():
            try:
                while True:
                    model.step()
                    set_tick(increment_tick)  # ✅ FIX: Using function instead of lambda
                    await asyncio.sleep(0.1)
            except asyncio.CancelledError:
                # Normal cleanup - do not re-raise
                pass
        
        # Create new task
        task = asyncio.create_task(loop())
        set_current_task(task)
        
        def cleanup():
            task.cancel()
            set_current_task(None)
        
        return cleanup

    solara.use_effect(run_loop, [is_playing])

    # ==========================================
    # STATS CALCULATION
    # ==========================================
    
    agents = list(model.agent_set)
    alive_agents = [a for a in agents if a.is_alive]
    n_alive = len(alive_agents)
    
    if n_alive > 0:
        avg_E = sum(a.E_int for a in alive_agents) / n_alive
        avg_T = sum(a.T_int for a in alive_agents) / n_alive
        avg_Valence = sum(a.valence_integrated for a in alive_agents) / n_alive
    else:
        avg_E = 0; avg_T = 0; avg_Valence = 0

    # ==========================================
    # UI RENDERING
    # ==========================================
    
    with solara.Sidebar():
        solara.Markdown("## 🧠 Swarm FEP Agents")
        solara.Markdown("Multi-agent simulation with shared food trails.")
        
        with solara.Row():
            solara.Button("Step", on_click=on_step, color="warning")
            solara.Button("Play/Pause", on_click=on_play, color="success" if is_playing else "primary")
            solara.Button("Reset", on_click=on_reset, color="error")
            
        solara.Markdown("---")
        # Step Counter
        solara.Markdown(f"### Steps: {tick}")
        solara.Markdown(f"**Alive:** {n_alive} / **Dead:** {model.dead_count}")
        
        solara.Markdown("---")
        solara.Markdown(f"**Avg Energy:** {avg_E:.1f}")
        solara.Markdown(f"**Avg Temp:** {avg_T:.1f}")
        solara.Markdown(f"**Avg Mood:** {avg_Valence:.2f}")
        
        solara.Markdown("---")
        solara.Markdown("**Legend:**")
        solara.Markdown(f"⚪ **White:** OK (Optimal)")
        solara.Markdown(f"🟤 **Brown:** Hungry (Searching)")
        solara.Markdown(f"🔵 **Blue:** Cold (Hypothermia)")
        solara.Markdown(f"🔴 **Red:** Hot (Hyperthermia)")
        solara.Markdown(f"🟠 **Orange:** Scent Trail")
        solara.Markdown(f"🟢 **Green:** Food")

        solara.Markdown("---")
        solara.Markdown("### Theoretical Framework")
        
        # 1. Philosophy
        solara.Markdown("**1. Philosophy (Solms/Friston):**")
        solara.Markdown("""
        Agents are self-evidencing systems resisting entropy. 
        **Affect** (Emotion) is the subjective experience of the changing rate of error (Free Energy).
        """)

        # 2. Math - Homeostatic Error
        solara.Markdown("**2. Physiological Error ($H$):**")
        solara.Markdown(r"""
        The drive to correct internal states:
        $$ H = w\_T |T\_{int} - T\_{pref}| + w\_E \max(0, E\_{crit} - E\_{int}) $$
        """)

        # 3. Math - Active Inference
        solara.Markdown("**3. Active Inference ($G$):**")
        solara.Markdown(r"""
        Agents select moves to minimize Expected Free Energy ($G$):
        $$ G(action) = \underbrace{G\_{pragmatic}}\_{\text{Survival}} + \underbrace{G\_{epistemic}}\_{\text{Curiosity}} + \underbrace{G\_{social}}\_{\text{Swarm}} $$
        """)
        
        # 4. Affect and Precision
        solara.Markdown("**4. Affect and Precision:**")
        solara.Markdown(r"""
        Affect (Mood) is the rate of change of the error. This integrated valence determines the agent's Precision β.
        $$\beta_t = -\frac{H_t - H_{t-1}}{\Delta t}$$
        """)

        # 5. Math - Action Selection
        solara.Markdown("**5. Action Selection:**")
        solara.Markdown(r"""
        Softmax probability modulated by Precision ($\beta$):
        $$ P(a) = \frac{e^{\beta \cdot G(a)}}{\sum e^{\beta \cdot G(a_i)}} $$
        """)

    # Generate figure at each render with step number passed for overlay
    fig = get_plot_figure(model, step_number=tick)
    solara.FigureMatplotlib(fig)


# ==================== VIDEO FRAME GENERATION (OPTIONAL) ====================
def generate_video_frames(steps=200, output_dir="frames"):
    """Runs the simulation and saves each step as an image."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Initialize model
    model = DualDriveModel()
    print(f"Generating {steps} frames in folder '{output_dir}'...")

    for i in range(steps):
        model.step()
        # Pass loop index 'i' as step_number for the overlay
        fig = get_plot_figure(model, step_number=i)
        
        # Save frame
        fig.savefig(f"{output_dir}/frame_{i:04d}.png", dpi=120)
        plt.close(fig) # Critical to prevent memory leak
        
        if i % 10 == 0:
            print(f"Frames processed: {i}/{steps}")

    print("Done! All frames have been saved.")


if __name__ == "__main__":
    print(f"Starting simulation with {NUM_AGENTS} agents...")
    print("Run with: solara run multiagent_FEP.py")

    # uncomment the lines below to generate video frames (make sure to have enough disk space and time)
    # print("Generate video file comand: ffmpeg -framerate 10 -i frames/frame_%04d.png -c:v libx264 -pix_fmt yuv420p swarm_simulation.mp4")
    # generate_video_frames(steps=3000)
