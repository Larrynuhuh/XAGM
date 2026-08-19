import xagm
from xagm.manifolds import vectors as vct
from xagm.manifolds import calc
from xagm.basis import metrics as mtc
from xagm.basis import linear as lin
import jax
import jax.numpy as jnp
import time

import numpy as np
import plotly.graph_objects as go
import jax
import jax.numpy as jnp

def saddle_embedding(params):
    u, v = params
    return jnp.array([u, v, u**2 - v**2])

# =====================================================================
# 1. TEXTBOOK GRID EMBEDDING GENERATION
# =====================================================================

# Generate smooth grid evaluation points for the surface
# 45x45 resolution creates perfectly spaced, clean geometric "chops"
u_vals = np.linspace(-1.5, 1.5, 45)
v_vals = np.linspace(-1.5, 1.5, 45)
U, V = np.meshgrid(u_vals, v_vals)
X, Y, Z = U, V, U**2 - V**2

# --- YOUR REAL MATH ENGINE ARRAYS ---
# [Plug your native numpy-cast arrays here]
# Mapping matching your exact asymmetric test parameters [1.5, 0.2]
p_start_3d = np.array(saddle_embedding(p_start)) if 'p_start' in locals() else np.array([0.0, 0.0, 0.0])
p_end_3d = np.array(saddle_embedding(pos)) if 'pos' in locals() else np.array([1.2, 0.16, 1.2**2 - 0.16**2])

# Simulating your true continuous path_3d data stream from diffrax
t_steps = np.linspace(0, 1, 100)
u_track = 1.2 * t_steps
v_track = 0.16 * (t_steps**1.5) # Simulating curve drift away from the chord
z_track = u_track**2 - v_track**2
path_3d = np.stack([u_track, v_track, z_track], axis=1)

# Vector 3D direction values from your Jacobian matrix multiplication
v_start_dir = np.array([0.0, 0.4, 0.0])   # Cyan initial arrow vector
v_end_dir = np.array([0.2, 0.35, 0.3])    # Pink transported arrow vector

# =====================================================================
# 2. THE TEXTBOOK COVER GRAPHIC CONSTRUCT
# =====================================================================

fig = go.Figure()

# ─── THE MANIFOLD SURFACE (Boxy, semi-translucent geometric glass) ───
# ─── THE MANIFOLD SURFACE (Opaque Matte Geometry) ───
fig.add_trace(go.Surface(
    x=X, y=Y, z=Z,
    colorscale=[[0, '#005F73'], [0.5, '#0A192F'], [1, '#9B2226']], 
    opacity=1.0,          # FIXED: Completely solid surface texture
    showscale=False,
    # FIXED: Roughness increased, specular slashed to eliminate the reflection spot
    lighting=dict(ambient=0.7, roughness=0.9, diffuse=0.8, specular=0.1),
    # FIXED: High-contrast grid borders that pop over an opaque surface
    contours=dict(
        x=dict(show=True, color='#FFFFFF', width=1.5, highlight=False),
        y=dict(show=True, color='#FFFFFF', width=1.5, highlight=False)
    ),
    hoverinfo='none'
))


# ─── TRUE GEODESIC TRAJECTORY (Glowing Neon Green Ribbon) ───
fig.add_trace(go.Scatter3d(
    x=path_3d[:, 0], y=path_3d[:, 1], z=path_3d[:, 2],
    mode='lines',
    line=dict(color='#00FF66', width=7.5),
    name='True Geodesic Trajectory (In-Manifold)',
    hoverinfo='none'
))

# ─── AMBIENT EUCLIDEAN SHORTCUT (Crisp Electric Orange Laser Line) ───
fig.add_trace(go.Scatter3d(
    x=[p_start_3d[0], p_end_3d[0]],
    y=[p_start_3d[1], p_end_3d[1]],
    z=[p_start_3d[2], p_end_3d[2]],
    mode='lines',
    line=dict(color='#FF5733', width=3.5, dash='dash'),
    name='Euclidean Ambient Chord',
    hoverinfo='none'
))

# =====================================================================
# HYBRID TRUE VECTOR ARROW GENERATOR (LINE SHAFT + CONE CAP)
# =====================================================================

def add_vector_arrow(fig, base_pt, dir_vec, color, name):
    # Calculate the definitive endpoint of the vector arrow
    end_pt = base_pt + dir_vec
    
    # 1. THE SHAFT: Clean, high-visibility solid 3D line
    fig.add_trace(go.Scatter3d(
        x=[base_pt[0], end_pt[0]],
        y=[base_pt[1], end_pt[1]],
        z=[base_pt[2], end_pt[2]],
        mode='lines',
        line=dict(color=color, width=5.5),
        name=f"{name} Shaft",
        showlegend=False, hoverinfo='none'
    ))
    
    # 2. THE HEAD: Scale a small cone pinned exactly at the end destination
    # Setting u, v, w components parallel to dir_vec keeps it pointing seamlessly
    fig.add_trace(go.Cone(
        x=[end_pt[0]], y=[end_pt[1]], z=[end_pt[2]],
        u=[dir_vec[0]], v=[dir_vec[1]], w=[dir_vec[2]],
        colorscale=[[0, color], [1, color]], showscale=False,
        sizemode='scaled', sizeref=0.08, # Keeps the arrow point tiny and sharp
        anchor='tip', # 'tip' ensures the pointy end points cleanly out from the line
        name=name
    ))

# --- Trigger the Hybrid Vector Generators ---
# Vector 1: Initial Vector (Cyan Arrow)
add_vector_arrow(fig, p_start_3d, v_start_dir, '#00D2FF', 'Initial Vector')

# Vector 2: Parallel Transported Vector (Hot Pink Arrow)
add_vector_arrow(fig, p_end_3d, v_end_dir, '#FF007F', 'Transported Vector')



# Anchor Node Pins (Glowing white-rimmed coordinate boundaries)
fig.add_trace(go.Scatter3d(
    x=[p_start_3d[0], p_end_3d[0]],
    y=[p_start_3d[1], p_end_3d[1]],
    z=[p_start_3d[2], p_end_3d[2]],
    mode='markers',
    marker=dict(color=['#00D2FF', '#FF007F'], size=7, line=dict(color='#FFFFFF', width=2)),
    showlegend=False, hoverinfo='none'
))

# =====================================================================
# 3. INTERACTIVE LAYOUT DESIGN (THE TYPOGRAPHY)
# =====================================================================

fig.update_layout(
        scene=dict(
        # Perfect aspect ratio proportions ensures shapes aren't squished
        aspectratio=dict(x=1.2, y=1.2, z=0.9),
        camera=dict(
            eye=dict(x=1.4, y=-1.4, z=1.0), # Optimal cinematic overview angle
            up=dict(x=0, y=0, z=1)
        ),
        # Clean, stripped-down minimalist layout grid walls
        xaxis=dict(
            gridcolor='#121820', 
            zeroline=False, 
            title=dict(text='U Grid', font=dict(family='monospace', color='#66FCF1')), 
            tickfont=dict(family='monospace', color='#C5C6C7')
        ),
        yaxis=dict(
            gridcolor='#121820', 
            zeroline=False, 
            title=dict(text='V Grid', font=dict(family='monospace', color='#66FCF1')), 
            tickfont=dict(family='monospace', color='#C5C6C7')
        ),
        zaxis=dict(
            gridcolor='#121820', 
            zeroline=False, 
            title=dict(text='Ambient Z', font=dict(family='monospace', color='#66FCF1')), 
            tickfont=dict(family='monospace', color='#C5C6C7')
        ),
    ),

)

# Open up the premium canvas in your default web browser instantly!
fig.show()
