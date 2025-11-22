
import streamlit as st
import pandas as pd
import sqlite3
import torch
import torchvision
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import tempfile
import os
import io

# ============================
# PHASE 1: PREPARATION & DB VIEWER
# ============================

st.set_page_config(layout="wide", page_title="SNN-PPO Analysis Dashboard")
st.title("Operational Intelligence: SNN-PPO Analysis Dashboard")

# Sidebar for Navigation and Uploads
st.sidebar.header("Data Sources")

# ----------------------------
# DATABASE UPLOAD & LOADING
# ----------------------------
uploaded_db = st.sidebar.file_uploader("Upload Training Logs (.db)", type="db")

@st.cache_data
def load_db_data(file_obj):
    """
    Loads data from a SQLite file object into Pandas DataFrames.
    Uses tempfile to handle BytesIO because sqlite3 requires a path.
    """
    # Create a temp file to store the uploaded DB
    with tempfile.NamedTemporaryFile(delete=False, suffix=".db") as tmp:
        tmp.write(file_obj.read())
        tmp_path = tmp.name

    try:
        conn = sqlite3.connect(tmp_path)
        
        # Load Episodes
        episodes_df = pd.read_sql_query("SELECT * FROM episodes", conn)
        
        # Load Steps
        # Use aggregation for large datasets instead of LIMIT
        count = pd.read_sql_query("SELECT COUNT(*) FROM steps", conn).iloc[0,0]
        
        # If dataset is huge, we can't load it all at once to memory easily without chunking.
        # However, purely inside pandas, read_sql is often okay for a few million rows.
        # But to be safe and efficient, let's load it. 
        # Ideally, we would do aggregation in SQL, but that's complex for variable step counts per episode.
        # We'll load it and then downsample in Pandas if needed for plotting.
        # Warn if extremely large.
        if count > 1000000:
             st.warning(f"Large dataset ({count} steps). This might take a moment.")
        
        steps_df = pd.read_sql_query("SELECT * FROM steps", conn)

        # Load Reward Components
        reward_components_df = pd.read_sql_query("SELECT * FROM reward_components", conn)
        
        conn.close()
    finally:
        # Cleanup temp file
        os.remove(tmp_path)
        
    return episodes_df, steps_df, reward_components_df

# ----------------------------
# MAIN DASHBOARD LOGIC
# ----------------------------

if uploaded_db:
    with st.spinner("Loading Database..."):
        episodes_df, steps_df, reward_components_df = load_db_data(uploaded_db)
    
    st.success("Database Loaded Successfully!")
    
    # --- METRICS SUMMARY ---
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Episodes", len(episodes_df))
    with col2:
        st.metric("Avg Total Reward", f"{episodes_df['total_reward'].mean():.2f}")
    with col3:
        st.metric("Max Total Reward", f"{episodes_df['total_reward'].max():.2f}")
    with col4:
        st.metric("Avg Episode Length", f"{episodes_df['steps'].mean():.1f}")
    
    st.markdown("---")
    
    # ============================
    # PHASE 2: DYNAMICS
    # ============================
    st.subheader("Training Dynamics")

    # 1. Total Reward with Smoothing
    st.markdown("### Total Reward Trends")
    smoothing_window = st.slider("Smoothing Window (Episodes)", min_value=1, max_value=100, value=10)
    
    # Calculate smoothed curve
    episodes_df['smoothed_reward'] = episodes_df['total_reward'].rolling(window=smoothing_window).mean()
    
    fig_reward = go.Figure()
    # Raw data (faint)
    fig_reward.add_trace(go.Scatter(
        x=episodes_df['episode_id'], 
        y=episodes_df['total_reward'],
        mode='lines',
        name='Raw Reward',
        line=dict(color='rgba(0,100,255,0.2)'),
        hoverinfo='skip'
    ))
    # Smoothed data (solid)
    fig_reward.add_trace(go.Scatter(
        x=episodes_df['episode_id'], 
        y=episodes_df['smoothed_reward'],
        mode='lines',
        name=f'Smoothed (MA-{smoothing_window})',
        line=dict(color='blue')
    ))
    fig_reward.update_layout(title="Total Reward per Episode", xaxis_title="Episode", yaxis_title="Reward")
    st.plotly_chart(fig_reward, use_container_width=True)
    
    # 2. Reward Decomposition (Stacked Area)
    st.markdown("### Reward Composition")
    # Melt the dataframe for plotly
    reward_cols = [c for c in reward_components_df.columns if 'reward' in c and c != 'total_reward' and c != 'episode_id']
    
    if reward_cols:
        # Applying smoothing to components as well
        rc_smoothed = reward_components_df.copy()
        for col in reward_cols:
            rc_smoothed[col] = rc_smoothed[col].rolling(window=smoothing_window).mean()
        
        # Remove NaNs from smoothing
        rc_smoothed = rc_smoothed.dropna()
        
        fig_area = px.area(
            rc_smoothed, 
            x='episode_id', 
            y=reward_cols,
            title="Reward Breakdown (Stacked)"
        )
        st.plotly_chart(fig_area, use_container_width=True)
    else:
        st.info("No reward components found.")

    # 3. Action Distribution (Heatmap)
    st.markdown("### Action Policy Evolution")
    
    if not steps_df.empty:
        # We need to bin episodes to create a heatmap (Time vs Action)
        # Bins: e.g., every 10 episodes, but ensure we don't have too many bins.
        # Target ~50-100 bins on X axis for readability
        n_bins = 50
        max_ep = steps_df['episode_id'].max()
        bin_size = max(1, int(max_ep / n_bins))
        
        steps_df['episode_bin'] = (steps_df['episode_id'] // bin_size) * bin_size
        
        # Count actions per bin
        action_counts = steps_df.groupby(['episode_bin', 'action']).size().reset_index(name='count')
        
        # Calculate probability (normalize by total steps in that bin)
        total_per_bin = steps_df.groupby(['episode_bin']).size().reset_index(name='total')
        action_counts = action_counts.merge(total_per_bin, on='episode_bin')
        action_counts['probability'] = action_counts['count'] / action_counts['total']
        
        fig_heatmap = px.density_heatmap(
            action_counts, 
            x="episode_bin", 
            y="action", 
            z="probability", 
            nbinsx=n_bins,
            title="Action Probability Heatmap (Policy Drift)",
            color_continuous_scale="Viridis",
            labels={'episode_bin': 'Episode (Binned)', 'probability': 'Probability'}
        )
        st.plotly_chart(fig_heatmap, use_container_width=True)
        
        # Entropy Calculation
        # H(X) = -sum(p * log(p))
        st.markdown("#### Policy Entropy")
        # Reuse action_counts
        # Add small epsilon to avoid log(0)
        action_counts['entropy_component'] = -action_counts['probability'] * np.log(action_counts['probability'] + 1e-9)
        entropy_df = action_counts.groupby('episode_bin')['entropy_component'].sum().reset_index(name='entropy')
        
        fig_entropy = px.line(entropy_df, x='episode_bin', y='entropy', title="Policy Entropy Over Time")
        st.plotly_chart(fig_entropy, use_container_width=True)

    # 4. Episode Length Analysis
    st.markdown("### Episode Length Analysis")
    fig_scatter = px.scatter(
        episodes_df, 
        x="steps", 
        y="total_reward", 
        title="Episode Length vs. Total Reward (Cluster Analysis)",
        color="episode_id", # Color by time to see drift
        hover_data=["episode_id"]
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

else:
    st.info("Please upload a 'training_logs.db' file to see analysis.")

# ============================
# PHASE 3: CHECKPOINT INSPECTOR
# ============================
st.sidebar.markdown("---")
uploaded_ckpt = st.sidebar.file_uploader("Upload Checkpoint (.pth)", type="pth")

@st.cache_resource
def load_model_ckpt(file_obj):
    """
    Loads PyTorch checkpoint.
    Uses @st.cache_resource to avoid reloading heavy models.
    """
    # Load to CPU to avoid GPU/OOM issues
    return torch.load(file_obj, map_location='cpu')

def get_state_dict(ckpt):
    # Logic from analyze_pth.py
    if isinstance(ckpt, dict):
        for key in ["agent_state", "policy_net_state_dict", "model_state_dict", "state_dict"]:
            if key in ckpt and isinstance(ckpt[key], dict):
                return ckpt[key]
        if all(isinstance(v, torch.Tensor) for v in ckpt.values()):
            return ckpt
    return None

if uploaded_ckpt:
    st.header("Checkpoint Introspection")
    ckpt = load_model_ckpt(uploaded_ckpt)
    
    state_dict = get_state_dict(ckpt)
    
    if state_dict:
        # Layer Selector
        layer_names = list(state_dict.keys())
        selected_layer = st.selectbox("Select Layer/Parameter", layer_names)
        
        tensor = state_dict[selected_layer]
        st.write(f"Shape: {tuple(tensor.shape)} | Type: {tensor.dtype}")
        
        col1, col2 = st.columns(2)
        
        # Stats
        with col1:
            st.write("Statistics:")
            st.write(f"Mean: {tensor.float().mean():.4f}")
            st.write(f"Std: {tensor.float().std():.4f}")
            st.write(f"Min: {tensor.float().min():.4f}")
            st.write(f"Max: {tensor.float().max():.4f}")

        # Visualization
        with col2:
            st.write("Visualization:")
            
            # 1. Histogram (Weight Distribution)
            flat_tensor = tensor.float().cpu().numpy().flatten()
            fig_hist = px.histogram(
                x=flat_tensor, 
                nbins=50, 
                title=f"Weight Distribution: {selected_layer}",
                log_y=True # Log scale to see outliers/sparsity
            )
            st.plotly_chart(fig_hist, use_container_width=True)
            
        # 2. Tensor Visualization (Grid or Heatmap)
        st.markdown("#### Tensor Visualization")
        
        if tensor.ndim == 4: # Conv2D (Out, In, H, W)
            st.write("Conv2D Weights Visualization (Grid View)")
            
            # Normalize for visualization
            # torchvision.utils.make_grid expects (B, C, H, W) or (B, H, W) for single channel
            # Our tensor is (Out, In, H, W).
            # We want to visualize filters.
            # Flatten to (Out * In, 1, H, W) to show every single 2D kernel?
            # Or just show the first few filters?
            
            # Let's show the kernels for the first input channel across all output channels, 
            # and first output channel across all input channels.
            
            view_mode = st.radio("View Mode", ["Grid (First Input Channel)", "Grid (First Output Channel)", "Slice Explorer"])
            
            if view_mode == "Grid (First Input Channel)":
                # Visualize kernels (Out, 0, H, W)
                kernels = tensor[:, 0:1, :, :].float()
                # Normalize to 0-1 range for plotting
                min_v, max_v = kernels.min(), kernels.max()
                if max_v > min_v:
                    kernels = (kernels - min_v) / (max_v - min_v)
                
                grid = torchvision.utils.make_grid(kernels, nrow=8, padding=1)
                # Convert to numpy (C, H, W) -> (H, W, C)
                grid_np = grid.permute(1, 2, 0).cpu().numpy()
                
                st.image(grid_np, caption="Kernels for Input Channel 0", width=600)
                
            elif view_mode == "Grid (First Output Channel)":
                # Visualize kernels (0, In, H, W) -> treat In as Batch
                kernels = tensor[0:1, :, :, :].permute(1, 0, 2, 3).float()
                min_v, max_v = kernels.min(), kernels.max()
                if max_v > min_v:
                    kernels = (kernels - min_v) / (max_v - min_v)
                
                grid = torchvision.utils.make_grid(kernels, nrow=8, padding=1)
                grid_np = grid.permute(1, 2, 0).cpu().numpy()
                
                st.image(grid_np, caption="Kernels for Output Channel 0", width=600)
                
            else:
                # Slicer
                out_channels = tensor.shape[0]
                in_channels = tensor.shape[1]
                
                c1, c2 = st.columns(2)
                sel_out = c1.slider("Output Channel", 0, out_channels-1, 0)
                sel_in = c2.slider("Input Channel", 0, in_channels-1, 0)
                
                kernel = tensor[sel_out, sel_in].float().cpu().numpy()
                fig_kernel = px.imshow(kernel, color_continuous_scale='RdBu', title=f"Kernel [{sel_out}, {sel_in}]")
                st.plotly_chart(fig_kernel, use_container_width=False)
            
        elif tensor.ndim == 2: # Linear (Out, In)
            st.write("Linear Weight Matrix")
            # Heatmap might be too big. Downsample?
            if tensor.numel() > 10000:
                st.warning("Matrix too large for full heatmap. Showing top-left 100x100 slice.")
                matrix_slice = tensor[:100, :100].float().cpu().numpy()
            else:
                matrix_slice = tensor.float().cpu().numpy()
                
            fig_matrix = px.imshow(matrix_slice, color_continuous_scale='RdBu', title="Weight Heatmap")
            st.plotly_chart(fig_matrix, use_container_width=True)
            
    else:
        st.error("Could not identify state_dict in checkpoint.")

import streamlit as st
import pandas as pd
import sqlite3
import torch
import torchvision
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import tempfile
import os
import io

# ============================
# PHASE 1: PREPARATION & DB VIEWER
# ============================

st.set_page_config(layout="wide", page_title="SNN-PPO Analysis Dashboard")
st.title("Operational Intelligence: SNN-PPO Analysis Dashboard")

# Sidebar for Navigation and Uploads
st.sidebar.header("Data Sources")

# ----------------------------
# DATABASE UPLOAD & LOADING
# ----------------------------
uploaded_db = st.sidebar.file_uploader("Upload Training Logs (.db)", type="db")

@st.cache_data
def load_db_data(file_obj):
    """
    Loads data from a SQLite file object into Pandas DataFrames.
    Uses tempfile to handle BytesIO because sqlite3 requires a path.
    """
    # Create a temp file to store the uploaded DB
    with tempfile.NamedTemporaryFile(delete=False, suffix=".db") as tmp:
        tmp.write(file_obj.read())
        tmp_path = tmp.name

    try:
        conn = sqlite3.connect(tmp_path)

        # Load Episodes
        episodes_df = pd.read_sql_query("SELECT * FROM episodes", conn)

        # Load Steps
        # Use aggregation for large datasets instead of LIMIT
        count = pd.read_sql_query("SELECT COUNT(*) FROM steps", conn).iloc[0,0]

        # If dataset is huge, we can't load it all at once to memory easily without chunking.
        # However, purely inside pandas, read_sql is often okay for a few million rows.
        # But to be safe and efficient, let's load it.
        # Ideally, we would do aggregation in SQL, but that's complex for variable step counts per episode.
        # We'll load it and then downsample in Pandas if needed for plotting.
        # Warn if extremely large.
        if count > 1000000:
             st.warning(f"Large dataset ({count} steps). This might take a moment.")

        steps_df = pd.read_sql_query("SELECT * FROM steps", conn)

        # Load Reward Components
        reward_components_df = pd.read_sql_query("SELECT * FROM reward_components", conn)

        conn.close()
    finally:
        # Cleanup temp file
        os.remove(tmp_path)

    return episodes_df, steps_df, reward_components_df

# ----------------------------
# MAIN DASHBOARD LOGIC
# ----------------------------

if uploaded_db:
    with st.spinner("Loading Database..."):
        episodes_df, steps_df, reward_components_df = load_db_data(uploaded_db)

    st.success("Database Loaded Successfully!")

    # --- METRICS SUMMARY ---
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Episodes", len(episodes_df))
    with col2:
        st.metric("Avg Total Reward", f"{episodes_df['total_reward'].mean():.2f}")
    with col3:
        st.metric("Max Total Reward", f"{episodes_df['total_reward'].max():.2f}")
    with col4:
        st.metric("Avg Episode Length", f"{episodes_df['steps'].mean():.1f}")

    st.markdown("---")

    # ============================
    # PHASE 2: DYNAMICS
    # ============================
    st.subheader("Training Dynamics")

    # 1. Total Reward with Smoothing
    st.markdown("### Total Reward Trends")
    smoothing_window = st.slider("Smoothing Window (Episodes)", min_value=1, max_value=100, value=10)

    # Calculate smoothed curve
    episodes_df['smoothed_reward'] = episodes_df['total_reward'].rolling(window=smoothing_window).mean()

    fig_reward = go.Figure()
    # Raw data (faint)
    fig_reward.add_trace(go.Scatter(
        x=episodes_df['episode_id'],
        y=episodes_df['total_reward'],
        mode='lines',
        name='Raw Reward',
        line=dict(color='rgba(0,100,255,0.2)'),
        hoverinfo='skip'
    ))
    # Smoothed data (solid)
    fig_reward.add_trace(go.Scatter(
        x=episodes_df['episode_id'],
        y=episodes_df['smoothed_reward'],
        mode='lines',
        name=f'Smoothed (MA-{smoothing_window})',
        line=dict(color='blue')
    ))
    fig_reward.update_layout(title="Total Reward per Episode", xaxis_title="Episode", yaxis_title="Reward")
    st.plotly_chart(fig_reward, use_container_width=True)

    # 2. Reward Decomposition (Stacked Area)
    st.markdown("### Reward Composition")
    # Melt the dataframe for plotly
    reward_cols = [c for c in reward_components_df.columns if 'reward' in c and c != 'total_reward' and c != 'episode_id']

    if reward_cols:
        # Applying smoothing to components as well
        rc_smoothed = reward_components_df.copy()
        for col in reward_cols:
            rc_smoothed[col] = rc_smoothed[col].rolling(window=smoothing_window).mean()

        # Remove NaNs from smoothing
        rc_smoothed = rc_smoothed.dropna()

        fig_area = px.area(
            rc_smoothed,
            x='episode_id',
            y=reward_cols,
            title="Reward Breakdown (Stacked)"
        )
        st.plotly_chart(fig_area, use_container_width=True)
    else:
        st.info("No reward components found.")

    # 3. Action Distribution (Heatmap)
    st.markdown("### Action Policy Evolution")

    if not steps_df.empty:
        # We need to bin episodes to create a heatmap (Time vs Action)
        # Bins: e.g., every 10 episodes, but ensure we don't have too many bins.
        # Target ~50-100 bins on X axis for readability
        n_bins = 50
        max_ep = steps_df['episode_id'].max()
        bin_size = max(1, int(max_ep / n_bins))

        steps_df['episode_bin'] = (steps_df['episode_id'] // bin_size) * bin_size

        # Count actions per bin
        action_counts = steps_df.groupby(['episode_bin', 'action']).size().reset_index(name='count')

        # Calculate probability (normalize by total steps in that bin)
        total_per_bin = steps_df.groupby(['episode_bin']).size().reset_index(name='total')
        action_counts = action_counts.merge(total_per_bin, on='episode_bin')
        action_counts['probability'] = action_counts['count'] / action_counts['total']

        fig_heatmap = px.density_heatmap(
            action_counts,
            x="episode_bin",
            y="action",
            z="probability",
            nbinsx=n_bins,
            title="Action Probability Heatmap (Policy Drift)",
            color_continuous_scale="Viridis",
            labels={'episode_bin': 'Episode (Binned)', 'probability': 'Probability'}
        )
        st.plotly_chart(fig_heatmap, use_container_width=True)

        # Entropy Calculation
        # H(X) = -sum(p * log(p))
        st.markdown("#### Policy Entropy")
        # Reuse action_counts
        # Add small epsilon to avoid log(0)
        action_counts['entropy_component'] = -action_counts['probability'] * np.log(action_counts['probability'] + 1e-9)
        entropy_df = action_counts.groupby('episode_bin')['entropy_component'].sum().reset_index(name='entropy')

        fig_entropy = px.line(entropy_df, x='episode_bin', y='entropy', title="Policy Entropy Over Time")
        st.plotly_chart(fig_entropy, use_container_width=True)

    # 4. Episode Length Analysis
    st.markdown("### Episode Length Analysis")
    fig_scatter = px.scatter(
        episodes_df,
        x="steps",
        y="total_reward",
        title="Episode Length vs. Total Reward (Cluster Analysis)",
        color="episode_id", # Color by time to see drift
        hover_data=["episode_id"]
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

else:
    st.info("Please upload a 'training_logs.db' file to see analysis.")

# ============================
# PHASE 3: CHECKPOINT INSPECTOR
# ============================
st.sidebar.markdown("---")
uploaded_ckpt = st.sidebar.file_uploader("Upload Checkpoint (.pth)", type="pth")

@st.cache_resource
def load_model_ckpt(file_obj):
    """
    Loads PyTorch checkpoint.
    Uses @st.cache_resource to avoid reloading heavy models.
    """
    # Load to CPU to avoid GPU/OOM issues
    return torch.load(file_obj, map_location='cpu')

def get_state_dict(ckpt):
    # Logic from analyze_pth.py
    if isinstance(ckpt, dict):
        for key in ["agent_state", "policy_net_state_dict", "model_state_dict", "state_dict"]:
            if key in ckpt and isinstance(ckpt[key], dict):
                return ckpt[key]
        if all(isinstance(v, torch.Tensor) for v in ckpt.values()):
            return ckpt
    return None

if uploaded_ckpt:
    st.header("Checkpoint Introspection")
    ckpt = load_model_ckpt(uploaded_ckpt)

    state_dict = get_state_dict(ckpt)

    if state_dict:
        # Layer Selector
        layer_names = list(state_dict.keys())
        selected_layer = st.selectbox("Select Layer/Parameter", layer_names)

        tensor = state_dict[selected_layer]
        st.write(f"Shape: {tuple(tensor.shape)} | Type: {tensor.dtype}")

        col1, col2 = st.columns(2)

        # Stats
        with col1:
            st.write("Statistics:")
            st.write(f"Mean: {tensor.float().mean():.4f}")
            st.write(f"Std: {tensor.float().std():.4f}")
            st.write(f"Min: {tensor.float().min():.4f}")
            st.write(f"Max: {tensor.float().max():.4f}")

        # Visualization
        with col2:
            st.write("Visualization:")

            # 1. Histogram (Weight Distribution)
            flat_tensor = tensor.float().cpu().numpy().flatten()
            fig_hist = px.histogram(
                x=flat_tensor,
                nbins=50,
                title=f"Weight Distribution: {selected_layer}",
                log_y=True # Log scale to see outliers/sparsity
            )
            st.plotly_chart(fig_hist, use_container_width=True)

        # 2. Tensor Visualization (Grid or Heatmap)
        st.markdown("#### Tensor Visualization")

        if tensor.ndim == 4: # Conv2D (Out, In, H, W)
            st.write("Conv2D Weights Visualization (Grid View)")

            # Normalize for visualization
            # torchvision.utils.make_grid expects (B, C, H, W) or (B, H, W) for single channel
            # Our tensor is (Out, In, H, W).
            # We want to visualize filters.
            # Flatten to (Out * In, 1, H, W) to show every single 2D kernel?
            # Or just show the first few filters?

            # Let's show the kernels for the first input channel across all output channels,
            # and first output channel across all input channels.

            view_mode = st.radio("View Mode", ["Grid (First Input Channel)", "Grid (First Output Channel)", "Slice Explorer"])

            if view_mode == "Grid (First Input Channel)":
                # Visualize kernels (Out, 0, H, W)
                kernels = tensor[:, 0:1, :, :].float()
                # Normalize to 0-1 range for plotting
                min_v, max_v = kernels.min(), kernels.max()
                if max_v > min_v:
                    kernels = (kernels - min_v) / (max_v - min_v)

                grid = torchvision.utils.make_grid(kernels, nrow=8, padding=1)
                # Convert to numpy (C, H, W) -> (H, W, C)
                grid_np = grid.permute(1, 2, 0).cpu().numpy()

                st.image(grid_np, caption="Kernels for Input Channel 0", width=600)

            elif view_mode == "Grid (First Output Channel)":
                # Visualize kernels (0, In, H, W) -> treat In as Batch
                kernels = tensor[0:1, :, :, :].permute(1, 0, 2, 3).float()
                min_v, max_v = kernels.min(), kernels.max()
                if max_v > min_v:
                    kernels = (kernels - min_v) / (max_v - min_v)

                grid = torchvision.utils.make_grid(kernels, nrow=8, padding=1)
                grid_np = grid.permute(1, 2, 0).cpu().numpy()

                st.image(grid_np, caption="Kernels for Output Channel 0", width=600)

            else:
                # Slicer
                out_channels = tensor.shape[0]
                in_channels = tensor.shape[1]

                c1, c2 = st.columns(2)
                sel_out = c1.slider("Output Channel", 0, out_channels-1, 0)
                sel_in = c2.slider("Input Channel", 0, in_channels-1, 0)

                kernel = tensor[sel_out, sel_in].float().cpu().numpy()
                fig_kernel = px.imshow(kernel, color_continuous_scale='RdBu', title=f"Kernel [{sel_out}, {sel_in}]")
                st.plotly_chart(fig_kernel, use_container_width=False)

        elif tensor.ndim == 2: # Linear (Out, In)
            st.write("Linear Weight Matrix")
            # Heatmap might be too big. Downsample?
            if tensor.numel() > 10000:
                st.warning("Matrix too large for full heatmap. Showing top-left 100x100 slice.")
                matrix_slice = tensor[:100, :100].float().cpu().numpy()
            else:
                matrix_slice = tensor.float().cpu().numpy()

            fig_matrix = px.imshow(matrix_slice, color_continuous_scale='RdBu', title="Weight Heatmap")
            st.plotly_chart(fig_matrix, use_container_width=True)

    else:
        st.error("Could not identify state_dict in checkpoint.")

=