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

# ### NEW: Imports for dimensionality reduction
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

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
    with tempfile.NamedTemporaryFile(delete=False, suffix=".db") as tmp:
        tmp.write(file_obj.read())
        tmp_path = tmp.name

    try:
        conn = sqlite3.connect(tmp_path)
        episodes_df = pd.read_sql_query("SELECT * FROM episodes", conn)
        
        # Check size before loading all steps
        count = pd.read_sql_query("SELECT COUNT(*) FROM steps", conn).iloc[0,0]
        if count > 1000000:
            st.warning(f"Large dataset ({count} steps). This might take a moment.")
        
        steps_df = pd.read_sql_query("SELECT * FROM steps", conn)
        reward_components_df = pd.read_sql_query("SELECT * FROM reward_components", conn)
        conn.close()
    finally:
        os.remove(tmp_path)
        
    return episodes_df, steps_df, reward_components_df

# ### NEW: Helper functions for Weight-Space Map
def extract_weight_vectors(state_dict, max_points=5000):
    """
    Extracts flattened weight vectors from the state_dict for visualization.
    Handles Linear (row-wise) and Conv2D (filter-wise).
    """
    vectors = []
    metadata = []

    for name, tensor in state_dict.items():
        if not isinstance(tensor, torch.Tensor):
            continue
        # filter for weights only to keep the map clean
        if not name.endswith("weight"):
            continue

        arr = tensor.detach().cpu().float().numpy()
        shape = arr.shape
        ndim = arr.ndim

        # Extract layer name (remove .weight)
        layer = name.rsplit(".", 1)[0] if "." in name else name

        if ndim == 2:
            # Linear: [out, in] -> Each row is a neuron
            out_features = shape[0]
            for i in range(out_features):
                v = arr[i].ravel()
                vectors.append(v.copy())
                metadata.append({
                    "layer": layer,
                    "name": name,
                    "unit_idx": i,
                    "shape": shape,
                })

        elif ndim >= 3:
            # Conv: [out, in, kH, kW] -> Each [in, kH, kW] block is a filter
            out_channels = shape[0]
            for i in range(out_channels):
                v = arr[i].ravel()
                vectors.append(v.copy())
                metadata.append({
                    "layer": layer,
                    "name": name,
                    "unit_idx": i,
                    "shape": shape,
                })
        
        elif ndim == 1:
            # 1D params
            v = arr.ravel()
            vectors.append(v.copy())
            metadata.append({
                "layer": layer,
                "name": name,
                "unit_idx": None,
                "shape": shape,
            })

        if len(vectors) >= max_points:
            break

    if not vectors:
        return None, None

    return vectors, metadata

def build_2d_embedding(vectors):
    """
    Pads vectors to the same length, standardizes them, runs PCA, then t-SNE.
    """
    lengths = [v.size for v in vectors]
    max_len = max(lengths)
    N = len(vectors)

    # Pad with zeros to handle different layer sizes
    X = np.zeros((N, max_len), dtype=np.float32)
    for i, v in enumerate(vectors):
        L = v.size
        X[i, :L] = v

    # Standardize
    mean = X.mean(axis=0, keepdims=True)
    std = X.std(axis=0, keepdims=True) + 1e-8
    X_std = (X - mean) / std

    # 1. PCA (Reduce noise/dims before t-SNE)
    n_components_pca = min(50, X_std.shape[0], X_std.shape[1])
    if n_components_pca < 2:
        return np.zeros((N, 2)) # Fallback

    pca = PCA(n_components=n_components_pca)
    X_pca = pca.fit_transform(X_std)

    # 2. t-SNE
    perplexity = min(30.0, max(5.0, (N - 1) / 3))
    tsne = TSNE(n_components=2, perplexity=perplexity, init="pca", verbose=0)
    X_2d = tsne.fit_transform(X_pca)
    
    return X_2d

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
    
    episodes_df['smoothed_reward'] = episodes_df['total_reward'].rolling(window=smoothing_window).mean()
    
    fig_reward = go.Figure()
    fig_reward.add_trace(go.Scatter(
        x=episodes_df['episode_id'], y=episodes_df['total_reward'],
        mode='lines', name='Raw Reward', line=dict(color='rgba(0,100,255,0.2)'), hoverinfo='skip'
    ))
    fig_reward.add_trace(go.Scatter(
        x=episodes_df['episode_id'], y=episodes_df['smoothed_reward'],
        mode='lines', name=f'Smoothed (MA-{smoothing_window})', line=dict(color='blue')
    ))
    fig_reward.update_layout(title="Total Reward per Episode", xaxis_title="Episode", yaxis_title="Reward")
    st.plotly_chart(fig_reward, use_container_width=True)
    
    # 2. Reward Decomposition
    st.markdown("### Reward Composition")
    reward_cols = [c for c in reward_components_df.columns if 'reward' in c and c != 'total_reward' and c != 'episode_id']
    
    if reward_cols:
        rc_smoothed = reward_components_df.copy()
        for col in reward_cols:
            rc_smoothed[col] = rc_smoothed[col].rolling(window=smoothing_window).mean()
        rc_smoothed = rc_smoothed.dropna()
        
        fig_area = px.area(rc_smoothed, x='episode_id', y=reward_cols, title="Reward Breakdown (Stacked)")
        st.plotly_chart(fig_area, use_container_width=True)
    else:
        st.info("No reward components found.")

    # 3. Action Heatmap & Entropy
    st.markdown("### Action Policy Evolution")
    
    if not steps_df.empty:
        n_bins = 50
        max_ep = steps_df['episode_id'].max()
        bin_size = max(1, int(max_ep / n_bins))
        steps_df['episode_bin'] = (steps_df['episode_id'] // bin_size) * bin_size
        
        action_counts = steps_df.groupby(['episode_bin', 'action']).size().reset_index(name='count')
        total_per_bin = steps_df.groupby(['episode_bin']).size().reset_index(name='total')
        action_counts = action_counts.merge(total_per_bin, on='episode_bin')
        action_counts['probability'] = action_counts['count'] / action_counts['total']
        
        fig_heatmap = px.density_heatmap(
            action_counts, x="episode_bin", y="action", z="probability", nbinsx=n_bins,
            title="Action Probability Heatmap", color_continuous_scale="Viridis",
            labels={'episode_bin': 'Episode (Binned)', 'probability': 'Probability'}
        )
        st.plotly_chart(fig_heatmap, use_container_width=True)
        
        st.markdown("#### Policy Entropy")
        action_counts['entropy_component'] = -action_counts['probability'] * np.log(action_counts['probability'] + 1e-9)
        entropy_df = action_counts.groupby('episode_bin')['entropy_component'].sum().reset_index(name='entropy')
        fig_entropy = px.line(entropy_df, x='episode_bin', y='entropy', title="Policy Entropy Over Time")
        st.plotly_chart(fig_entropy, use_container_width=True)

    # 4. Episode Length vs Reward
    st.markdown("### Episode Length Analysis")
    fig_scatter = px.scatter(
        episodes_df, x="steps", y="total_reward", title="Episode Length vs. Total Reward",
        color="episode_id", hover_data=["episode_id"]
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
    return torch.load(file_obj, map_location='cpu')

def get_state_dict(ckpt):
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
        # --- PER LAYER INSPECTOR ---
        layer_names = list(state_dict.keys())
        selected_layer = st.selectbox("Select Layer/Parameter", layer_names)
        
        tensor = state_dict[selected_layer]
        st.write(f"Shape: {tuple(tensor.shape)} | Type: {tensor.dtype}")
        
        col1, col2 = st.columns(2)
        with col1:
            st.write("Statistics:")
            st.write(f"Mean: {tensor.float().mean():.4f}")
            st.write(f"Std: {tensor.float().std():.4f}")
            st.write(f"Min: {tensor.float().min():.4f}")
            st.write(f"Max: {tensor.float().max():.4f}")

        with col2:
            st.write("Visualization:")
            flat_tensor = tensor.float().cpu().numpy().flatten()
            fig_hist = px.histogram(x=flat_tensor, nbins=50, title=f"Weight Dist: {selected_layer}", log_y=True)
            st.plotly_chart(fig_hist, use_container_width=True)
            
        st.markdown("#### Tensor Visualization")
        if tensor.ndim == 4: # Conv2D
            view_mode = st.radio("View Mode", ["Grid (First Input Channel)", "Grid (First Output Channel)", "Slice Explorer"], horizontal=True)
            if view_mode == "Grid (First Input Channel)":
                kernels = tensor[:, 0:1, :, :].float()
                min_v, max_v = kernels.min(), kernels.max()
                if max_v > min_v: kernels = (kernels - min_v) / (max_v - min_v)
                grid = torchvision.utils.make_grid(kernels, nrow=8, padding=1)
                grid_np = grid.permute(1, 2, 0).cpu().numpy()
                st.image(grid_np, caption="Kernels for Input Channel 0", width=600)
            elif view_mode == "Grid (First Output Channel)":
                kernels = tensor[0:1, :, :, :].permute(1, 0, 2, 3).float()
                min_v, max_v = kernels.min(), kernels.max()
                if max_v > min_v: kernels = (kernels - min_v) / (max_v - min_v)
                grid = torchvision.utils.make_grid(kernels, nrow=8, padding=1)
                grid_np = grid.permute(1, 2, 0).cpu().numpy()
                st.image(grid_np, caption="Kernels for Output Channel 0", width=600)
            else:
                c1, c2 = st.columns(2)
                sel_out = c1.slider("Output Channel", 0, tensor.shape[0]-1, 0)
                sel_in = c2.slider("Input Channel", 0, tensor.shape[1]-1, 0)
                kernel = tensor[sel_out, sel_in].float().cpu().numpy()
                fig_kernel = px.imshow(kernel, color_continuous_scale='RdBu', title=f"Kernel [{sel_out}, {sel_in}]")
                st.plotly_chart(fig_kernel, use_container_width=False)
        elif tensor.ndim == 2: # Linear
            if tensor.numel() > 10000:
                st.warning("Matrix too large for full heatmap. Showing top-left 100x100 slice.")
                matrix_slice = tensor[:100, :100].float().cpu().numpy()
            else:
                matrix_slice = tensor.float().cpu().numpy()
            fig_matrix = px.imshow(matrix_slice, color_continuous_scale='RdBu', title="Weight Heatmap")
            st.plotly_chart(fig_matrix, use_container_width=True)

        # ### NEW: GLOBAL WEIGHT-SPACE MAP SECTION
        st.markdown("---")
        st.subheader("Global Weight-Space Map (t-SNE)")
        st.caption(
            "Each point represents a single neuron (linear layer row) or filter (conv layer block). "
            "Colors represent layers. Points closer together have similar weight patterns."
        )

        if st.button("Compute 2D Map"):
            with st.spinner("Extracting weights and running PCA + t-SNE... (this handles padding for different layer sizes)"):
                vectors, meta = extract_weight_vectors(state_dict)

                if vectors is None:
                    st.warning("No suitable weight tensors found.")
                else:
                    # Run dimensionality reduction
                    embedding = build_2d_embedding(vectors)
                    
                    # Create DF for Plotly
                    df_map = pd.DataFrame({
                        "x": embedding[:, 0],
                        "y": embedding[:, 1],
                        "layer": [m["layer"] for m in meta],
                        "param": [m["name"] for m in meta],
                        "unit_idx": [m["unit_idx"] for m in meta],
                        "shape": [str(m["shape"]) for m in meta],
                    })

                    fig_map = px.scatter(
                        df_map, x="x", y="y", color="layer",
                        hover_data=["param", "unit_idx", "shape"],
                        title="Neural Weight Space (t-SNE)",
                        symbol="layer" # Adds distinct symbols for layers too
                    )
                    
                    # Improve aesthetics
                    fig_map.update_traces(marker=dict(size=6, opacity=0.8), selector=dict(mode='markers'))
                    fig_map.update_layout(
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        plot_bgcolor='rgba(0,0,0,0)',
                        height=600
                    )
                    st.plotly_chart(fig_map, use_container_width=True)

    else:
        st.error("Could not identify state_dict in checkpoint.")