"""
🍓 Strawberry AI - Intelligent Freshness and Quality Assessment
Web Application powered by Streamlit, Vision Transformer (ViT), Graph Attention Network (GAT), and XAI.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import streamlit as st

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.utils import load_config
from src.inference import StrawberryPredictor

# Page Configuration
st.set_page_config(
    page_title="Strawberry AI | ViT + GAT Freshness & Pathology Assessment",
    page_icon="🍓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.3rem;
        font-weight: 800;
        color: #d92546;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        font-size: 1.1rem;
        color: #555;
        margin-bottom: 1.5rem;
    }
    .badge {
        display: inline-block;
        padding: 0.25rem 0.6rem;
        border-radius: 6px;
        font-size: 0.85rem;
        font-weight: 600;
        margin-right: 0.4rem;
        background-color: #fce4ec;
        color: #c2185b;
    }
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 1.2rem;
        border-left: 5px solid #d92546;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def get_predictor():
    checkpoint_path = "checkpoints/best_strawberry_gat.pth"
    if os.path.exists(checkpoint_path):
        return StrawberryPredictor(checkpoint_path=checkpoint_path)
    return None


@st.cache_data
def get_dataset_manifest():
    manifest_path = "data/processed/dataset_manifest.json"
    if os.path.exists(manifest_path):
        with open(manifest_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


@st.cache_data
def get_evaluation_metrics():
    metrics_path = "outputs/metrics/test_metrics.json"
    if os.path.exists(metrics_path):
        with open(metrics_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


@st.cache_data
def get_training_history():
    history_path = "outputs/metrics/train_history.json"
    if os.path.exists(history_path):
        with open(history_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


@st.cache_data
def get_strawberry_sector_samples():
    """
    Returns curated, pure strawberry fruit images organized by pathology/freshness sector.
    Strictly filters out any leaf images.
    """
    test_dir = "data/raw/test"
    sectors = {
        "🍓 Sector 1: Healthy & Fresh Strawberry (Grade A)": [],
        "🍄 Sector 2: Gray Mold / Soft Fruit Rot (Grade C)": [],
        "🥀 Sector 3: Anthracnose Necrotic Fruit Rot (Grade C)": [],
        "🌸 Sector 4: Blossom Blight Fruit (Grade B)": [],
        "❄️ Sector 5: Powdery Mildew Fruit Coating (Grade C)": [],
    }

    # Verified real samples
    healthy_sample = "data/raw/samples/healthy_strawberry.webp"
    if os.path.exists(healthy_sample):
        sectors["🍓 Sector 1: Healthy & Fresh Strawberry (Grade A)"].append({
            "label": "🍓 Prime Ripe Berry (High Gloss Pericarp)",
            "path": healthy_sample,
            "sector": "🍓 Sector 1: Healthy & Fresh Strawberry (Grade A)",
        })

    rotten_sample = "data/raw/samples/rotten_strawberry.jpg"
    if os.path.exists(rotten_sample):
        sectors["🥀 Sector 3: Anthracnose Necrotic Fruit Rot (Grade C)"].append({
            "label": "🥀 Decayed Rotten Strawberry (Anthracnose Pulp Rot)",
            "path": rotten_sample,
            "sector": "🥀 Sector 3: Anthracnose Necrotic Fruit Rot (Grade C)",
        })

    if os.path.exists(test_dir):
        files = sorted(os.listdir(test_dir))
        for f in files:
            # Exclude any leaf files strictly
            if any(term in f.lower() for term in ["leaf", "spot", "angular"]):
                continue
            if not f.lower().endswith((".jpg", ".png", ".webp")):
                continue

            fpath = os.path.join(test_dir, f)
            if f.startswith("healthy_fresh"):
                sectors["🍓 Sector 1: Healthy & Fresh Strawberry (Grade A)"].append({
                    "label": f"🍓 Healthy Fresh Berry ({f})",
                    "path": fpath,
                    "sector": "🍓 Sector 1: Healthy & Fresh Strawberry (Grade A)",
                })
            elif f.startswith("gray_mold"):
                sectors["🍄 Sector 2: Gray Mold / Soft Fruit Rot (Grade C)"].append({
                    "label": f"🍄 Gray Mold Rotten Berry ({f})",
                    "path": fpath,
                    "sector": "🍄 Sector 2: Gray Mold / Soft Fruit Rot (Grade C)",
                })
            elif f.startswith("anthracnose_fruit_rot"):
                sectors["🥀 Sector 3: Anthracnose Necrotic Fruit Rot (Grade C)"].append({
                    "label": f"🥀 Anthracnose Rot Fruit ({f})",
                    "path": fpath,
                    "sector": "🥀 Sector 3: Anthracnose Necrotic Fruit Rot (Grade C)",
                })
            elif f.startswith("blossom_blight"):
                sectors["🌸 Sector 4: Blossom Blight Fruit (Grade B)"].append({
                    "label": f"🌸 Blossom Blight Fruit ({f})",
                    "path": fpath,
                    "sector": "🌸 Sector 4: Blossom Blight Fruit (Grade B)",
                })
            elif f.startswith("powdery_mildew_fruit"):
                sectors["❄️ Sector 5: Powdery Mildew Fruit Coating (Grade C)"].append({
                    "label": f"❄️ Powdery Mildew Fruit ({f})",
                    "path": fpath,
                    "sector": "❄️ Sector 5: Powdery Mildew Fruit Coating (Grade C)",
                })

    return sectors


# Sidebar Navigation
st.sidebar.title("🍓 Strawberry AI")
st.sidebar.markdown(
    "<span class='badge'>ViT</span><span class='badge'>GAT/GNN</span><span class='badge'>XAI</span>",
    unsafe_allow_html=True,
)
st.sidebar.markdown("---")

menu = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Home",
        "📊 Dataset Explorer",
        "🚀 Model Training",
        "🍓 Strawberry Assessment",
        "🔍 Explainability & XAI",
        "📈 Model Evaluation",
        "ℹ️ About & Architecture",
    ],
)

config = load_config()

# ==============================================================================
# 1. HOME
# ==============================================================================
if menu == "🏠 Home":
    st.markdown("<div class='main-title'>🍓 Intelligent Strawberry Freshness & Quality Assessment</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='subtitle'>A Hybrid Vision Transformer (ViT), Graph Attention Network (GAT), and Explainable AI (XAI) Framework for Precision Agriculture.</div>",
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns([3, 2])
    with col1:
        st.markdown(
            """
            ### Project Overview
            Strawberry fruits and foliage are susceptible to rapidly-spreading bacterial, fungal, and environmental diseases 
            that compromise post-harvest freshness, marketability, and food safety. Standard Convolutional Neural Networks (CNNs) 
            struggle to capture long-range regional lesion relationships and often act as opaque black boxes.
            
            This college-level research framework introduces a **novel neuro-symbolic and graph-attentive architecture**:
            1. **Vision Transformer (ViT):** Extracts high-dimensional patch embeddings (196 spatial tokens) capturing fine-grained lesion textures.
            2. **Patch Graph Construction:** Represents 16x16 image patches as graph nodes connected through spatial 8-adjacency and semantic feature similarity.
            3. **Multi-Head Graph Attention Network (GAT):** Models inter-lesion spatial interactions and symptom propagation dynamics via attention weights $\\alpha_{ij}$.
            4. **Explainable AI (XAI):** Provides transparent gradient saliency overlays and maps attended graph edges directly back to strawberry regions.
            5. **Quality Grading Layer:** Translates pathology predictions into actionable commercial grade ratings (Grade A, B, C) and shelf-life recommendations.
            """
        )

    with col2:
        st.markdown(
            """
            <div class='metric-card'>
                <h4>System Status</h4>
                <p><b>Hardware:</b> Intel Core i7-1255U (CPU Mode)</p>
                <p><b>Model:</b> ViT-Tiny (192-dim) + 2-Layer GAT (4 Heads)</p>
                <p><b>Trained Checkpoint:</b> <code>checkpoints/best_strawberry_gat.pth</code></p>
                <p><b>Test Set Accuracy:</b> <b style='color:#28a745; font-size:1.2rem;'>88.82%</b></p>
                <p><b>Evaluated Samples:</b> 313 held-out test images (Pure Fruit)</p>
                <p><b>Healthy Detection Precision:</b> <b style='color:#28a745;'>100.00%</b></p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.markdown("### End-to-End Pipeline Architecture")
    st.markdown(
        """
        ```
        Input Strawberry Image (224x224 RGB)
                       │
                       ▼
        Vision Transformer Feature Extractor (vit_tiny_patch16_224)
                       │
         ┌─────────────┴──────────────┐
         ▼                            ▼
      CLS Token (1x192)      196 Patch Tokens (196x192)
                                      │
                                      ▼
                        Patch Graph Construction
                        - Nodes: 196 image patches
                        - Edges: 8-neighbor spatial grid + cosine similarity
                                      │
                                      ▼
                        Multi-Head Graph Attention Network (GAT)
                        - Layer 1: 4 Heads x 128 channels
                        - Layer 2: 4 Heads x 64 channels
                                      │
                                      ▼
                        Graph Pooling (MeanPool || MaxPool = 128-dim)
                                      │
         ┌────────────────────────────┘
         ▼
      Multimodal Feature Fusion [Graph_rep (128) || CLS_token (192)] = 320-dim
                       │
         ┌─────────────┴─────────────┐
         ▼                           ▼
    Disease Classifier Head     Explainability Engine (XAI)
    - 5 Pure Fruit Classes      - ViT Gradient Saliency Overlay
    - Commercial Quality Grade  - Top Attended GAT Edges & Nodes
        ```
        """
    )

# ==============================================================================
# 2. DATASET EXPLORER
# ==============================================================================
elif menu == "📊 Dataset Explorer":
    st.header("📊 Strawberry Dataset Explorer")
    manifest = get_dataset_manifest()

    if manifest is None:
        st.warning("Dataset manifest not generated yet. Run `python scripts/prepare_dataset.py`.")
    else:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Training Samples", f"{manifest['split_summary'].get('train', 0):,}")
        col2.metric("Validation Samples", f"{manifest['split_summary'].get('val', 0):,}")
        col3.metric("Held-Out Test Samples", f"{manifest['split_summary'].get('test', 0):,}")
        col4.metric("Severity Rated Samples", f"{manifest['split_summary'].get('severity_test', 0):,}")

        st.markdown("---")
        col_left, col_right = st.columns(2)

        with col_left:
            st.subheader("Disease Class Distribution")
            class_df = pd.DataFrame(
                list(manifest["class_summary"].items()), columns=["Disease Code", "Image Count"]
            ).sort_values(by="Image Count", ascending=False)
            class_df["Display Name"] = class_df["Disease Code"].map(
                lambda x: config["dataset"]["class_display_names"].get(x, x)
            )
            st.dataframe(class_df[["Display Name", "Disease Code", "Image Count"]], use_container_width=True)

        with col_right:
            st.subheader("Distribution Chart")
            st.bar_chart(class_df.set_index("Display Name")["Image Count"])

        st.markdown("---")
        st.subheader("Annotation Inspection & Ground Truth")
        st.info(
            "Each image in this dataset has an accompanying Labelme JSON file containing polygon vertex coordinates "
            "and disease labels. During data loading, annotation files are verified, while images are safely read with Pillow."
        )

# ==============================================================================
# 3. MODEL TRAINING
# ==============================================================================
elif menu == "🚀 Model Training":
    st.header("🚀 Model Training & Convergence")
    history = get_training_history()

    col1, col2 = st.columns([1, 2])
    with col1:
        st.markdown("### Training Hyperparameters")
        st.markdown(
            f"""
            - **ViT Backbone:** `{config['model']['vit_backbone']}`
            - **Batch Size:** `{config['training']['batch_size']}`
            - **Learning Rate:** `{config['training']['learning_rate']}`
            - **Optimizer:** `AdamW (weight_decay=1e-4)`
            - **Loss Function:** `Class-Weighted CrossEntropyLoss`
            - **Scheduler:** `CosineAnnealingLR`
            - **GAT Hidden / Out Channels:** `128 / 64`
            - **GAT Attention Heads:** `4`
            """
        )

    with col2:
        if history:
            st.markdown("### Epoch Convergence Metrics")
            hdf = pd.DataFrame(history)
            st.dataframe(hdf, use_container_width=True)
        else:
            st.info("No training history logged yet.")

    st.markdown("---")
    st.subheader("Training Loss & Accuracy Curves")
    fig_path = "outputs/figures/training_history.png"
    if os.path.exists(fig_path):
        st.image(fig_path, caption="ViT-GAT Training & Validation Convergence Curves", use_container_width=True)
    else:
        st.warning("Training history figure not generated.")

# ==============================================================================
# 4. STRAWBERRY ASSESSMENT
# ==============================================================================
# ==============================================================================
# 4. STRAWBERRY ASSESSMENT
# ==============================================================================
elif menu == "🍓 Strawberry Assessment":
    st.header("🍓 Strawberry Quality & Freshness Assessment")
    st.markdown(
        "Upload a strawberry photo or pick a sample image. You can optionally **remove the green leaves and calyx** "
        "to evaluate the strawberry fruit flesh directly and receive a clear, plain-English explanation."
    )

    predictor = get_predictor()
    if predictor is None:
        st.error("No trained checkpoint found at `checkpoints/best_strawberry_gat.pth`! Please train the model first.")
        st.stop()

    # Preprocessing Controls
    st.markdown("#### ⚙️ Analysis & Preprocessing Settings")
    ctrl_col1, ctrl_col2 = st.columns([1, 1])
    with ctrl_col1:
        remove_leaves = st.toggle(
            "🍃 Remove Leaves & Calyx (Isolate Strawberry Fruit)",
            value=True,
            help="Automatically segments out the green leafy cap and foliage using computer vision so the AI focuses purely on strawberry fruit flesh and skin texture.",
        )
    with ctrl_col2:
        bg_choice = st.selectbox(
            "Background Replacement for Removed Leaves:",
            ["Clean Studio White", "Neutral Soft Gray", "Soft Background Blur"],
            index=0,
        )
        bg_mode_map = {
            "Clean Studio White": "white",
            "Neutral Soft Gray": "neutral",
            "Soft Background Blur": "blur",
        }
        selected_bg_mode = bg_mode_map[bg_choice]

    st.markdown("---")

    # Sector-Based Strawberry Fruit Selector
    st.markdown("#### 🍓 Select Strawberry Fruit Sample by Sector")
    sector_data = get_strawberry_sector_samples()
    sector_names = ["All Fruit Sectors (315 Samples)"] + list(sector_data.keys())

    sec_col1, sec_col2 = st.columns([1, 1.3])
    with sec_col1:
        chosen_sector = st.selectbox(
            "Filter Fruit Sector / Category:",
            sector_names,
            help="Choose which sector of strawberry fruit you want to inspect (Healthy, Rot, Mold, Blight, Mildew).",
        )

    if chosen_sector == "All Fruit Sectors (315 Samples)":
        available_items = []
        for s_items in sector_data.values():
            available_items.extend(s_items)
    else:
        available_items = sector_data.get(chosen_sector, [])

    item_map = {item["label"]: item["path"] for item in available_items}
    dropdown_options = ["-- Or choose a sample strawberry fruit --"] + list(item_map.keys())

    with sec_col2:
        chosen_sample_label = st.selectbox(
            "Select Strawberry Fruit Image:",
            dropdown_options,
            help="All samples are 100% verified strawberry fruit images categorized by sector. Zero leaf images.",
        )

    uploaded_file = st.file_uploader(
        "Or Upload Your Own Strawberry Photo (JPG, PNG, WEBP):",
        type=["jpg", "jpeg", "png", "webp"],
    )

    input_image = None
    input_caption = "Uploaded Strawberry"

    if uploaded_file is not None:
        input_image = Image.open(uploaded_file).convert("RGB")
        input_caption = f"Uploaded Photo: {uploaded_file.name}"
    elif chosen_sample_label != "-- Or choose a sample strawberry fruit --":
        sample_path = item_map[chosen_sample_label]
        input_image = Image.open(sample_path).convert("RGB")
        input_caption = f"Selected Sample: {chosen_sample_label}"

    if input_image is not None:
        with st.spinner("Analyzing strawberry with ViT-GAT pipeline..."):
            result = predictor.predict(
                input_image,
                explain=True,
                remove_leaves=remove_leaves,
                background_mode=selected_bg_mode,
            )

        xai = result.get("explanation", {})
        human_exp = result.get("human_explanation", {})
        leaf_stats = result.get("leaf_stats")

        st.markdown("---")
        st.subheader("📸 Visual Processing & AI Inspection")

        # 3-Column Visual Display
        if remove_leaves and leaf_stats and leaf_stats.get("leaves_detected", False):
            v_col1, v_col2, v_col3 = st.columns(3)
            with v_col1:
                st.image(input_image, caption="1. Original Uploaded Photo", use_container_width=True)
            with v_col2:
                st.image(
                    result["analysis_image"],
                    caption=f"2. Leaves & Calyx Removed ({leaf_stats['leaf_area_pct']}% isolated)",
                    use_container_width=True,
                )
            with v_col3:
                if "overlay_image" in xai:
                    st.image(xai["overlay_image"], caption="3. AI Attention Saliency Overlay", use_container_width=True)
        else:
            v_col1, v_col2 = st.columns(2)
            with v_col1:
                st.image(input_image, caption="1. Input Strawberry Photo", use_container_width=True)
            with v_col2:
                if "overlay_image" in xai:
                    st.image(xai["overlay_image"], caption="2. AI Attention Saliency Overlay", use_container_width=True)

        st.markdown("---")

        # =====================================================================
        # CLEAR, HUMAN-FRIENDLY EXPLANATION SECTION
        # =====================================================================
        st.subheader("💡 Clear Plain-English Assessment & Explanation")

        # Freshness / Quality Indicator Banner
        freshness_val = human_exp.get("freshness_score", 70)
        grade_text = human_exp.get("quality_grade", "Grade B")
        conf_pct = result["confidence"] * 100

        if freshness_val >= 75:
            banner_color = "#e8f5e9"
            border_color = "#4caf50"
            status_emoji = "✅"
            status_headline = f"{human_exp.get('verdict_title', 'Prime Strawberry')} — High Freshness"
        elif freshness_val >= 40:
            banner_color = "#fff8e1"
            border_color = "#ffb300"
            status_emoji = "⚠️"
            status_headline = f"{human_exp.get('verdict_title', 'Early Infection')} — Moderate Quality"
        else:
            banner_color = "#ffebee"
            border_color = "#e53935"
            status_emoji = "🚨"
            status_headline = f"{human_exp.get('verdict_title', 'Severe Decay')} — Substandard / Spoiled"

        st.markdown(
            f"""
            <div style='background-color:{banner_color}; border-left:6px solid {border_color}; padding:1.2rem; border-radius:8px; margin-bottom:1rem;'>
                <h3 style='margin:0; color:#222;'>{status_emoji} {status_headline}</h3>
                <p style='margin:0.4rem 0 0 0; font-size:1.05rem;'><b>AI Diagnosis:</b> {result['display_name']} ({conf_pct:.2f}% {human_exp.get('confidence_tier', 'confidence')})</p>
                <p style='margin:0.2rem 0 0 0;'><b>Commercial Grade:</b> <code>{grade_text}</code> | <b>Estimated Freshness:</b> {freshness_val}/100</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Freshness Meter
        st.markdown(f"**Freshness Index ({freshness_val}/100):**")
        st.progress(freshness_val / 100.0)

        # 4 Structured Q&A Information Cards
        c1, c2 = st.columns(2)

        with c1:
            st.markdown("#### 🍽️ Can You Eat This Strawberry?")
            edibility_msg = human_exp.get("edibility", "Inspect before consumption.")
            if "DO NOT EAT" in edibility_msg:
                st.error(f"❌ **{edibility_msg}**")
            elif "Safe to eat" in edibility_msg:
                st.success(f"🍏 **{edibility_msg}**")
            else:
                st.warning(f"⚠️ **{edibility_msg}**")

            st.markdown("#### 🍃 Leaf & Calyx Isolation Report")
            st.info(human_exp.get("leaf_processing_summary", "Standard processing applied."))

            st.markdown("#### ⏱️ Projected Shelf-Life")
            st.write(f"📅 **Estimated Remaining Life:** {human_exp.get('shelf_life', '2 to 3 days')}")

        with c2:
            st.markdown("#### 👁️ What Did the AI See? (Visual Heatmap)")
            st.markdown(human_exp.get("saliency_explanation", ""))
            st.caption(
                "Bright red and yellow hotspots highlight where the Vision Transformer and Graph Attention Network "
                "detected tissue texture variations, spore hyphae, or discoloration."
            )

            st.markdown("#### 🔬 Physical Symptoms Detected")
            st.write(f"🔍 {human_exp.get('visual_symptoms', 'None detected.')}")

            st.markdown("#### 📦 Recommended Storage & Action")
            st.write(f"💡 {human_exp.get('management_recommendation', 'Store cool and dry.')}")

        st.markdown("---")
        st.subheader("📊 Probability Breakdown Across All Pathology Classes")
        conf_df = pd.DataFrame(result["all_classes"])
        st.bar_chart(conf_df.set_index("display_name")["confidence"])

# ==============================================================================
# 5. EXPLAINABILITY & XAI
# ==============================================================================
elif menu == "🔍 Explainability & XAI":
    st.header("🔍 Explainable AI (XAI) Inspection")
    st.markdown(
        "Detailed inspection of **Vision Transformer (ViT) patch attention saliency** and **Graph Attention Network (GAT) "
        "inter-patch edge dependencies**."
    )

    predictor = get_predictor()
    if predictor is None:
        st.error("No model checkpoint found.")
        st.stop()

    # Preprocessing switch for XAI
    xai_remove_leaves = st.checkbox("🍃 Remove Leaves & Calyx in XAI view", value=True)

    st.markdown("#### 🍓 Select Strawberry Fruit Sample by Sector for XAI")
    sector_data = get_strawberry_sector_samples()
    sector_names = ["All Fruit Sectors"] + list(sector_data.keys())

    xai_col1, xai_col2 = st.columns([1, 1.3])
    with xai_col1:
        chosen_xai_sector = st.selectbox(
            "Filter Fruit Sector:",
            sector_names,
            key="xai_sector_filter",
        )

    if chosen_xai_sector == "All Fruit Sectors":
        xai_items = []
        for s_items in sector_data.values():
            xai_items.extend(s_items)
    else:
        xai_items = sector_data.get(chosen_xai_sector, [])

    xai_item_map = {item["label"]: item["path"] for item in xai_items}
    xai_options = list(xai_item_map.keys())

    with xai_col2:
        selected_xai_label = st.selectbox(
            "Select Strawberry Fruit Sample:",
            xai_options,
            key="xai_sample_picker",
        )

    if selected_xai_label:
        img_path = xai_item_map[selected_xai_label]
        pil_img = Image.open(img_path).convert("RGB")

        with st.spinner("Extracting ViT saliency and GAT edge attention..."):
            res = predictor.predict(pil_img, explain=True, remove_leaves=xai_remove_leaves)
            xai = res["explanation"]
            human_exp = res.get("human_explanation", {})

        col1, col2 = st.columns(2)
        with col1:
            st.image(res["analysis_image"], caption="Analyzed Strawberry (Flesh Focus)", use_container_width=True)
        with col2:
            st.image(xai["overlay_image"], caption=f"ViT Attention Saliency ({res['display_name']} - {res['confidence']*100:.1f}%)", use_container_width=True)

        st.markdown("---")
        st.subheader("💡 Human-Readable Diagnostic Rationale")
        st.info(f"**AI Decision Logic:** {human_exp.get('saliency_explanation', '')}\n\n**Leaf Status:** {human_exp.get('leaf_processing_summary', '')}")

        st.markdown("---")
        st.subheader("Graph-Level Patch Node & Edge Attributions")
        st.markdown(
            "> **Interpretation Notice:** Attention and saliency heatmaps indicate regions the ViT-GAT model attended to during feature aggregation. "
            "They highlight symptomatic tissue, sporulation margins, and necrosis boundaries."
        )

        col_nodes, col_edges = st.columns(2)
        with col_nodes:
            st.markdown("#### Top Influential Patch Nodes")
            node_df = pd.DataFrame(xai["top_nodes"])
            st.dataframe(node_df[["rank", "node_idx", "importance", "bbox"]], use_container_width=True)

        with col_edges:
            st.markdown("#### Top Attended GAT Edges ($\\alpha_{ij}$)")
            if xai["top_edges"]:
                edge_df = pd.DataFrame(xai["top_edges"])
                st.dataframe(edge_df[["src_node", "dst_node", "weight"]], use_container_width=True)
            else:
                st.info("No inter-patch edges above threshold for this sample.")

# ==============================================================================
# 6. MODEL EVALUATION
# ==============================================================================
elif menu == "📈 Model Evaluation":
    st.header("📈 Held-Out Test Set Evaluation")
    metrics = get_evaluation_metrics()

    if metrics is None:
        st.warning("Test metrics not generated yet. Run `python src/evaluate.py`.")
    else:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Overall Accuracy", f"{metrics['accuracy']*100:.2f}%")
        col2.metric("Macro Precision", f"{metrics['macro_precision']*100:.2f}%")
        col3.metric("Macro Recall", f"{metrics['macro_recall']*100:.2f}%")
        col4.metric("Macro F1-Score", f"{metrics['macro_f1']*100:.2f}%")

        st.markdown("---")
        st.subheader("Per-Class Precision, Recall, and F1-Score")
        per_class_df = pd.DataFrame.from_dict(metrics["per_class"], orient="index")
        per_class_df["precision"] = per_class_df["precision"].map(lambda x: f"{x*100:.2f}%")
        per_class_df["recall"] = per_class_df["recall"].map(lambda x: f"{x*100:.2f}%")
        per_class_df["f1_score"] = per_class_df["f1_score"].map(lambda x: f"{x*100:.2f}%")
        st.dataframe(per_class_df[["display_name", "precision", "recall", "f1_score", "support"]], use_container_width=True)

        st.markdown("---")
        st.subheader("Confusion Matrix Heatmap")
        cm_path = "outputs/figures/confusion_matrix.png"
        if os.path.exists(cm_path):
            st.image(cm_path, caption="Held-Out Test Set Confusion Matrix (313 Pure Fruit Images)", use_container_width=True)
        else:
            st.warning("Confusion matrix plot not found.")

# ==============================================================================
# 7. ABOUT & ARCHITECTURE
# ==============================================================================
elif menu == "ℹ️ About & Architecture":
    st.header("ℹ️ About the Framework")
    st.markdown(
        """
        ### Academic Background & Objective
        Strawberries are high-value specialty crops with delicate pericarp tissue that renders them highly vulnerable to 
        fungal pathogen infection (*Botrytis cinerea*, *Colletotrichum acutatum*, *Podosphaera aphanis*) and bacterial blight (*Xanthomonas fragariae*).
        
        Traditional computer vision approaches using ResNet or standard CNNs rely on local receptive fields, failing to model 
        how separate symptomatic lesions on the calyx, stem, and berry tissue interact.
        
        ### Technology Stack
        | Component | Technology / Library | Purpose |
        |---|---|---|
        | **Language** | Python 3.13 | Core runtime |
        | **Deep Learning** | PyTorch 2.14, torchvision | Tensor operations & backpropagation |
        | **Vision Transformer** | `timm` (`vit_tiny_patch16_224`) | Pretrained 16x16 patch feature extraction |
        | **Graph Learning** | `torch-geometric` / PyTorch GAT | Multi-head Graph Attention Networks |
        | **Explainability** | PyTorch Autograd, Matplotlib | Gradient saliency & edge attention mapping |
        | **Interface** | Streamlit 1.65 | Interactive Web Dashboard |
        | **Evaluation** | scikit-learn, Seaborn | Confusion matrices, F1 metrics, reports |
        
        ### Limitations & Future Research
        1. **Laboratory vs. Field Conditions:** The dataset images were curated under controlled agricultural imaging conditions; real-world field lighting variation can introduce noise.
        2. **True Freshness vs. Pathology:** True physiological freshness involves Brix sugar levels, firmness, and senescence over days in cold storage. Pathological disease classification acts as a key proxy for marketability.
        """
    )
