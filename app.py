
import os
import tempfile
import cv2  
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet50_preprocess
import altair as alt

# ---------------- CONFIGURATION ----------------
IMG_SIZE = 224
MODEL_PATH = "/content/drive/MyDrive/Colab Notebooks/only_HVD/model2_hvd_resnet50_finetuned.keras"

CLASS_NAMES = ["advanced_glaucoma", "early_glaucoma", "normal_control"]

# Full names for text display
LABEL_MAP = {
    "advanced_glaucoma": "Advanced Glaucoma",
    "early_glaucoma": "Early Glaucoma",
    "normal_control": "Normal Control"
}

# Short names for the graph x-axis to prevent truncation
LABEL_MAP_SHORT = {
    "advanced_glaucoma": "Advanced",
    "early_glaucoma": "Early",
    "normal_control": "Normal"
}

# ---------------- PAGE CONFIGURATION ----------------
st.set_page_config(page_title="Glaucoma Detection", layout="wide")

# ---------------- THEME-ADAPTIVE CSS STYLING ----------------
st.markdown("""
<style>
    /* Use theme-agnostic fonts */
    [data-testid="stAppViewContainer"] {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    /* Clean Typography */
    h1, h2, h3 {
        font-weight: 300;
        letter-spacing: -0.5px;
    }
    
    /* Global Button Override (Applies to Analyse Scan) */
    div.stButton > button {
        background-color: #2c3e50 !important; 
        color: white !important;
        border-radius: 4px;
        height: 48px;
        font-weight: 600;
        letter-spacing: 0.5px;
        border: none !important;
        transition: background-color 0.2s ease;
    }
    div.stButton > button:hover {
        background-color: #1a252f !important;
    }
    div.stButton > button:disabled {
        background-color: rgba(130, 130, 130, 0.3) !important;
        color: rgba(130, 130, 130, 0.8) !important;
    }

    /* Sidebar Specific Adjustments (Reset Button) */
    [data-testid="stSidebar"] div.stButton > button {
        height: 38px !important; 
        font-size: 0.9rem !important; 
    }
    
    /* Adaptive Cards */
    .card {
        padding: 24px;
        border-radius: 8px;
        background-color: rgba(130, 130, 130, 0.05);
        border: 1px solid rgba(130, 130, 130, 0.2);
        margin-bottom: 24px;
    }
    
    /* Result Column Headers */
    .col-header {
        text-align: center;
        font-weight: 600;
        font-size: 1.1rem;
        margin-bottom: 1rem;
        letter-spacing: 1px;
        text-transform: uppercase;
        border-bottom: 2px solid rgba(130, 130, 130, 0.2);
        padding-bottom: 0.5rem;
    }
    
    /* Metrics Styling */
    .metric-label {
        font-weight: 700;
        opacity: 0.9;
    }
    .metric-value {
        opacity: 0.8;
        margin-bottom: 12px;
        line-height: 1.6;
    }
    
    /* Sidebar Disclaimer Styling */
    .sidebar-disclaimer {
        font-size: 0.75rem;
        color: #64748b;
        line-height: 1.4;
        margin-top: 2rem;
        padding-top: 1rem;
        border-top: 1px solid rgba(130, 130, 130, 0.2);
    }
</style>
""", unsafe_allow_html=True)

# ---------------- STATE MANAGEMENT ----------------
if 'analysis_results' not in st.session_state:
    st.session_state['analysis_results'] = []

# ---------------- MODEL INITIALIZATION ----------------
@st.cache_resource
def load_model_cached():
    model = load_model(MODEL_PATH)
    dummy = np.zeros((1, IMG_SIZE, IMG_SIZE, 3), dtype=np.float32)
    model.predict(dummy, verbose=0)
    return model

try:
    model = load_model_cached()
    model_loaded = True
except Exception as e:
    model_loaded = False
    st.error(f"System Error: Unable to initialize model. Details: {e}")

# ---------------- PREPROCESSING & PREDICTION ----------------
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = resnet50_preprocess(img_array)
    return img_array

def predict_image(img_path):
    img_array = preprocess_image(img_path)
    preds = model.predict(img_array, verbose=0)[0]
    pred_idx = int(np.argmax(preds))
    pred_class = CLASS_NAMES[pred_idx]
    pred_label = LABEL_MAP[pred_class]
    return pred_label, float(preds[pred_idx]), preds

# ---------------- BULLETPROOF GRAD-CAM ----------------
def generate_gradcam_heatmap(img_array, model):
    try:
        resnet_layer = None
        resnet_idx = -1
        
        for idx, layer in enumerate(model.layers):
            if isinstance(layer, tf.keras.Model) or 'resnet' in layer.name.lower():
                resnet_layer = layer
                resnet_idx = idx
                break

        inputs = tf.cast(img_array, tf.float32)

        if resnet_layer is not None:
            with tf.GradientTape() as tape:
                feature_maps = resnet_layer(inputs)
                tape.watch(feature_maps) 
                
                x = feature_maps
                for layer in model.layers[resnet_idx + 1:]:
                    x = layer(x)
                    
                preds = x
                pred_index = tf.argmax(preds[0])
                class_channel = preds[:, pred_index]

            grads = tape.gradient(class_channel, feature_maps)
        
        else:
            last_conv_layer_name = None
            for layer in reversed(model.layers):
                if len(layer.output_shape) == 4 and "conv" in layer.name.lower():
                    last_conv_layer_name = layer.name
                    break
            
            grad_model = tf.keras.models.Model(
                inputs=[model.inputs],
                outputs=[model.get_layer(last_conv_layer_name).output, model.output]
            )

            with tf.GradientTape() as tape:
                feature_maps, preds = grad_model(inputs)
                pred_index = tf.argmax(preds[0])
                class_channel = preds[:, pred_index]

            grads = tape.gradient(class_channel, feature_maps)

        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        feature_maps = feature_maps[0]
        heatmap = feature_maps @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        
        heatmap = tf.maximum(heatmap, 0)
        max_heat = tf.math.reduce_max(heatmap)
        
        if max_heat == 0:
            return np.zeros(heatmap.shape)
            
        heatmap = heatmap / max_heat
        return heatmap.numpy()
        
    except Exception as e:
        st.error(f"Grad-CAM Core Error: {str(e)}")
        return None

def apply_gradcam_overlay(img_path, heatmap, alpha=0.5):
    if heatmap is None:
        return Image.open(img_path).convert("RGB").resize((IMG_SIZE, IMG_SIZE))

    img = cv2.imread(img_path)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    superimposed_img = heatmap * alpha + img
    superimposed_img = np.clip(superimposed_img, 0, 255).astype('uint8')
    superimposed_img = cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB)
    
    return Image.fromarray(superimposed_img)

# ---------------- SIDEBAR NAVIGATION ----------------
with st.sidebar:
    st.markdown("### Configuration")
    st.markdown("---")
    
    analysis_mode = st.radio(
        "Select Input Mode",
        ["Single Image", "Multiple Images"]
    )
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    if st.button("Reset Application", use_container_width=True):
        st.session_state['analysis_results'] = []
        st.rerun()

    # Move disclaimer to the bottom of the sidebar so it's always visible but out of the way
    st.markdown("""
    <div class='sidebar-disclaimer'>
        <b>Disclaimer:</b> This application is a prototype designed for academic and research purposes only. 
        It does not constitute professional medical advice, diagnosis, or treatment.
    </div>
    """, unsafe_allow_html=True)

# ---------------- MAIN APPLICATION HEADER ----------------
st.markdown("<h1 style='text-align: center; margin-bottom: 2rem;'>GLAUCOMA DETECTION SYSTEM</h1>", unsafe_allow_html=True)

# ---------------- IMAGE INGESTION ----------------
uploaded_files = []
if analysis_mode == "Single Image":
    file = st.file_uploader("Select Fundus Image", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
    if file: 
        uploaded_files.append(file)
else:
    multi_files = st.file_uploader("Select Multiple Fundus Images", type=["jpg", "jpeg", "png"], accept_multiple_files=True, label_visibility="collapsed")
    if multi_files:
        uploaded_files = multi_files

# ---------------- ANALYSIS EXECUTION ----------------
st.markdown("<br>", unsafe_allow_html=True)

is_ready_to_analyze = len(uploaded_files) > 0 and model_loaded

col_spacer1, col_btn, col_spacer2 = st.columns([1, 2, 1])
with col_btn:
    execute_analysis = st.button(
        "Analyse Scan", 
        use_container_width=True, 
        disabled=not is_ready_to_analyze
    )

if execute_analysis:
    st.session_state['analysis_results'] = [] 
    
    with st.spinner('Processing clinical parameters and generating activation maps...'):
        for idx, file in enumerate(uploaded_files):
            
            img = Image.open(file).convert("RGB")
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
                img.save(tmp.name)
                path = tmp.name

            label, confidence, probs = predict_image(path)
            
            img_array = preprocess_image(path)
            heatmap = generate_gradcam_heatmap(img_array, model)
            gradcam_img = apply_gradcam_overlay(path, heatmap)
            
            os.remove(path)
            
            st.session_state['analysis_results'].append({
                "Filename": file.name,
                "Status": label,
                "Accuracy (%)": round(confidence * 100, 2),
                "Raw_Probs": probs,
                "Image": img,
                "GradCAM": gradcam_img 
            })

# ---------------- RENDERING RESULTS ----------------
if len(st.session_state['analysis_results']) > 0:
    st.markdown("---")
    
    # 1. SUMMARY REPORT & DOWNLOAD
    st.markdown("### SUMMARY REPORT")
    
    df_results = pd.DataFrame(st.session_state['analysis_results'])
    df_display = df_results[["Filename", "Status", "Accuracy (%)"]].copy()
    
    # Convert Accuracy to text to force left-alignment and cleanly display the % sign
    df_display["Accuracy (%)"] = df_display["Accuracy (%)"].astype(str) + " %"
    
    st.dataframe(df_display, use_container_width=True, hide_index=True)
    
    csv = df_display.to_csv(index=False).encode('utf-8')
    col_dl1, col_dl2, col_dl3 = st.columns([2, 1, 2])
    with col_dl2:
        st.download_button(
            label="Download CSV Report",
            data=csv,
            file_name="glaucoma_analysis_report.csv",
            mime="text/csv",
            use_container_width=True
        )

    st.markdown("<br><br>", unsafe_allow_html=True)
    
    # 2. DETAILED ANALYSIS
    st.markdown("### DETAILED ANALYSIS")
    
    for res in st.session_state['analysis_results']:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown(f"<div style='text-align: right; opacity: 0.6; font-size: 0.9rem;'>File: {res['Filename']}</div>", unsafe_allow_html=True)
        
        col_img, col_graph, col_answer = st.columns([1.5, 1, 1], gap="large")
        
        # Column 1: Images
        with col_img:
            st.markdown("<div class='col-header'>SCAN ANALYSIS</div>", unsafe_allow_html=True)
            
            sub_col1, sub_col2 = st.columns(2)
            with sub_col1:
                st.image(res['Image'], use_container_width=True, caption="Original Scan")
            with sub_col2:
                st.image(res['GradCAM'], use_container_width=True, caption="AI Focus Area")
        
        # Column 2: Graph
        with col_graph:
            st.markdown("<div class='col-header'>PROBABILITY</div>", unsafe_allow_html=True)
            
            prob_df = pd.DataFrame({
                "Diagnosis": [LABEL_MAP_SHORT[c] for c in CLASS_NAMES],
                "Accuracy Level": res['Raw_Probs']
            })
            
            chart = alt.Chart(prob_df).mark_bar(size=40, color="#2c3e50").encode(
                x=alt.X('Diagnosis', sort=None, axis=alt.Axis(labelAngle=0, title="")),
                y=alt.Y('Accuracy Level', scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format='%')),
                tooltip=['Diagnosis', alt.Tooltip('Accuracy Level', format='.2%')]
            ).properties(height=260) 
            
            st.altair_chart(chart, use_container_width=True, theme="streamlit")

        # Column 3: Assessment
        with col_answer:
            st.markdown("<div class='col-header'>CLINICAL ASSESSMENT</div>", unsafe_allow_html=True)
            
            st.markdown(f"<span class='metric-label'>Status:</span> <span class='metric-value'>{res['Status']}</span>", unsafe_allow_html=True)
            st.markdown(f"<span class='metric-label'>Accuracy:</span> <span class='metric-value'>The predicted image corresponds to {res['Status']} with an accuracy of {res['Accuracy (%)']}.</span>", unsafe_allow_html=True)
            
        st.markdown('</div>', unsafe_allow_html=True)