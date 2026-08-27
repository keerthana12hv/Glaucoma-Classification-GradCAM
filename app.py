import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet50_preprocess
from PIL import Image
import tempfile
import os
from matplotlib import colormaps

# ---------------- CONFIG ----------------
IMG_SIZE = 224

PHASE1_MODEL_PATH = "/content/drive/MyDrive/Colab Notebooks/Major/glaucoma_model_v2.keras"
PHASE2_MODEL_PATH = "/content/drive/MyDrive/Colab Notebooks/Model2_HVD/hvd_model2_workspace/model2_hvd_resnet50_finetuned.keras"

PHASE1_LABELS = {0: "Glaucoma", 1: "Normal"}

PHASE2_CLASS_ORDER = ["advanced_glaucoma", "early_glaucoma", "normal_control"]
PHASE2_LABELS = {
    "advanced_glaucoma": "Advanced Glaucoma",
    "early_glaucoma": "Early Glaucoma",
    "normal_control": "Normal Control"
}

GRADCAM_LAYER_NAME = "conv5_block3_out"

# ---------------- PAGE ----------------
st.set_page_config(page_title="Glaucoma Detection System", layout="wide")
st.title("Glaucoma Detection and Staging System")
st.write("Two-model glaucoma analysis using fundus images")
st.markdown("---")

# ---------------- LOAD MODELS ----------------
@st.cache_resource
def load_models():
    phase1_model = load_model(PHASE1_MODEL_PATH)
    phase2_model = load_model(PHASE2_MODEL_PATH)

    # Build models once so inference graph is ready
    _ = phase1_model(tf.zeros((1, IMG_SIZE, IMG_SIZE, 3)))
    _ = phase2_model(tf.zeros((1, IMG_SIZE, IMG_SIZE, 3)))

    return phase1_model, phase2_model

phase1_model, phase2_model = load_models()
phase2_base_model = phase2_model.layers[0]

# ---------------- PREPROCESSING ----------------
def preprocess_for_phase1(img_path):
    img = image.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def preprocess_for_phase2(img_path):
    img = image.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = resnet50_preprocess(img_array)
    return img_array

# ---------------- PHASE 1 ----------------
def predict_phase1(img_path, threshold=0.5):
    img_array = preprocess_for_phase1(img_path)
    pred = phase1_model.predict(img_array, verbose=0)[0][0]

    normal_probability = float(pred)
    glaucoma_probability = float(1 - pred)

    if pred >= threshold:
        predicted_index = 1
    else:
        predicted_index = 0

    predicted_label = PHASE1_LABELS[predicted_index]

    return {
        "phase": "phase1",
        "predicted_index": predicted_index,
        "predicted_label": predicted_label,
        "normal_probability": normal_probability,
        "glaucoma_probability": glaucoma_probability
    }

# ---------------- PHASE 2 ----------------
def predict_phase2(img_path):
    img_array = preprocess_for_phase2(img_path)
    preds = phase2_model.predict(img_array, verbose=0)[0]

    pred_idx = int(np.argmax(preds))
    pred_class = PHASE2_CLASS_ORDER[pred_idx]
    pred_label = PHASE2_LABELS[pred_class]

    return {
        "phase": "phase2",
        "predicted_index": pred_idx,
        "predicted_class_folder": pred_class,
        "predicted_label": pred_label,
        "confidence": float(preds[pred_idx]),
        "probabilities": {
            PHASE2_CLASS_ORDER[i]: float(preds[i]) for i in range(len(PHASE2_CLASS_ORDER))
        }
    }

# ---------------- GRAD-CAM ----------------
def build_classifier_head(model, base_model):
    classifier_input = tf.keras.Input(shape=base_model.output.shape[1:])
    x = classifier_input
    for layer in model.layers[1:]:
        x = layer(x)
    classifier_model = tf.keras.Model(classifier_input, x)
    return classifier_model

def make_gradcam_heatmap(img_array, phase2_model, phase2_base_model, last_conv_layer_name, pred_index=None):
    last_conv_layer = phase2_base_model.get_layer(last_conv_layer_name)

    conv_base_model = tf.keras.Model(
        inputs=phase2_base_model.input,
        outputs=[last_conv_layer.output, phase2_base_model.output]
    )

    classifier_model = build_classifier_head(phase2_model, phase2_base_model)

    with tf.GradientTape() as tape:
        conv_outputs, base_outputs = conv_base_model(img_array)
        tape.watch(conv_outputs)

        preds = classifier_model(base_outputs)

        if pred_index is None:
            pred_index = tf.argmax(preds[0])

        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)

    heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()

def generate_gradcam_overlay(img_path):
    img_array = preprocess_for_phase2(img_path)

    heatmap = make_gradcam_heatmap(
        img_array=img_array,
        phase2_model=phase2_model,
        phase2_base_model=phase2_base_model,
        last_conv_layer_name=GRADCAM_LAYER_NAME
    )

    original = Image.open(img_path).convert("RGB")
    original_np = np.array(original)

    heatmap_uint8 = np.uint8(255 * heatmap)
    jet = colormaps["jet"]
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap_uint8]
    jet_heatmap = Image.fromarray(np.uint8(jet_heatmap * 255))
    jet_heatmap = jet_heatmap.resize((original_np.shape[1], original_np.shape[0]))
    jet_heatmap = np.array(jet_heatmap)

    overlay = np.clip(jet_heatmap * 0.35 + original_np, 0, 255).astype("uint8")
    return original_np, overlay, heatmap

# ---------------- DECISION FUSION ----------------
def combine_decisions(phase1_result, phase2_result):
    p1_label = phase1_result["predicted_label"]
    p2_folder = phase2_result["predicted_class_folder"]
    p2_label = phase2_result["predicted_label"]

    # Case 1: both models support normal
    if p1_label == "Normal" and p2_folder == "normal_control":
        return {
            "status": "Normal",
            "message": "Both models support a normal finding.",
            "confidence": max(phase1_result["normal_probability"], phase2_result["confidence"]),
            "result_type": "normal"
        }

    # Case 2: both support glaucoma and phase 2 gives stage
    if p1_label == "Glaucoma" and p2_folder in ["early_glaucoma", "advanced_glaucoma"]:
        return {
            "status": p2_label,
            "message": f"Phase 1 detected glaucoma, and Phase 2 classified it as {p2_label}.",
            "confidence": phase2_result["confidence"],
            "result_type": "glaucoma_stage"
        }

    # Case 3: phase 1 says glaucoma but phase 2 says normal_control
    if p1_label == "Glaucoma" and p2_folder == "normal_control":
        return {
            "status": "Glaucoma Suspected - Stage Uncertain",
            "message": "Phase 1 detected glaucoma, but Phase 2 returned normal_control. This may indicate a borderline or uncertain case.",
            "confidence": max(phase1_result["glaucoma_probability"], phase2_result["confidence"]),
            "result_type": "uncertain"
        }

    # Case 4: phase 1 says normal but phase 2 says glaucoma class
    if p1_label == "Normal" and p2_folder in ["early_glaucoma", "advanced_glaucoma"]:
        return {
            "status": f"{p2_label} (Review Required)",
            "message": f"Phase 1 predicted Normal, but Phase 2 detected {p2_label}. This disagreement suggests the case should be reviewed carefully.",
            "confidence": phase2_result["confidence"],
            "result_type": "review_required"
        }

    # Fallback
    return {
        "status": "Uncertain",
        "message": "The two models produced an uncertain result.",
        "confidence": phase2_result["confidence"],
        "result_type": "uncertain"
    }

# ---------------- PIPELINE ----------------
def predict_glaucoma_pipeline(img_path, phase1_threshold=0.5):
    phase1_result = predict_phase1(img_path, threshold=phase1_threshold)
    phase2_result = predict_phase2(img_path)

    final_decision = combine_decisions(phase1_result, phase2_result)

    final_result = {
        "image_path": img_path,
        "phase1_result": phase1_result,
        "phase2_result": phase2_result,
        "final_decision": final_decision
    }

    return final_result

# ---------------- UI ----------------
uploaded_file = st.file_uploader(
    "Upload a retinal fundus image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    col1, col2 = st.columns([1, 1])

    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_file:
        pil_img = Image.open(uploaded_file).convert("RGB")
        pil_img.save(tmp_file.name)
        temp_path = tmp_file.name

    with col1:
        st.image(pil_img, caption="Uploaded Fundus Image", use_container_width=True)

    with col2:
        if st.button("Run Glaucoma Analysis", use_container_width=True):
            with st.spinner("Analyzing image..."):
                result = predict_glaucoma_pipeline(temp_path)

            final_status = result["final_decision"]["status"]
            confidence = result["final_decision"]["confidence"]
            result_type = result["final_decision"]["result_type"]

            st.subheader("Final Result")

            if result_type == "normal":
                st.success(f"Prediction: {final_status}")
            elif result_type in ["uncertain", "review_required"]:
                st.warning(f"Prediction: {final_status}")
            else:
                st.error(f"Prediction: {final_status}")

            st.info(f"Confidence: {confidence:.2%}")
            st.write(result["final_decision"]["message"])

            st.markdown("---")
            st.subheader("Phase 1 Result")
            st.write(f"Predicted Label: **{result['phase1_result']['predicted_label']}**")
            st.write(f"Normal Probability: **{result['phase1_result']['normal_probability']:.2%}**")
            st.write(f"Glaucoma Probability: **{result['phase1_result']['glaucoma_probability']:.2%}**")

            st.markdown("---")
            st.subheader("Phase 2 Result")
            st.write(f"Predicted Stage: **{result['phase2_result']['predicted_label']}**")
            st.write(f"Stage Confidence: **{result['phase2_result']['confidence']:.2%}**")

            probs = result["phase2_result"]["probabilities"]
            st.write("Class Probabilities:")
            st.write(f"- Advanced Glaucoma: {probs['advanced_glaucoma']:.2%}")
            st.write(f"- Early Glaucoma: {probs['early_glaucoma']:.2%}")
            st.write(f"- Normal Control: {probs['normal_control']:.2%}")

            st.markdown("---")
            st.subheader("Grad-CAM Visualization")

            try:
                original_np, overlay_np, heatmap = generate_gradcam_overlay(temp_path)

                g1, g2 = st.columns(2)
                with g1:
                    st.image(original_np, caption="Original Image", use_container_width=True)
                with g2:
                    st.image(overlay_np, caption="Grad-CAM Overlay", use_container_width=True)

            except Exception as e:
                st.warning("Grad-CAM could not be generated for this image.")
                st.code(str(e))

            st.markdown("---")
            with st.expander("Show Raw JSON Output"):
                st.json(result)

    if os.path.exists(temp_path):
        os.remove(temp_path)
