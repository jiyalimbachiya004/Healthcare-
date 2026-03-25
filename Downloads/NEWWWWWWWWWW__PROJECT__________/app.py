import json
import base64
from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image


DATA_DIR = Path(__file__).parent
MODEL_CONFIG = {
    "Brain Tumor MRI (Training/Testing)": {
        "model_path": DATA_DIR / "brain_tumor_classifier.keras",
        "classes_path": DATA_DIR / "class_names.json",
        "title": "Brain Tumor MRI Classifier",
        "help": "Upload an MRI image to classify tumor type.",
        "train_cmd": "python train_model.py",
    },
    "Data Folder (train/valid/test)": {
        "model_path": DATA_DIR / "data_classifier.keras",
        "classes_path": DATA_DIR / "data_class_names.json",
        "title": "CT Scan Analysis",
        "help": "Upload a CT image to classify scan patterns.",
        "train_cmd": "python train_data_model.py",
    },
}
IMG_SIZE = (224, 224)
BACKGROUND_IMAGE_PATH = DATA_DIR / "assets" / "background.png"


@st.cache_resource
def load_model_and_classes(model_path: str, classes_path: str):
    model_path_obj = Path(model_path)
    classes_path_obj = Path(classes_path)
    if not model_path_obj.exists() or not classes_path_obj.exists():
        return None, None

    model = tf.keras.models.load_model(model_path_obj)
    with classes_path_obj.open("r", encoding="utf-8") as f:
        class_names = json.load(f)
    return model, class_names


def preprocess_image(image: Image.Image) -> np.ndarray:
    image = image.convert("RGB").resize(IMG_SIZE)
    arr = np.array(image, dtype=np.float32)
    arr = np.expand_dims(arr, axis=0)
    return arr


def is_valid_medical_image(image: Image.Image, preds: np.ndarray, model_name: str) -> bool:
    rgb = np.array(image.convert("RGB"), dtype=np.float32) / 255.0

    # MRI/X-ray images are typically low-color (near grayscale).
    channel_diff = (
        np.mean(np.abs(rgb[:, :, 0] - rgb[:, :, 1]))
        + np.mean(np.abs(rgb[:, :, 1] - rgb[:, :, 2]))
        + np.mean(np.abs(rgb[:, :, 0] - rgb[:, :, 2]))
    ) / 3.0

    top_conf = float(np.max(preds))

    # Reject likely non-medical images: highly colorful + weak confidence.
    if channel_diff > 0.13 and top_conf < 0.75:
        return False
    if top_conf < 0.45:
        return False

    # Slightly stricter confidence gate for CT flow.
    if model_name == "Data Folder (train/valid/test)" and top_conf < 0.55:
        return False
    return True


def get_case_insights(
    model_name: str,
    predicted_label: str,
    top_conf: float,
    top2_margin: float,
) -> dict:
    label = predicted_label.lower()

    # Informational guidance only. This is not a diagnosis.
    # Be conservative when confidence is low or classes are close.
    is_uncertain = top_conf < 0.75 or top2_margin < 0.12

    if model_name == "Brain Tumor MRI (Training/Testing)":
        if "notumor" in label:
            return {
                "phase": "AI prediction: no-tumor pattern" + (" (uncertain)" if is_uncertain else ""),
                "recovery_time": "Not applicable (needs clinical correlation)",
                "immediate_actions": (
                    "AI can miss tumors (false negative). Do not treat this as a confirmed normal result. "
                    "If you have symptoms (headache, seizures, weakness, vomiting, vision changes) or a prior report "
                    "suggesting tumor, seek urgent specialist review and confirm with a radiologist/neurologist."
                ),
            }
        if "glioma" in label:
            return {
                "phase": "Possible tumor pattern (glioma)" + (" (uncertain)" if is_uncertain else ""),
                "recovery_time": "Varies widely (weeks to years) depending on grade and treatment response",
                "immediate_actions": (
                    "Urgent neuro/oncology consultation, contrast MRI review, and confirmatory diagnosis "
                    "(radiology review ± biopsy as advised)."
                ),
            }
        if "meningioma" in label:
            return {
                "phase": "Possible tumor pattern (meningioma)" + (" (uncertain)" if is_uncertain else ""),
                "recovery_time": "Often weeks to months after treatment; varies by size/location",
                "immediate_actions": (
                    "Consult neurology/neurosurgery, confirm with radiology review, and assess observation vs surgery "
                    "based on symptoms and imaging."
                ),
            }
        if "pituitary" in label:
            return {
                "phase": "Possible tumor pattern (pituitary)" + (" (uncertain)" if is_uncertain else ""),
                "recovery_time": "Weeks to months depending on hormones, vision impact, and treatment",
                "immediate_actions": (
                    "Endocrinology + neurosurgery review, hormone panel testing, and vision field evaluation. "
                    "Confirm diagnosis with specialist."
                ),
            }

    if model_name == "Data Folder (train/valid/test)":
        if "normal" in label:
            return {
                "phase": "AI prediction: no-obvious abnormal pattern" + (" (uncertain)" if is_uncertain else ""),
                "recovery_time": "Not applicable (needs clinical correlation)",
                "immediate_actions": (
                    "AI can miss disease (false negative). Do not treat this as a confirmed normal result. "
                    "Confirm with a radiologist/doctor, especially if symptoms exist (persistent cough, chest pain, "
                    "breathlessness, blood in sputum)."
                ),
            }
        if "adenocarcinoma" in label:
            return {
                "phase": "Possible abnormal/tumor pattern (adenocarcinoma)" + (" (uncertain)" if is_uncertain else ""),
                "recovery_time": "Typically months to years; depends on stage and therapy",
                "immediate_actions": (
                    "Urgent doctor/oncology consultation, confirmatory diagnosis (radiology + biopsy if advised), "
                    "and staging workup (CT/PET) if confirmed."
                ),
            }
        if "large.cell.carcinoma" in label:
            return {
                "phase": "Possible abnormal/tumor pattern (large cell)" + (" (uncertain)" if is_uncertain else ""),
                "recovery_time": "Variable; often prolonged treatment course if confirmed",
                "immediate_actions": (
                    "Fast-track doctor/oncology referral and confirmatory diagnosis. If confirmed, staging and "
                    "treatment planning (chemo/immunotherapy/radiation) may be needed."
                ),
            }
        if "squamous.cell.carcinoma" in label:
            return {
                "phase": "Possible abnormal/tumor pattern (squamous)" + (" (uncertain)" if is_uncertain else ""),
                "recovery_time": "Variable; months to years based on stage and response if confirmed",
                "immediate_actions": (
                    "Urgent doctor/oncology evaluation and confirmatory diagnosis (imaging + pathology). "
                    "Address risk factors (e.g., smoking) and follow specialist guidance."
                ),
            }

    return {
        "phase": "Pattern uncertain",
        "recovery_time": "Needs clinical assessment",
        "immediate_actions": (
            "Consult a specialist for confirmatory diagnosis. Do not rely on AI output alone for treatment decisions."
        ),
    }


def render_probability_panel(class_names: list[str], preds: np.ndarray) -> None:
    pairs = sorted(
        [(class_names[i], float(preds[i])) for i in range(len(class_names))],
        key=lambda x: x[1],
        reverse=True,
    )
    top_label, top_score = pairs[0]
    st.markdown(
        f"""
        <div class="prob-headline">
            Top confidence: <strong>{top_label}</strong> ({top_score * 100:.2f}%)
        </div>
        """,
        unsafe_allow_html=True,
    )

    for label, score in pairs:
        percent = score * 100
        st.markdown(
            f"""
            <div class="prob-row">
                <div class="prob-row-top">
                    <span class="prob-label">{label}</span>
                    <span class="prob-value">{percent:.2f}%</span>
                </div>
                <div class="prob-track">
                    <div class="prob-fill" style="width: {percent:.2f}%;"></div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def get_base64_image(path: Path) -> str:
    if not path.exists():
        return ""
    return base64.b64encode(path.read_bytes()).decode("utf-8")


def apply_custom_style() -> None:
    bg_base64 = get_base64_image(BACKGROUND_IMAGE_PATH)
    bg_css = ""
    if bg_base64:
        bg_css = f"""
        .stApp {{
            background:
                linear-gradient(135deg, rgba(8, 18, 40, 0.88), rgba(90, 20, 120, 0.58)),
                url("data:image/png;base64,{bg_base64}") center/cover no-repeat fixed;
        }}
        """

    st.markdown(
        f"""
        <style>
            {bg_css}
            .main .block-container {{
                padding-top: 2rem;
                max-width: 940px;
            }}
            .glass-card {{
                background: linear-gradient(
                    135deg,
                    rgba(22, 31, 62, 0.72),
                    rgba(98, 38, 135, 0.52)
                );
                border: 1px solid rgba(255, 255, 255, 0.15);
                border-radius: 16px;
                padding: 1.1rem 1.2rem;
                margin-bottom: 1rem;
                box-shadow: 0 8px 30px rgba(0, 0, 0, 0.28);
                backdrop-filter: blur(6px);
            }}
            .flow-card {{
                background: linear-gradient(135deg, rgba(27, 66, 122, 0.68), rgba(129, 56, 156, 0.56));
                border: 1px solid rgba(255, 255, 255, 0.18);
                border-radius: 16px;
                padding: 1rem 1rem 0.3rem 1rem;
                box-shadow: 0 10px 26px rgba(0, 0, 0, 0.28);
                margin-bottom: 1rem;
            }}
            .flow-title {{
                font-size: 1.08rem;
                font-weight: 700;
                color: #f4f8ff;
                margin-bottom: 0.25rem;
            }}
            .flow-subtitle {{
                color: #d9e9ff;
                margin-bottom: 0.85rem;
                font-size: 0.95rem;
            }}
            .insight-card {{
                background: linear-gradient(130deg, rgba(21, 83, 92, 0.6), rgba(107, 42, 119, 0.55));
                border: 1px solid rgba(255, 255, 255, 0.18);
                border-radius: 14px;
                padding: 0.9rem 1rem;
                margin-top: 0.75rem;
            }}
            .insight-title {{
                font-weight: 700;
                margin-bottom: 0.45rem;
                color: #f1fbff;
            }}
            .disclaimer {{
                color: #ffd7d7;
                font-size: 0.88rem;
                margin-top: 0.5rem;
            }}
            .prob-headline {{
                background: linear-gradient(120deg, rgba(41, 126, 172, 0.4), rgba(155, 66, 189, 0.35));
                border: 1px solid rgba(255, 255, 255, 0.18);
                border-radius: 12px;
                padding: 0.55rem 0.75rem;
                margin-bottom: 0.8rem;
                color: #ebf5ff;
            }}
            .prob-row {{
                margin-bottom: 0.55rem;
            }}
            .prob-row-top {{
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-bottom: 0.22rem;
            }}
            .prob-label {{
                color: #edf5ff;
                font-weight: 600;
                text-transform: capitalize;
            }}
            .prob-value {{
                color: #d7f0ff;
                font-weight: 700;
            }}
            .prob-track {{
                width: 100%;
                height: 10px;
                background: rgba(255, 255, 255, 0.18);
                border-radius: 999px;
                overflow: hidden;
                border: 1px solid rgba(255, 255, 255, 0.15);
            }}
            .prob-fill {{
                height: 100%;
                border-radius: 999px;
                background: linear-gradient(90deg, #2ad7ff, #7d78ff, #ff5bbd);
                box-shadow: 0 0 10px rgba(121, 126, 255, 0.55);
            }}
            [data-testid="stSelectbox"] label,
            [data-testid="stFileUploader"] label {{
                font-weight: 700 !important;
                letter-spacing: 0.25px;
            }}
            [data-testid="stSelectbox"] [data-baseweb="select"] > div {{
                background: linear-gradient(100deg, rgba(255, 255, 255, 0.2), rgba(148, 182, 255, 0.18)) !important;
                border: 1px solid rgba(255, 255, 255, 0.32) !important;
                border-radius: 12px !important;
                min-height: 46px !important;
            }}
            [role="listbox"] {{
                background: linear-gradient(140deg, rgba(9, 20, 49, 0.95), rgba(76, 29, 109, 0.92)) !important;
                border: 1px solid rgba(255, 255, 255, 0.18) !important;
                border-radius: 12px !important;
                box-shadow: 0 14px 30px rgba(0, 0, 0, 0.38) !important;
            }}
            [role="option"] {{
                background: transparent !important;
                color: #eef4ff !important;
                border-radius: 8px !important;
                margin: 3px 6px !important;
            }}
            [role="option"]:hover {{
                background: linear-gradient(90deg, rgba(52, 98, 189, 0.36), rgba(154, 74, 196, 0.36)) !important;
            }}
            [role="option"][aria-selected="true"] {{
                background: linear-gradient(90deg, rgba(70, 124, 230, 0.45), rgba(176, 92, 222, 0.45)) !important;
                border: 1px solid rgba(255, 255, 255, 0.14) !important;
            }}
            [data-testid="stFileUploaderDropzone"] {{
                background: linear-gradient(100deg, rgba(255, 255, 255, 0.14), rgba(157, 107, 255, 0.18)) !important;
                border: 1px dashed rgba(255, 255, 255, 0.45) !important;
                border-radius: 12px !important;
                padding: 1rem 0.6rem !important;
            }}
            [data-testid="stFileUploaderDropzoneInstructions"] span {{
                color: #eaf2ff !important;
            }}
            .stButton > button {{
                width: 100%;
                border: 0;
                border-radius: 12px;
                font-weight: 600;
                color: #ffffff;
                background: linear-gradient(90deg, #00c6ff, #7a5cff, #ff4f9a);
                padding: 0.62rem 0.9rem;
                box-shadow: 0 8px 18px rgba(0, 0, 0, 0.3);
                transition: transform 0.15s ease, box-shadow 0.15s ease;
            }}
            .stButton > button:hover {{
                transform: translateY(-1px);
                box-shadow: 0 12px 22px rgba(0, 0, 0, 0.34);
            }}
            h1, h2, h3, p, label {{
                color: #f8fbff !important;
            }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_home() -> None:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.title("Choose a Testing Flow")
    st.write("Select one model card to open its dedicated prediction page.")
    st.markdown("</div>", unsafe_allow_html=True)

    for model_name, config in MODEL_CONFIG.items():
        st.markdown('<div class="flow-card">', unsafe_allow_html=True)
        st.markdown(f'<div class="flow-title">{config["title"]}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="flow-subtitle">{config["help"]}</div>', unsafe_allow_html=True)
        if st.button(f"Open {config['title']}", key=f"open_{model_name}"):
            st.session_state["page"] = "predict"
            st.session_state["selected_model"] = model_name
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)


def render_predict_page(model_name: str) -> None:
    selected = MODEL_CONFIG[model_name]
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    if st.button("← Back to Home", key="back_home"):
        st.session_state["page"] = "home"
        st.rerun()
    st.title(selected["title"])
    st.write(selected["help"])
    st.markdown("</div>", unsafe_allow_html=True)

    model, class_names = load_model_and_classes(
        str(selected["model_path"]),
        str(selected["classes_path"]),
    )
    if model is None:
        st.warning(f"Model not found. Run `{selected['train_cmd']}` first.")
        return

    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Upload image",
        type=["png", "jpg", "jpeg", "bmp", "webp"],
        key=f"upload_{model_name}",
    )
    predict_clicked = st.button("Predict", key=f"predict_{model_name}")
    st.markdown("</div>", unsafe_allow_html=True)

    if predict_clicked:
        if uploaded_file is None:
            st.warning("Please upload an image first.")
            return

        image = Image.open(uploaded_file)
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.image(image, caption="Uploaded image", use_container_width=True)

        input_tensor = preprocess_image(image)
        preds = model.predict(input_tensor, verbose=0)[0]
        top_idx = int(np.argmax(preds))
        top_conf = float(np.max(preds))
        top2 = float(np.sort(preds)[-2]) if len(preds) >= 2 else 0.0
        top2_margin = top_conf - top2

        if not is_valid_medical_image(image, preds, model_name):
            st.error("Image is not valid. Please upload a valid CT scan or brain tumor MRI image.")
            st.toast("Upload a valid CT scan or brain tumor MRI image.")
            st.markdown("</div>", unsafe_allow_html=True)
            return

        st.subheader("Prediction")
        predicted_label = class_names[top_idx]
        st.success(f"{predicted_label} ({preds[top_idx] * 100:.2f}%)")

        pl = predicted_label.lower()
        if model_name == "Brain Tumor MRI (Training/Testing)" and "notumor" in pl:
            st.warning(
                "Important: `notumor` is only the AI model’s prediction. It can be wrong (false negatives). "
                "Please confirm with a radiologist/neurologist—especially if symptoms or prior reports suggest tumor."
            )
        if model_name == "Data Folder (train/valid/test)" and "normal" in pl:
            st.warning(
                "Important: `normal` is only the AI model’s prediction. It can be wrong (false negatives). "
                "Please confirm with a radiologist/doctor—especially if symptoms are present."
            )

        st.subheader("Class probabilities")
        render_probability_panel(class_names, preds)

        insights = get_case_insights(model_name, predicted_label, top_conf, top2_margin)
        st.markdown('<div class="insight-card">', unsafe_allow_html=True)
        st.markdown('<div class="insight-title">Clinical Insight (AI-assisted)</div>', unsafe_allow_html=True)
        st.write(f"**Possible phase:** {insights['phase']}")
        st.write(f"**Estimated recovery timeline:** {insights['recovery_time']}")
        st.write(f"**Immediate actions:** {insights['immediate_actions']}")
        st.markdown(
            '<div class="disclaimer">For educational support only, not a medical diagnosis. '
            "Always confirm with a qualified doctor.</div>",
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)


def main() -> None:
    st.set_page_config(page_title="Image Classifier", page_icon="🧠")
    apply_custom_style()

    if "page" not in st.session_state:
        st.session_state["page"] = "home"
    if "selected_model" not in st.session_state:
        st.session_state["selected_model"] = list(MODEL_CONFIG.keys())[0]

    if st.session_state["page"] == "home":
        render_home()
    else:
        render_predict_page(st.session_state["selected_model"])


if __name__ == "__main__":
    main()
