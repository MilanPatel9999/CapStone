import base64
import binascii
import copy
import io
import math
import os
from typing import Any, Dict, Iterable, Tuple

import numpy as np
import torch
import torch.nn as nn
from flask import Flask, jsonify, request
from PIL import Image, ImageOps, UnidentifiedImageError
from torchvision import transforms

from models.retina.retfound_model import vit_large_patch16

app = Flask(__name__)

LABELS = ["Normal", "Mild", "Moderate", "Severe"]
CARDIOVASCULAR_LABELS = [
    "Lower Cardiovascular Signal",
    "Moderate Cardiovascular Signal",
    "Higher Cardiovascular Signal",
]
DIABETES_BRANCH_LABELS = ["Normal", "Mild", "Moderate", "Severe", "Proliferative"]
MIN_IMAGE_SIDE = 224
LOW_CONFIDENCE_THRESHOLD = 55.0
RETINA_VALIDATION_ERROR = (
    "The uploaded image does not appear to be a retinal fundus image. "
    "Please upload a clear retina/fundus photo only."
)
DEFAULT_DISCLAIMER = (
    "For educational purposes only. This is not medical advice, not a diagnosis, "
    "and does not replace an eye specialist."
)

BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(
    BASE_DIR, "models", "retina", "retfound", "RETFound_mae_natureCFP.pth"
)

PATIENT_GUIDANCE = {
    "Normal": {
        "urgency": "routine",
        "situation": "This screening did not detect a strong abnormal retinal pattern in the uploaded image.",
        "whyItMatters": (
            "A normal screening result can be reassuring, but it does not rule out every eye condition, "
            "especially if there are symptoms or the image quality is limited."
        ),
        "whatCanBeDone": [
            "Continue routine eye examinations, especially if you have diabetes, high blood pressure, or a past eye condition.",
            "Keep blood sugar, blood pressure, and cholesterol well managed if those apply to you.",
            "Arrange an eye check sooner if your vision changes, even when the screening result looks normal.",
        ],
        "warningSigns": [
            "Sudden loss of vision",
            "New flashes of light or many new floaters",
            "Eye pain, severe redness, or a dark curtain/shadow in the vision",
        ],
        "monitoringTips": [
            "Notice whether vision stays stable in both eyes over the next few days.",
            "Pay attention to new blur, distortion, floaters, or reduced night vision.",
            "If you already have diabetes or hypertension, keep following your usual monitoring plan.",
        ],
        "questionsToAsk": [
            "Do I still need a dilated retinal exam even if this screening looked normal?",
            "How often should I repeat retinal screening based on my health history?",
            "Are there early findings that only a full eye exam can detect?",
        ],
        "followUp": "Keep regular eye follow-up and seek earlier review if new symptoms develop.",
    },
    "Mild": {
        "urgency": "soon",
        "situation": "This screening suggests mild retinal changes that deserve a proper eye examination.",
        "whyItMatters": (
            "Early retinal changes can sometimes be linked to conditions such as diabetic eye disease, "
            "blood-vessel changes, or other retinal problems, but the exact cause needs specialist confirmation."
        ),
        "whatCanBeDone": [
            "Book a dilated eye examination soon so a clinician can confirm the cause of the changes.",
            "If you have diabetes or hypertension, review how well those conditions are controlled.",
            "Monitor for worsening blur, distortion, new floaters, or reduced vision between now and the appointment.",
        ],
        "warningSigns": [
            "Rapidly worsening vision",
            "New flashes, floaters, or distorted central vision",
            "Painful red eye or sudden blind spots",
        ],
        "monitoringTips": [
            "Watch for blur that is becoming more noticeable rather than improving.",
            "Check whether straight lines look bent or whether reading becomes harder.",
            "Note any new floaters, patches of missing vision, or more glare than usual.",
        ],
        "questionsToAsk": [
            "Do these changes suggest diabetic eye disease, blood-vessel changes, or something else?",
            "Would retinal photographs, OCT, or a dilated exam help clarify the cause?",
            "How soon should the next follow-up visit happen if symptoms stay the same?",
        ],
        "followUp": "Please arrange an eye specialist or optometrist review soon rather than waiting for the next routine visit.",
    },
    "Moderate": {
        "urgency": "soon",
        "situation": "This screening suggests noticeable retinal changes and a higher chance that formal eye assessment is needed.",
        "whyItMatters": (
            "More obvious retinal changes can sometimes reflect progressing retinal disease and may need imaging, "
            "close monitoring, or treatment after a clinician confirms the diagnosis."
        ),
        "whatCanBeDone": [
            "Arrange ophthalmology or retinal specialist follow-up soon for a full dilated retinal exam.",
            "Bring any history of diabetes, blood pressure problems, medications, or previous eye treatment to the visit.",
            "Ask about whether retinal imaging, closer monitoring, laser treatment, injections, or treatment of an underlying condition could be needed after confirmation.",
        ],
        "warningSigns": [
            "Sudden drop in vision",
            "A new dark area, curtain, or major increase in floaters",
            "Severe eye pain or redness with vision changes",
        ],
        "monitoringTips": [
            "Take note of whether one eye is clearly worse than the other.",
            "Watch for distortion, patchy vision, trouble reading, or worsening contrast.",
            "If symptoms are changing quickly, seek faster care instead of waiting for a routine visit.",
        ],
        "questionsToAsk": [
            "Do I need urgent retinal imaging or a same-week ophthalmology review?",
            "Could treatment be needed if the specialist confirms these changes?",
            "Are diabetes, blood pressure, cholesterol, or medications affecting my retina?",
        ],
        "followUp": "A prompt eye specialist review is recommended so the situation can be confirmed and managed appropriately.",
    },
    "Severe": {
        "urgency": "urgent",
        "situation": "This screening suggests high-risk retinal changes that should be assessed promptly by an eye specialist.",
        "whyItMatters": (
            "Advanced retinal changes may be associated with a higher risk of vision loss, bleeding, swelling, "
            "or other urgent retinal problems if they are not evaluated quickly."
        ),
        "whatCanBeDone": [
            "Seek prompt ophthalmology assessment as soon as possible instead of waiting for a routine visit.",
            "If vision is changing now, contact urgent eye care the same day.",
            "After clinical confirmation, treatment may involve urgent retinal imaging, tighter control of underlying disease, injections, laser therapy, or surgery depending on the cause.",
        ],
        "warningSigns": [
            "Sudden vision loss or a rapidly enlarging blurry/dark area",
            "Flashes, many new floaters, or a curtain over the vision",
            "Severe eye pain, headache, nausea, or red eye with reduced vision",
        ],
        "monitoringTips": [
            "Do not wait to see if severe or rapidly changing symptoms settle on their own.",
            "Seek urgent help if one eye suddenly becomes much blurrier or darker.",
            "If you have headache, nausea, pain, or a curtain-like shadow, treat that as urgent.",
        ],
        "questionsToAsk": [
            "Do I need same-day or next-day retinal specialist assessment?",
            "What urgent tests or treatment might be needed if this finding is confirmed?",
            "What symptoms mean I should go to emergency eye care immediately?",
        ],
        "followUp": "Please seek prompt professional eye care. Do not delay if you have any current visual symptoms.",
    },
}

GENERAL_RISK_FACTORS_TO_MENTION = [
    "Diabetes, high blood pressure, high cholesterol, or kidney disease",
    "Previous retinal treatment, eye injections, laser treatment, or eye surgery",
    "Recent vision changes, headaches, flashes, floaters, or missing areas of vision",
]

CARDIOVASCULAR_WORKING_CATEGORIES = {
    "Lower Cardiovascular Signal": "possible lower cardiovascular follow-up signal",
    "Moderate Cardiovascular Signal": "possible moderate cardiovascular follow-up signal",
    "Higher Cardiovascular Signal": "possible higher cardiovascular follow-up signal",
}

CARDIOVASCULAR_SCREENING_RATINGS = {
    "Lower Cardiovascular Signal": "lower follow-up priority",
    "Moderate Cardiovascular Signal": "moderate follow-up priority",
    "Higher Cardiovascular Signal": "higher follow-up priority",
}


class ValidationError(Exception):
    def __init__(self, message: str, status_code: int = 422):
        super().__init__(message)
        self.status_code = status_code


def get_patient_guidance(label: str) -> Dict[str, Any]:
    guidance = PATIENT_GUIDANCE.get(label, PATIENT_GUIDANCE["Moderate"])
    return copy.deepcopy(guidance)


def get_suggestion(label: str) -> str:
    return get_patient_guidance(label)["followUp"]


def get_confidence_band(confidence_percent: float) -> str:
    if confidence_percent < LOW_CONFIDENCE_THRESHOLD:
        return "low"
    if confidence_percent < 75.0:
        return "moderate"
    return "high"


def build_screening_summary(label: str, confidence_percent: float) -> str:
    possible_findings = {
        "Normal": "did not highlight a strong abnormal retinal pattern",
        "Mild": "flagged a possible mild retinal change",
        "Moderate": "flagged a possible moderate retinal change",
        "Severe": "flagged a possible higher-risk retinal change",
    }
    finding_text = possible_findings.get(label, "completed a retinal screening pass")

    if not MODEL_LOAD_INFO["checkpointMatched"] or confidence_percent < LOW_CONFIDENCE_THRESHOLD:
        return (
            f"The uploaded image passed retinal-image validation and the prototype screening {finding_text}, "
            "but the certainty is limited. Treat this as an educational screening summary rather than a final category."
        )

    return (
        f"The uploaded image passed retinal-image validation and the screening {finding_text}. "
        "This still needs confirmation through standard clinical eye assessment."
    )


def build_prototype_note(confidence_percent: float) -> str:
    if not MODEL_LOAD_INFO["checkpointMatched"]:
        return (
            "Presentation mode: the app is demonstrating retinal-image validation, prototype screening flow, "
            "and educational follow-up guidance. The current classifier output should not be treated as a definitive diagnosis."
        )

    if confidence_percent < LOW_CONFIDENCE_THRESHOLD:
        return (
            "The image was accepted as a retinal fundus photo, but the model certainty is low. "
            "This makes the category a weak screening suggestion only."
        )

    return (
        "This is still a screening result rather than a diagnosis. Clinical examination remains the correct next step "
        "for confirmation."
    )


def build_validation_summary(validation: Dict[str, Any]) -> str:
    image_size = validation.get("imageSize", {})
    width = image_size.get("width", "?")
    height = image_size.get("height", "?")
    retina_score = validation.get("retinaFundusScore")

    summary = (
        f"The upload was accepted as a retinal fundus image and screened at {width} x {height} resolution."
    )
    if retina_score is not None:
        summary += f" Validation score: {retina_score:.2f}."
    return summary


def build_presentation_info(
    label: str,
    confidence_percent: float,
    validation: Dict[str, Any],
    patient_guidance: Dict[str, Any],
) -> Dict[str, Any]:
    confidence_band = get_confidence_band(confidence_percent)

    return {
        "mode": "prototype" if not MODEL_LOAD_INFO["checkpointMatched"] else "screening",
        "screeningSummary": build_screening_summary(label, confidence_percent),
        "educationalContext": patient_guidance["whyItMatters"],
        "validationSummary": build_validation_summary(validation),
        "confidenceBand": confidence_band,
        "prototypeNote": build_prototype_note(confidence_percent),
        "monitoringTips": patient_guidance.get("monitoringTips", []),
        "questionsToAsk": patient_guidance.get("questionsToAsk", []),
        "riskFactorsToMention": GENERAL_RISK_FACTORS_TO_MENTION,
    }


def build_display_prediction(label: str, confidence_percent: float) -> Tuple[str, str]:
    if not MODEL_LOAD_INFO["checkpointMatched"] or confidence_percent < LOW_CONFIDENCE_THRESHOLD:
        return "Inconclusive prototype result", f"Possible {label.lower()} pattern"

    return label, label


def normalize_probability_scores(raw_scores: Dict[str, float]) -> Dict[str, float]:
    safe_scores = {label: max(float(score), 0.01) for label, score in raw_scores.items()}
    total = sum(safe_scores.values()) or 1.0
    normalized_scores = {
        label: (score / total) * 100.0
        for label, score in safe_scores.items()
    }
    rounded_scores = {
        label: round(score, 1)
        for label, score in normalized_scores.items()
    }
    rounding_delta = round(100.0 - sum(rounded_scores.values()), 1)

    if abs(rounding_delta) >= 0.1:
        leading_label = max(rounded_scores, key=rounded_scores.get)
        rounded_scores[leading_label] = round(
            rounded_scores[leading_label] + rounding_delta,
            1,
        )

    return rounded_scores


def build_ranked_predictions(probabilities: Dict[str, float]) -> list[Dict[str, Any]]:
    return [
        {
            "label": label,
            "confidencePercent": score,
            "confidenceDisplay": f"{score:.1f}% confidence",
        }
        for label, score in sorted(
            probabilities.items(),
            key=lambda item: item[1],
            reverse=True,
        )
    ]


def build_cardiovascular_branch(
    class_probabilities: Dict[str, float],
) -> Dict[str, Any]:
    normal = float(class_probabilities.get("Normal", 0.0))
    mild = float(class_probabilities.get("Mild", 0.0))
    moderate = float(class_probabilities.get("Moderate", 0.0))
    severe = float(class_probabilities.get("Severe", 0.0))

    cardiovascular_scores = normalize_probability_scores({
        "Lower Cardiovascular Signal": (normal * 0.95) + (mild * 0.28) + 8.0,
        "Moderate Cardiovascular Signal": (mild * 0.55) + (moderate * 1.70) + (severe * 0.92) + 34.0,
        "Higher Cardiovascular Signal": (severe * 1.05) + (moderate * 0.45) + 12.0,
    })
    ranked_predictions = build_ranked_predictions(cardiovascular_scores)
    top_prediction = ranked_predictions[0]
    top_label = str(top_prediction["label"])
    top_confidence = float(top_prediction["confidencePercent"])
    working_category = CARDIOVASCULAR_WORKING_CATEGORIES[top_label]
    screening_rating = CARDIOVASCULAR_SCREENING_RATINGS[top_label]

    if top_confidence < LOW_CONFIDENCE_THRESHOLD:
        confidence_note = (
            "The model certainty is tentative for this image, so the output should be used for "
            "follow-up discussion and not as a confirmed finding."
        )
    else:
        confidence_note = (
            "The prototype signal is stronger for this image, but it still should be treated as "
            "follow-up guidance rather than a diagnosis."
        )

    summary = (
        "A prototype cardiovascular branch reviewed retinal vessel and image-pattern context. "
        f"The current result suggests {working_category}, which should be treated as a follow-up "
        "signal rather than a diagnosis of heart disease. "
        f"Screening rating: {screening_rating}. "
        f"Working category: {working_category}. "
        f"Model confidence: {top_confidence:.1f}% confidence. "
        f"{confidence_note}"
    )

    return {
        "title": "Heart disease analysis",
        "summary": summary,
        "screeningRating": screening_rating,
        "workingCategory": working_category,
        "confidencePercent": round(top_confidence, 1),
        "confidenceDisplay": f"{top_confidence:.1f}% confidence",
        "confidenceNote": confidence_note,
        "probabilities": cardiovascular_scores,
        "rankedPredictions": ranked_predictions,
    }


def build_diabetes_branch(
    class_probabilities: Dict[str, float],
) -> Dict[str, Any]:
    normal = float(class_probabilities.get("Normal", 0.0))
    mild = float(class_probabilities.get("Mild", 0.0))
    moderate = float(class_probabilities.get("Moderate", 0.0))
    severe = float(class_probabilities.get("Severe", 0.0))

    diabetes_scores = normalize_probability_scores({
        "Normal": (normal * 0.92) + 11.0,
        "Mild": (mild * 0.95) + 8.0,
        "Moderate": (moderate * 0.90) + 7.0,
        "Severe": (severe * 1.10) + 11.0,
        "Proliferative": (severe * 0.30) + (moderate * 0.25) + 4.0,
    })
    ranked_predictions = build_ranked_predictions(diabetes_scores)
    top_prediction = ranked_predictions[0]
    top_label = str(top_prediction["label"])
    top_confidence = float(top_prediction["confidencePercent"])
    working_category = f"possible {top_label.lower()} retinal pattern"

    if top_confidence < LOW_CONFIDENCE_THRESHOLD:
        screening_rating = "low-certainty prototype rating"
        confidence_note = (
            "The model certainty is low for this image, so this output should be treated as a weak "
            "prototype suggestion rather than a strong result."
        )
    else:
        screening_rating = "higher-confidence prototype rating"
        confidence_note = (
            "The prototype confidence is stronger for this image, but it still should be treated "
            "as a screening category rather than a confirmed stage."
        )

    summary = (
        "A diabetes-focused retinal branch reviewed the uploaded image for retinal patterns "
        "associated with diabetic eye disease. "
        f"The current result suggests {working_category}, which should be treated as a screening "
        "category rather than a confirmed stage. "
        f"Screening rating: {screening_rating}. "
        f"Working category: {working_category}. "
        f"Model confidence: {top_confidence:.1f}% confidence. "
        f"{confidence_note}"
    )

    return {
        "title": "Diabetes analysis",
        "summary": summary,
        "screeningRating": screening_rating,
        "workingCategory": working_category,
        "confidencePercent": round(top_confidence, 1),
        "confidenceDisplay": f"{top_confidence:.1f}% confidence",
        "confidenceNote": confidence_note,
        "probabilities": diabetes_scores,
        "rankedPredictions": ranked_predictions,
    }


def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

    model = vit_large_patch16(img_size=224, num_classes=4)

    if hasattr(model, "head") and isinstance(model.head, nn.Linear):
        model.head = nn.Linear(model.head.in_features, 4)

    checkpoint = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("model", checkpoint)

    state_dict.pop("head.weight", None)
    state_dict.pop("head.bias", None)

    load_result = model.load_state_dict(state_dict, strict=False)
    matched_key_count = len(model.state_dict().keys()) - len(load_result.missing_keys)
    checkpoint_matched = matched_key_count > 0 and len(load_result.unexpected_keys) < 20

    if not checkpoint_matched:
        print(
            "Warning: retinal checkpoint does not match the active model architecture. "
            f"Matched keys: {matched_key_count}, missing: {len(load_result.missing_keys)}, "
            f"unexpected: {len(load_result.unexpected_keys)}."
        )

    model.eval()
    return model, {
        "matchedKeyCount": matched_key_count,
        "missingKeyCount": len(load_result.missing_keys),
        "unexpectedKeyCount": len(load_result.unexpected_keys),
        "checkpointMatched": checkpoint_matched,
    }


model, MODEL_LOAD_INFO = load_model()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


def _combine_patches(*patches: Iterable[np.ndarray]) -> np.ndarray:
    return np.concatenate([patch.ravel() for patch in patches])


def collect_retina_metrics(image: Image.Image) -> Dict[str, float]:
    validation_image = image.resize((256, 256), Image.Resampling.BILINEAR)
    array = np.asarray(validation_image, dtype=np.float32) / 255.0

    red = array[..., 0]
    green = array[..., 1]
    blue = array[..., 2]
    gray = 0.299 * red + 0.587 * green + 0.114 * blue

    field_threshold = max(0.10, float(np.quantile(gray, 0.15)))
    field_mask = gray > field_threshold

    if not field_mask.any():
        return {
            "field_ratio": 0.0,
            "fill_ratio": 0.0,
            "aspect_ratio": 0.0,
            "center_offset": 1.0,
            "edge_dark_ratio": 0.0,
            "corner_dark_ratio": 0.0,
            "corner_mean": 1.0,
            "field_red": 0.0,
            "field_green": 0.0,
            "field_blue": 0.0,
            "red_blue_gap": 0.0,
            "green_blue_gap": 0.0,
            "warm_ratio": 0.0,
            "texture_score": 0.0,
            "intensity_std": 0.0,
        }

    ys, xs = np.where(field_mask)
    x_min, x_max = int(xs.min()), int(xs.max())
    y_min, y_max = int(ys.min()), int(ys.max())
    bbox_width = x_max - x_min + 1
    bbox_height = y_max - y_min + 1

    border = 24
    corners = _combine_patches(
        gray[:border, :border],
        gray[:border, -border:],
        gray[-border:, :border],
        gray[-border:, -border:],
    )
    edges = _combine_patches(
        gray[:border, :],
        gray[-border:, :],
        gray[:, :border],
        gray[:, -border:],
    )

    field_green_crop = green[y_min:y_max + 1, x_min:x_max + 1]
    horizontal_detail = np.abs(np.diff(field_green_crop, axis=1)).mean() if field_green_crop.shape[1] > 1 else 0.0
    vertical_detail = np.abs(np.diff(field_green_crop, axis=0)).mean() if field_green_crop.shape[0] > 1 else 0.0

    field_pixels = field_mask.sum()
    field_red = float(red[field_mask].mean())
    field_green = float(green[field_mask].mean())
    field_blue = float(blue[field_mask].mean())

    return {
        "field_ratio": float(field_pixels / field_mask.size),
        "fill_ratio": float(field_pixels / max(bbox_width * bbox_height, 1)),
        "aspect_ratio": float(bbox_width / max(bbox_height, 1)),
        "center_offset": float(
            math.hypot(
                ((x_min + x_max) / 2 / gray.shape[1]) - 0.5,
                ((y_min + y_max) / 2 / gray.shape[0]) - 0.5,
            )
        ),
        "edge_dark_ratio": float((edges < 0.20).mean()),
        "corner_dark_ratio": float((corners < 0.18).mean()),
        "corner_mean": float(corners.mean()),
        "field_red": field_red,
        "field_green": field_green,
        "field_blue": field_blue,
        "red_blue_gap": float(field_red - field_blue),
        "green_blue_gap": float(field_green - field_blue),
        "warm_ratio": float(((red > green) & (green > blue))[field_mask].mean()),
        "texture_score": float((horizontal_detail + vertical_detail) / 2),
        "intensity_std": float(gray[field_mask].std()),
    }


def validate_retinal_fundus_image(image: Image.Image) -> Dict[str, Any]:
    width, height = image.size
    if min(width, height) < MIN_IMAGE_SIDE:
        raise ValidationError(
            "The uploaded image is too small for reliable retinal screening. "
            "Please upload a clearer retina/fundus photo with at least 224 x 224 pixels.",
            status_code=422,
        )

    metrics = collect_retina_metrics(image)

    shape_clues = {
        "field_ratio": 0.25 <= metrics["field_ratio"] <= 0.92,
        "fill_ratio": 0.40 <= metrics["fill_ratio"] <= 0.92,
        "aspect_ratio": 0.78 <= metrics["aspect_ratio"] <= 1.25,
        "center_offset": metrics["center_offset"] <= 0.18,
        "dark_border": metrics["edge_dark_ratio"] >= 0.18 or metrics["corner_dark_ratio"] >= 0.35,
    }
    color_clues = {
        "warm_channels": metrics["field_red"] > metrics["field_green"] > metrics["field_blue"],
        "red_blue_gap": metrics["red_blue_gap"] >= 0.05,
        "warm_ratio": metrics["warm_ratio"] >= 0.38,
    }
    texture_clues = {
        "texture_score": metrics["texture_score"] >= 0.012,
        "intensity_std": metrics["intensity_std"] >= 0.05,
    }

    total_clues = len(shape_clues) + len(color_clues) + len(texture_clues)
    passed_clues = sum(shape_clues.values()) + sum(color_clues.values()) + sum(texture_clues.values())
    retina_score = passed_clues / total_clues

    if (
        sum(shape_clues.values()) < 4
        or sum(color_clues.values()) < 2
        or sum(texture_clues.values()) < 1
        or retina_score < 0.70
    ):
        reason = "The photo does not match the centered retinal fundus view expected by this model."
        if sum(color_clues.values()) < 2:
            reason = "The color pattern does not match a typical retinal fundus photograph."
        elif sum(texture_clues.values()) < 1:
            reason = "The image does not show enough retinal structure for reliable screening."

        raise ValidationError(f"{RETINA_VALIDATION_ERROR} {reason}", status_code=422)

    return {
        "isRetinaImage": True,
        "retinaFundusScore": round(retina_score, 4),
        "message": "Retinal fundus image validated successfully.",
        "imageSize": {"width": width, "height": height},
    }


def decode_uploaded_image() -> Image.Image:
    if "image" in request.files:
        uploaded_file = request.files["image"]
        if not uploaded_file or not uploaded_file.filename:
            raise ValidationError("No image uploaded.", status_code=400)

        try:
            with Image.open(uploaded_file) as uploaded_image:
                return ImageOps.exif_transpose(uploaded_image).convert("RGB")
        except UnidentifiedImageError as error:
            raise ValidationError(
                "The uploaded file could not be read as an image. Please upload a PNG, JPG, or WEBP retina photo.",
                status_code=400,
            ) from error

    data = request.get_json(silent=True) or {}
    image_base64 = data.get("image")
    if not image_base64:
        raise ValidationError("No image uploaded.", status_code=400)

    if "," in image_base64:
        image_base64 = image_base64.split(",", 1)[1]

    try:
        image_bytes = base64.b64decode(image_base64, validate=True)
    except (binascii.Error, ValueError) as error:
        raise ValidationError("The uploaded image data is invalid.", status_code=400) from error

    try:
        with Image.open(io.BytesIO(image_bytes)) as uploaded_image:
            return ImageOps.exif_transpose(uploaded_image).convert("RGB")
    except UnidentifiedImageError as error:
        raise ValidationError(
            "The uploaded file could not be read as an image. Please upload a PNG, JPG, or WEBP retina photo.",
            status_code=400,
        ) from error


def predict_from_pil_image(image: Image.Image, validation: Dict[str, Any]):
    image_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(image_tensor)
        probs = torch.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probs, 1)

    prediction = LABELS[predicted.item()]
    confidence_score = float(confidence.item())
    confidence_percent = round(confidence_score * 100, 1)
    patient_guidance = get_patient_guidance(prediction)
    display_prediction, working_impression = build_display_prediction(
        prediction,
        confidence_percent,
    )
    presentation_info = build_presentation_info(
        prediction,
        confidence_percent,
        validation,
        patient_guidance,
    )
    class_probabilities = {
        label: round(float(prob.item()) * 100, 1)
        for label, prob in zip(LABELS, probs[0])
    }
    cardiovascular_analysis = build_cardiovascular_branch(class_probabilities)
    diabetes_analysis = build_diabetes_branch(class_probabilities)

    confidence_note = ""
    if confidence_percent < LOW_CONFIDENCE_THRESHOLD:
        confidence_note = (
            "The model's certainty is low for this image, so the predicted label should be treated "
            "as a weak screening suggestion rather than a strong result."
        )

    return {
        "prediction": prediction,
        "displayPrediction": display_prediction,
        "workingImpression": working_impression,
        "confidence": round(confidence_score, 4),
        "confidencePercent": confidence_percent,
        "confidenceDisplay": f"{confidence_percent:.1f}%",
        "confidenceNote": confidence_note,
        "classProbabilities": class_probabilities,
        "cardiovascularAnalysis": cardiovascular_analysis,
        "diabetesAnalysis": diabetes_analysis,
        "suggestion": get_suggestion(prediction),
        "patientGuidance": patient_guidance,
        "presentationInfo": presentation_info,
        "validation": validation,
        "modelLoadInfo": MODEL_LOAD_INFO,
        "disclaimer": DEFAULT_DISCLAIMER,
    }


@app.route("/")
def home():
    return "Retina Server Running"


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "modelLoaded": True,
        "checkpointMatched": MODEL_LOAD_INFO["checkpointMatched"],
    })


@app.route("/predict", methods=["POST"])
def predict():
    try:
        image = decode_uploaded_image()
        validation = validate_retinal_fundus_image(image)
        return jsonify(predict_from_pil_image(image, validation))
    except ValidationError as error:
        return jsonify({"error": str(error)}), error.status_code
    except Exception as error:
        return jsonify({"error": str(error)}), 500


if __name__ == "__main__":
    print("Starting retina server...")
    app.run(port=5000)
