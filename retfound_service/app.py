import base64
import io
import math
import os
import threading
from dataclasses import dataclass
from typing import Any

from fastapi import FastAPI, HTTPException
from PIL import Image, ImageFile, ImageStat
from pydantic import BaseModel, Field


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_env_file(env_path):
    if not os.path.exists(env_path):
        return

    with open(env_path, "r", encoding="utf-8") as env_file:
        for raw_line in env_file:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"')
            if key and key not in os.environ:
                os.environ[key] = value


def env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)

    if value is None:
        return default

    return value.strip().lower() in {"1", "true", "yes", "on"}


def safe_float(value: Any, digits: int = 4) -> float | None:
    try:
        return round(float(value), digits)
    except (TypeError, ValueError):
        return None


load_env_file(os.path.join(ROOT_DIR, ".env"))
ImageFile.LOAD_TRUNCATED_IMAGES = True


@dataclass
class ServiceSettings:
    requested_mode: str = os.environ.get("RETFOUND_MODE", "demo").strip().lower() or "demo"
    model_name: str = os.environ.get("RETFOUND_MODEL", "RETFound_dinov2").strip() or "RETFound_dinov2"
    model_arch: str = os.environ.get("RETFOUND_MODEL_ARCH", "retfound_dinov2").strip() or "retfound_dinov2"
    heart_checkpoint: str = os.environ.get("RETFOUND_HEART_CHECKPOINT", "").strip()
    class_labels: tuple[str, ...] = tuple(
        label.strip()
        for label in os.environ.get("RETFOUND_CLASS_LABELS", "").split(",")
        if label.strip()
    )
    device: str = os.environ.get("RETFOUND_DEVICE", "cpu").strip() or "cpu"
    port: int = int(os.environ.get("RETFOUND_PORT", "8001") or "8001")
    retina_validator_enabled: bool = env_bool("RETINA_VALIDATOR_ENABLED", True)
    retina_validator_model: str = os.environ.get("RETINA_VALIDATOR_MODEL", "openai/clip-vit-base-patch32").strip()
    retina_validator_threshold: float = float(os.environ.get("RETINA_VALIDATOR_THRESHOLD", "0.65") or "0.65")
    fundus_quality_enabled: bool = env_bool("FUNDUS_QUALITY_ENABLED", True)
    diabetes_model_enabled: bool = env_bool("DIABETES_MODEL_ENABLED", True)
    diabetes_model_provider: str = os.environ.get("DIABETES_MODEL_PROVIDER", "tanish").strip().lower() or "tanish"
    diabetes_model_repo: str = os.environ.get(
        "DIABETES_MODEL_REPO",
        "Tanishrajput/Diabetic-Retinopathy-Detection",
    ).strip()
    diabetes_hf_token: str = os.environ.get("HUGGINGFACE_TOKEN", "").strip()


class RetinalAnalysisRequest(BaseModel):
    question: str = Field(default="")
    topic: str = Field(default="eye-health")
    image_name: str = Field(default="")
    image_data_url: str = Field(min_length=20)


class ClipRetinalValidator:
    retinal_labels = (
        "retinal fundus photograph",
        "retinal scan image",
        "ophthalmology eye image",
    )
    non_retinal_labels = (
        "ordinary everyday photograph",
        "vehicle photo",
        "animal photo",
    )

    def __init__(self):
        self.model_name = settings.retina_validator_model
        self.threshold = settings.retina_validator_threshold
        self.pipeline = None
        self.pipeline_error = None
        self.quality_ensemble = None
        self.quality_error = None
        self._lock = threading.Lock()

    def validate(self, image: Image.Image) -> dict[str, Any]:
        notes: list[str] = []
        scores: dict[str, float] = {}
        accepted = True
        top_label = ""

        if settings.retina_validator_enabled:
            try:
                results = self._classify_with_clip(image)
                scores = {row["label"]: float(row["score"]) for row in results}
                top_label = results[0]["label"]
                retinal_score = sum(scores.get(label, 0.0) for label in self.retinal_labels)
                non_retinal_score = sum(scores.get(label, 0.0) for label in self.non_retinal_labels)
                accepted = retinal_score >= self.threshold and top_label in self.retinal_labels
                notes.append(
                    f'Image validator top label: "{top_label}" with retinal confidence {retinal_score:.2f}.'
                )
                if non_retinal_score > 0:
                    notes.append(f"Non-retinal confidence estimate: {non_retinal_score:.2f}.")
            except Exception as error:
                accepted = False
                notes.append(f"Retinal-image validator could not run: {error}")
        else:
            retinal_score = None
            non_retinal_score = None
            notes.append("Retinal-image validator is disabled, so the upload was accepted without a fundus-image gate.")

        quality_details = None
        if accepted and settings.fundus_quality_enabled:
            quality_details = self._predict_quality(image, notes)

        return {
            "accepted": accepted,
            "modelName": self.model_name,
            "topLabel": top_label,
            "retinalScore": safe_float(sum(scores.get(label, 0.0) for label in self.retinal_labels)),
            "nonRetinalScore": safe_float(sum(scores.get(label, 0.0) for label in self.non_retinal_labels)),
            "quality": quality_details,
            "notes": notes,
        }

    def _classify_with_clip(self, image: Image.Image):
        with self._lock:
            if self.pipeline is None and self.pipeline_error is None:
                from transformers import pipeline

                pipeline_device = -1
                if settings.device.startswith("cuda"):
                    try:
                        import torch

                        if torch.cuda.is_available():
                            pipeline_device = 0
                    except Exception:
                        pipeline_device = -1

                self.pipeline = pipeline(
                    "zero-shot-image-classification",
                    model=self.model_name,
                    device=pipeline_device,
                )

        if self.pipeline_error is not None:
            raise RuntimeError(self.pipeline_error)

        return self.pipeline(
            image.convert("RGB"),
            candidate_labels=list(self.retinal_labels + self.non_retinal_labels),
        )

    def _predict_quality(self, image: Image.Image, notes: list[str]):
        try:
            with self._lock:
                if self.quality_ensemble is None and self.quality_error is None:
                    from fundus_image_toolbox.quality_prediction.scripts.ensemble_inference import get_ensemble

                    self.quality_ensemble = get_ensemble(device=settings.device)

            from fundus_image_toolbox.quality_prediction.scripts.ensemble_inference import ensemble_predict
            import numpy as np

            confidence, label = ensemble_predict(
                self.quality_ensemble,
                np.asarray(image.convert("RGB")),
                threshold=0.5,
            )
            quality_label = "gradable" if int(label) == 1 else "ungradable"
            notes.append(
                f'Fundus quality model labeled the upload as "{quality_label}" with confidence {float(confidence):.2f}.'
            )
            return {
                "label": quality_label,
                "confidence": safe_float(confidence),
            }
        except Exception as error:
            self.quality_error = str(error)
            notes.append(f"Fundus quality model could not run: {error}")
            return None


class PrototypeHeartAnalyzer:
    analyzer_name = "AIcura cardiovascular prototype"
    active_mode = "prototype-heart-screening"
    model_loaded = True

    def analyze(
        self,
        image_name: str,
        image: Image.Image,
        validation: dict[str, Any] | None = None,
        image_properties: dict[str, Any] | None = None,
        diabetes_analysis: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if image_properties is None:
            image_properties, _ = inspect_image(image, image_name)

        grayscale_stats = ImageStat.Stat(image.convert("L"))
        rgb_stats = ImageStat.Stat(image.convert("RGB"))
        brightness = float(image_properties.get("averageBrightness") or grayscale_stats.mean[0])
        aspect_ratio = image_properties.get("aspectRatio")
        contrast = float(grayscale_stats.stddev[0])
        green_channel_std = float(rgb_stats.stddev[1])

        score = 0.34

        quality_label = validation.get("quality", {}).get("label") if validation else None
        if quality_label == "gradable":
            score += 0.08
        elif quality_label == "ungradable":
            score -= 0.05

        if 75 <= brightness <= 185:
            score += 0.04
        if contrast >= 55:
            score += 0.12
        elif contrast >= 40:
            score += 0.07
        elif contrast >= 25:
            score += 0.03

        if green_channel_std >= 45:
            score += 0.08
        elif green_channel_std >= 30:
            score += 0.04

        if aspect_ratio and 0.85 <= float(aspect_ratio) <= 1.25:
            score += 0.03

        diabetes_label = None
        diabetes_confidence_band = None
        if diabetes_analysis and diabetes_analysis.get("status") == "available" and diabetes_analysis.get("topPrediction"):
            diabetes_label = diabetes_analysis["topPrediction"]["label"]
            diabetes_confidence_band = diabetes_analysis.get("confidenceBand")

            if diabetes_confidence_band == "stronger":
                score += 0.12
            elif diabetes_confidence_band == "tentative":
                score += 0.06

            if diabetes_confidence_band != "low":
                if diabetes_label in {"Severe", "Proliferative_DR"}:
                    score += 0.08
                elif diabetes_label == "Moderate":
                    score += 0.04
                elif diabetes_label == "Mild":
                    score += 0.02

        score = max(0.18, min(0.82, round(score, 4)))

        lower_weight = max(0.05, 1.05 - score * 1.35)
        moderate_weight = max(0.05, 1.0 - abs(score - 0.52) * 2.2)
        higher_weight = max(0.05, (score - 0.18) * 1.35)
        total_weight = lower_weight + moderate_weight + higher_weight

        predictions = [
            {
                "label": "lower cardiovascular follow-up signal",
                "confidence": round(lower_weight / total_weight, 4),
            },
            {
                "label": "moderate cardiovascular follow-up signal",
                "confidence": round(moderate_weight / total_weight, 4),
            },
            {
                "label": "higher cardiovascular follow-up signal",
                "confidence": round(higher_weight / total_weight, 4),
            },
        ]
        predictions.sort(key=lambda item: item["confidence"], reverse=True)
        top_prediction = predictions[0]
        confidence_band = classify_screening_confidence(top_prediction["confidence"])

        if confidence_band == "low":
            summary = (
                "The heart branch produced a low-confidence prototype cardiovascular follow-up signal. "
                "This means the current image and screening context did not support a strong heart-related flag."
            )
        elif confidence_band == "tentative":
            summary = (
                "The heart branch produced a tentative prototype cardiovascular follow-up signal from retinal image characteristics "
                "and screening context. This is a heuristic prototype, not a trained heart diagnosis model."
            )
        else:
            summary = (
                "The heart branch produced a stronger prototype cardiovascular follow-up signal from retinal image characteristics "
                "and screening context. This is still a prototype screening result and needs clinician review."
            )

        if diabetes_label and diabetes_confidence_band in {"tentative", "stronger"}:
            summary += f' The diabetes branch contributed supporting screening context with top label "{diabetes_label}".'

        return {
            "status": "prototype",
            "modelLoaded": True,
            "modelName": "AIcura cardiovascular prototype",
            "summary": summary,
            "topPrediction": top_prediction,
            "predictions": predictions,
            "confidenceBand": confidence_band,
            "disclaimer": "Prototype cardiovascular screening output only. This is not a trained RETFound heart checkpoint, not medical advice, and not a diagnosis.",
        }


class RealRetFoundHeartAnalyzer:
    analyzer_name = "RETFound"
    active_mode = "retfound-heart"

    def __init__(self):
        import torch
        from torchvision import transforms
        from retfound_service.retfound_models import build_retfound_model
        from retfound_service.retfound_pos_embed import interpolate_pos_embed

        if not settings.heart_checkpoint:
            raise RuntimeError("RETFOUND_HEART_CHECKPOINT is missing.")
        if not settings.class_labels:
            raise RuntimeError("RETFOUND_CLASS_LABELS is missing.")

        self.torch = torch
        self.device = torch.device(settings.device)
        self.labels = list(settings.class_labels)
        self.model = build_retfound_model(
            model_name=settings.model_name,
            num_classes=len(self.labels),
            drop_path_rate=0.0,
            model_arch=settings.model_arch,
        )
        self.model_loaded = False
        self.transform = transforms.Compose(
            [
                transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                ),
            ]
        )

        checkpoint = torch.load(settings.heart_checkpoint, map_location="cpu")
        checkpoint_model = checkpoint.get("model") or checkpoint.get("teacher") or checkpoint
        checkpoint_model = {key.replace("backbone.", ""): value for key, value in checkpoint_model.items()}
        checkpoint_model = {key.replace("mlp.w12.", "mlp.fc1."): value for key, value in checkpoint_model.items()}
        checkpoint_model = {key.replace("mlp.w3.", "mlp.fc2."): value for key, value in checkpoint_model.items()}

        state_dict = self.model.state_dict()
        for key in ["head.weight", "head.bias"]:
            if key in checkpoint_model and key in state_dict and checkpoint_model[key].shape != state_dict[key].shape:
                del checkpoint_model[key]

        interpolate_pos_embed(self.model, checkpoint_model)
        self.model.load_state_dict(checkpoint_model, strict=False)
        self.model.eval()
        self.model.to(self.device)
        self.model_loaded = True

    def analyze(self, image_name: str, image: Image.Image, **_kwargs) -> dict[str, Any]:
        tensor = self.transform(image.convert("RGB")).unsqueeze(0).to(self.device)

        with self.torch.no_grad():
            logits = self.model(tensor)
            probabilities = self.torch.softmax(logits, dim=-1)[0].detach().cpu().tolist()

        indexed_scores = sorted(
            (
                {"label": label, "confidence": round(float(score), 4)}
                for label, score in zip(self.labels, probabilities)
            ),
            key=lambda item: item["confidence"],
            reverse=True,
        )

        return {
            "status": "available",
            "modelLoaded": True,
            "modelName": settings.model_name,
            "summary": "A task-specific RETFound checkpoint processed the retinal image. Treat the result as educational research support unless it has been clinically validated for your exact task.",
            "topPrediction": indexed_scores[0] if indexed_scores else None,
            "predictions": indexed_scores[:3],
            "disclaimer": "Research-stage heart output only. This is not medical advice, not a diagnosis, and does not replace a licensed clinician.",
        }


class TanishDiabetesAnalyzer:
    class_names = ["No_DR", "Mild", "Moderate", "Severe", "Proliferative_DR"]

    def __init__(self):
        self.model_name = settings.diabetes_model_repo
        self.model_loaded = False
        self.error = None
        self.model = None
        self.transform = None
        self.device = None
        self._lock = threading.Lock()

    def analyze(self, image_name: str, image: Image.Image) -> dict[str, Any]:
        try:
            self._ensure_loaded()
        except Exception as error:
            return {
                "status": "load_failed",
                "modelLoaded": False,
                "modelName": self.model_name,
                "summary": f"The diabetes model could not be loaded: {error}",
                "topPrediction": None,
                "predictions": [],
                "disclaimer": "No diabetes-model output was generated for this upload.",
            }

        with self._lock:
            image_tensor = self.transform(image.convert("RGB")).unsqueeze(0).to(self.device)
            with self.torch.no_grad():
                logits = self.model(image_tensor)
                probabilities = self.torch.softmax(logits, dim=-1)[0].detach().cpu().tolist()

        indexed_scores = sorted(
            (
                {"label": label, "confidence": round(float(score), 4)}
                for label, score in zip(self.class_names, probabilities)
            ),
            key=lambda item: item["confidence"],
            reverse=True,
        )

        top_prediction = indexed_scores[0] if indexed_scores else None
        confidence_band = classify_screening_confidence(top_prediction["confidence"] if top_prediction else None)

        if not top_prediction:
            summary = "The diabetes branch ran, but no prediction could be read."
        elif confidence_band == "low":
            summary = (
                f'The diabetes branch produced a low-confidence screening output. The top label was "{top_prediction["label"]}" '
                f'at {top_prediction["confidence"] * 100:.1f}% confidence, which is too uncertain to treat as a likely stage.'
            )
        elif confidence_band == "tentative":
            summary = (
                f'The diabetes branch produced a tentative screening output with top label "{top_prediction["label"]}" '
                f'at {top_prediction["confidence"] * 100:.1f}% confidence.'
            )
        else:
            summary = (
                f'The diabetes branch produced a stronger screening output with top label "{top_prediction["label"]}" '
                f'at {top_prediction["confidence"] * 100:.1f}% confidence.'
            )

        return {
            "status": "available",
            "modelLoaded": True,
            "modelName": self.model_name,
            "summary": summary,
            "topPrediction": top_prediction,
            "predictions": indexed_scores[:5],
            "confidenceBand": confidence_band,
            "disclaimer": "This diabetes-related retinal screening output is educational only. It is not medical advice or a diagnosis.",
        }

    def _ensure_loaded(self):
        if self.model_loaded:
            return
        if self.error:
            raise RuntimeError(self.error)

        with self._lock:
            if self.model_loaded:
                return
            if self.error:
                raise RuntimeError(self.error)

            try:
                import timm
                import torch
                import torch.nn as nn
                from huggingface_hub import hf_hub_download
                from torchvision import models, transforms

                self.torch = torch
                self.device = torch.device(settings.device)

                base_mobile = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V2)
                num_features_mobile = base_mobile.classifier[1].in_features
                base_mobile.classifier = nn.Identity()

                base_pit = timm.create_model("pit_b_224", pretrained=False, num_classes=5)
                num_features_pit = base_pit.head.in_features
                base_pit.head = nn.Identity()

                class CombinedModel(nn.Module):
                    def __init__(self, mobile_model, pit_model, num_classes):
                        super().__init__()
                        self.mobile_model = mobile_model
                        self.pit_model = pit_model
                        self.fc = nn.Linear(num_features_mobile + num_features_pit, num_classes)

                    def forward(self, x):
                        mobile_features = self.mobile_model(x)
                        pit_features = self.pit_model(x)

                        if len(mobile_features.shape) > 2:
                            mobile_features = mobile_features.flatten(1)
                        if len(pit_features.shape) > 2:
                            pit_features = pit_features.flatten(1)

                        combined = torch.cat((mobile_features, pit_features), dim=1)
                        return self.fc(combined)

                self.model = CombinedModel(base_mobile, base_pit, num_classes=5).to(self.device)
                checkpoint_path = hf_hub_download(
                    repo_id=self.model_name,
                    filename="Mobile_Pit_last_checkpoint.pt",
                    token=settings.diabetes_hf_token or None,
                )
                checkpoint = torch.load(checkpoint_path, map_location=self.device)
                state_dict = checkpoint.get("model_state_dict") or checkpoint
                self.model.load_state_dict(state_dict, strict=False)
                self.model.eval()
                self.transform = transforms.Compose(
                    [
                        transforms.Resize((224, 224)),
                        transforms.ToTensor(),
                        transforms.Normalize(
                            mean=(0.485, 0.456, 0.406),
                            std=(0.229, 0.224, 0.225),
                        ),
                    ]
                )
                self.model_loaded = True
            except Exception as error:
                self.error = str(error)
                raise


class DisabledDiabetesAnalyzer:
    model_loaded = False

    def analyze(self, image_name: str, image: Image.Image) -> dict[str, Any]:
        return {
            "status": "disabled",
            "modelLoaded": False,
            "modelName": settings.diabetes_model_repo,
            "summary": "The diabetes branch is currently disabled.",
            "topPrediction": None,
            "predictions": [],
            "disclaimer": "No diabetes-model output was generated for this upload.",
        }


class RetinalAnalysisService:
    def __init__(self):
        self.validator = ClipRetinalValidator()
        self.heart_analyzer = self._build_heart_analyzer()
        self.diabetes_analyzer = self._build_diabetes_analyzer()

    def analyze(self, question: str, image_name: str, image_data_url: str) -> dict[str, Any]:
        image = decode_image(image_data_url).convert("RGB")
        image_properties, image_notes = inspect_image(image, image_name)
        validation = self.validator.validate(image)

        if not validation["accepted"]:
            combined_notes = validation["notes"] + image_notes
            return {
                "analyzerName": "AIcura retinal analysis service",
                "requestedMode": settings.requested_mode,
                "activeMode": self.heart_analyzer.active_mode,
                "modelLoaded": getattr(self.heart_analyzer, "model_loaded", False),
                "modelName": settings.model_name,
                "isRetinalImage": False,
                "summary": "This upload does not appear to be a retinal fundus image, so retinal analysis was not run.",
                "imageProperties": image_properties,
                "validation": validation,
                "heartAnalysis": {
                    "status": "skipped",
                    "modelLoaded": False,
                    "modelName": settings.model_name,
                    "summary": "Heart analysis was skipped because the upload failed the retinal-image validation step.",
                    "topPrediction": None,
                    "predictions": [],
                    "disclaimer": "Upload a centered retinal fundus image before attempting heart-related retinal analysis.",
                },
                "diabetesAnalysis": {
                    "status": "skipped",
                    "modelLoaded": False,
                    "modelName": settings.diabetes_model_repo,
                    "summary": "Diabetes analysis was skipped because the upload failed the retinal-image validation step.",
                    "topPrediction": None,
                    "predictions": [],
                    "disclaimer": "Upload a centered retinal fundus image before attempting diabetes-related retinal analysis.",
                },
                "qualityNotes": combined_notes,
                "topPrediction": None,
                "predictions": [],
                "nextStep": "Upload a clear retinal fundus image with the inside of the eye visible and centered in frame.",
                "disclaimer": "This upload was rejected by the retinal-image validator. No retinal disease or heart-risk analysis was performed.",
            }

        diabetes_analysis = self.diabetes_analyzer.analyze(image_name, image)
        heart_analysis = self.heart_analyzer.analyze(
            image_name,
            image,
            validation=validation,
            image_properties=image_properties,
            diabetes_analysis=diabetes_analysis,
        )
        combined_notes = validation["notes"] + image_notes
        quality_details = validation.get("quality")
        if quality_details and quality_details.get("label") == "ungradable":
            combined_notes.append("The image passed the retina gate, but the fundus quality model considers it ungradable.")

        summary_parts = ["The upload passed the retinal-image validation step."]
        if heart_analysis["status"] == "available":
            summary_parts.append("A heart-specific RETFound checkpoint generated a research-stage output.")
        elif heart_analysis["status"] == "prototype":
            heart_confidence_band = heart_analysis.get("confidenceBand")
            if heart_confidence_band == "low":
                summary_parts.append("The heart branch produced a low-confidence prototype cardiovascular follow-up signal.")
            elif heart_confidence_band == "tentative":
                summary_parts.append("The heart branch produced a tentative prototype cardiovascular follow-up signal.")
            else:
                summary_parts.append("The heart branch produced a stronger prototype cardiovascular follow-up signal.")
        else:
            summary_parts.append("The heart-specific RETFound checkpoint is still missing, so a real heart report is not available yet.")

        if diabetes_analysis["status"] == "available" and diabetes_analysis["topPrediction"]:
            confidence_band = diabetes_analysis.get("confidenceBand")
            if confidence_band == "low":
                summary_parts.append(
                    "The diabetes branch produced a low-confidence screening output that should be interpreted cautiously."
                )
            elif confidence_band == "tentative":
                summary_parts.append(
                    f'The diabetes branch produced a tentative screening output with top label "{diabetes_analysis["topPrediction"]["label"]}".'
                )
            else:
                summary_parts.append(
                    f'The diabetes branch produced a stronger screening output with top label "{diabetes_analysis["topPrediction"]["label"]}".'
                )
        elif diabetes_analysis["status"] == "load_failed":
            summary_parts.append("The diabetes branch is configured, but the model could not be loaded.")
        else:
            summary_parts.append("The diabetes branch is disabled or did not run.")

        return {
            "analyzerName": "AIcura retinal analysis service",
            "requestedMode": settings.requested_mode,
            "activeMode": self.heart_analyzer.active_mode,
            "modelLoaded": getattr(self.heart_analyzer, "model_loaded", False),
            "modelName": settings.model_name,
            "isRetinalImage": True,
            "summary": " ".join(summary_parts),
            "imageProperties": image_properties,
            "validation": validation,
            "heartAnalysis": heart_analysis,
            "diabetesAnalysis": diabetes_analysis,
            "qualityNotes": combined_notes,
            "topPrediction": heart_analysis.get("topPrediction"),
            "predictions": heart_analysis.get("predictions", []),
            "nextStep": build_next_step(heart_analysis, diabetes_analysis),
            "disclaimer": "Educational retinal analysis only. These outputs are not medical advice, not a diagnosis, and do not replace a licensed clinician.",
        }

    def _build_heart_analyzer(self):
        if settings.requested_mode == "retfound":
            try:
                return RealRetFoundHeartAnalyzer()
            except Exception as error:
                print(f"[retfound_service] Falling back to prototype heart analyzer: {error}")
        return PrototypeHeartAnalyzer()

    def _build_diabetes_analyzer(self):
        if not settings.diabetes_model_enabled:
            return DisabledDiabetesAnalyzer()

        provider = settings.diabetes_model_provider
        if provider == "tanish":
            return TanishDiabetesAnalyzer()

        return DisabledDiabetesAnalyzer()


def inspect_image(image: Image.Image, image_name: str) -> tuple[dict[str, Any], list[str]]:
    grayscale = image.convert("L")
    brightness = ImageStat.Stat(grayscale).mean[0]
    width, height = image.size
    notes: list[str] = []

    if min(width, height) < 224:
        notes.append("The uploaded image is smaller than 224px on one side, which may limit model quality.")
    if width != height:
        notes.append("The uploaded image is not square, so most model pipelines will crop or pad it before inference.")
    if brightness < 45:
        notes.append("The image appears quite dark, which may reduce visible retinal detail.")
    if brightness > 210:
        notes.append("The image appears very bright, which can wash out vessel detail.")

    return (
        {
            "name": image_name or "uploaded-retinal-image",
            "format": image.format or "unknown",
            "width": width,
            "height": height,
            "averageBrightness": round(brightness, 1),
            "aspectRatio": round(width / height, 3) if height else None,
        },
        notes,
    )


def build_next_step(heart_analysis: dict[str, Any], diabetes_analysis: dict[str, Any]) -> str:
    if heart_analysis["status"] == "prototype":
        return "The prototype heart branch is active for this demo. If you later obtain a task-specific RETFound checkpoint, you can replace the prototype with a trained heart model."

    if heart_analysis["status"] != "available":
        return "Add or train a heart-specific RETFound checkpoint to enable a real heart report. The retina validator and diabetes branch can already be used for image intake and DR screening."

    if diabetes_analysis["status"] != "available":
        return "The heart branch is active. If you want diabetes output too, enable the diabetes model branch in the retinal service settings."

    return "Both the retinal gate and diabetes branch are active. Review the output as educational screening support only."


def classify_screening_confidence(confidence: float | None) -> str:
    if confidence is None:
        return "unknown"
    if confidence < 0.4:
        return "low"
    if confidence < 0.65:
        return "tentative"
    return "stronger"


def decode_image(image_data_url: str) -> Image.Image:
    if "," in image_data_url:
        _, encoded = image_data_url.split(",", 1)
    else:
        encoded = image_data_url

    image_bytes = base64.b64decode(encoded)
    image = Image.open(io.BytesIO(image_bytes))
    image.load()
    return image


settings = ServiceSettings()
analysis_service = RetinalAnalysisService()
app = FastAPI(title="AIcura RETFound Service", version="2.0.0")


@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "retfound_service",
        "requestedMode": settings.requested_mode,
        "activeMode": analysis_service.heart_analyzer.active_mode,
        "modelLoaded": getattr(analysis_service.heart_analyzer, "model_loaded", False),
        "modelName": settings.model_name,
        "checkpointConfigured": bool(settings.heart_checkpoint),
        "classLabelsConfigured": bool(settings.class_labels),
        "validatorEnabled": settings.retina_validator_enabled,
        "validatorModel": settings.retina_validator_model,
        "diabetesEnabled": settings.diabetes_model_enabled,
        "diabetesProvider": settings.diabetes_model_provider,
        "diabetesModelLoaded": getattr(analysis_service.diabetes_analyzer, "model_loaded", False),
    }


@app.post("/analyze")
def analyze_retinal_image(request: RetinalAnalysisRequest):
    try:
        return analysis_service.analyze(
            question=request.question,
            image_name=request.image_name,
            image_data_url=request.image_data_url,
        )
    except HTTPException:
        raise
    except Exception as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("retfound_service.app:app", host="127.0.0.1", port=settings.port, reload=False)
