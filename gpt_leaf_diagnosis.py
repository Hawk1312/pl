import base64
import importlib.metadata
import importlib.util
import json
import os
import platform
import re
import sys
from datetime import datetime
from io import BytesIO
from pathlib import Path

from PIL import Image

try:
    from dotenv import load_dotenv
except ModuleNotFoundError:
    load_dotenv = None


BASE_DIR = Path(__file__).resolve().parent
ENV_PATH = BASE_DIR / ".env"
ENV_EXAMPLE_PATH = BASE_DIR / ".env.example"


def load_environment() -> bool:
    if not ENV_PATH.exists():
        return False
    if load_dotenv is not None:
        load_dotenv(ENV_PATH, override=True)
        return True

    for line in ENV_PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ[key.strip()] = value.strip().strip("\"'")
    return True


load_environment()

OPENAI_LEAF_MODEL = os.getenv("OPENAI_LEAF_MODEL", "gpt-4.1")
GEMINI_LEAF_MODEL = os.getenv("GEMINI_LEAF_MODEL", "gemini-2.5-flash")

DIAGNOSIS_PROMPT = (
    "Analyze this plant leaf image. Return only valid JSON with these keys: "
    "plant, status, disease, confidence, solution. "
    "status must be Healthy, Infected, or Unknown. "
    "confidence must be a percentage string. "
    "If you are unsure, use Unknown and explain safe next steps in solution."
)


def safe_folder_name(value: str) -> str:
    value = value.strip().replace(" ", "_")
    value = re.sub(r"[^A-Za-z0-9_.-]+", "", value)
    return value or "unknown"


def image_to_data_url(image: Image.Image) -> str:
    buffer = BytesIO()
    image.convert("RGB").save(buffer, format="JPEG")
    encoded_image = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{encoded_image}"


def image_to_jpeg_bytes(image: Image.Image) -> bytes:
    buffer = BytesIO()
    image.convert("RGB").save(buffer, format="JPEG")
    return buffer.getvalue()


def parse_json_response(response_text: str) -> dict:
    try:
        return json.loads(response_text)
    except json.JSONDecodeError:
        start = response_text.find("{")
        end = response_text.rfind("}")
        if start == -1 or end == -1:
            raise
        return json.loads(response_text[start : end + 1])


def normalize_diagnosis(result: dict, provider: str) -> dict:
    return {
        "plant": str(result.get("plant", "Unknown")),
        "status": str(result.get("status", "Unknown")),
        "disease": str(result.get("disease", "Unknown")),
        "confidence": str(result.get("confidence", "Not available")),
        "solution": str(result.get("solution", "Ask a local agriculture expert to inspect the leaf.")),
        "provider": provider,
    }


def diagnose_leaf_with_openai(image: Image.Image) -> dict:
    load_environment()
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise RuntimeError("OPENAI_API_KEY is not set.")

    if not openai_api_key.startswith("sk-"):
        raise RuntimeError("OPENAI_API_KEY does not look like an OpenAI API key.")

    try:
        from openai import OpenAI
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "OpenAI package is not installed. Run: python -m pip install openai"
        ) from error

    client = OpenAI()

    response = client.responses.create(
        model=OPENAI_LEAF_MODEL,
        input=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": DIAGNOSIS_PROMPT,
                    },
                    {
                        "type": "input_image",
                        "image_url": image_to_data_url(image),
                    },
                ],
            }
        ],
    )

    result = parse_json_response(response.output_text)
    return normalize_diagnosis(result, "OpenAI")


def diagnose_leaf_with_gemini(image: Image.Image) -> dict:
    load_environment()
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    if not gemini_api_key:
        raise RuntimeError("GEMINI_API_KEY is not set.")

    if gemini_api_key.startswith("sk-"):
        raise RuntimeError(
            "GEMINI_API_KEY looks like an OpenAI key. Use a Google AI Studio Gemini API key instead."
        )

    try:
        import importlib

        genai = importlib.import_module("google.genai")
        from google.genai import types
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "Google GenAI package is not installed. Run: python -m pip install google-genai"
        ) from error

    client = genai.Client(api_key=gemini_api_key)
    response = client.models.generate_content(
        model=GEMINI_LEAF_MODEL,
        contents=[
            types.Part.from_text(text=DIAGNOSIS_PROMPT),
            types.Part.from_bytes(
                data=image_to_jpeg_bytes(image),
                mime_type="image/jpeg",
            ),
        ],
    )

    result = parse_json_response(response.text)
    return normalize_diagnosis(result, "Gemini")


def diagnose_leaf_with_gpt(image: Image.Image) -> dict:
    load_environment()
    errors = []

    if os.getenv("OPENAI_API_KEY"):
        try:
            return diagnose_leaf_with_openai(image)
        except Exception as error:
            errors.append(f"OpenAI failed: {error}")
    else:
        errors.append("OPENAI_API_KEY is not set")

    if os.getenv("GEMINI_API_KEY"):
        try:
            return diagnose_leaf_with_gemini(image)
        except Exception as error:
            errors.append(f"Gemini failed: {error}")
    else:
        errors.append("GEMINI_API_KEY is not set")

    raise RuntimeError("; ".join(errors))


def get_ai_fallback_status() -> dict:
    load_environment()
    return {
        "python_executable": sys.executable,
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "working_directory": str(Path.cwd()),
        "env_file": ENV_PATH,
        "env_file_exists": ENV_PATH.exists(),
        "env_example_file": ENV_EXAMPLE_PATH,
        "env_example_file_exists": ENV_EXAMPLE_PATH.exists(),
        "dotenv_available": load_dotenv is not None,
        "dotenv_package_version": _package_version("python-dotenv"),
        "openai_api_key_set": bool(os.getenv("OPENAI_API_KEY")),
        "openai_api_key_preview": _secret_preview(os.getenv("OPENAI_API_KEY")),
        "gemini_api_key_set": bool(os.getenv("GEMINI_API_KEY")),
        "gemini_api_key_preview": _secret_preview(os.getenv("GEMINI_API_KEY")),
        "openai_model": OPENAI_LEAF_MODEL,
        "gemini_model": GEMINI_LEAF_MODEL,
        "google_genai_available": _module_available("google.genai"),
        "google_genai_package_version": _package_version("google-genai"),
        "openai_available": _module_available("openai"),
        "openai_package_version": _package_version("openai"),
    }


def _module_available(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


def _package_version(package_name: str) -> str:
    try:
        return importlib.metadata.version(package_name)
    except importlib.metadata.PackageNotFoundError:
        return "not installed"


def _secret_preview(value: str | None) -> str:
    if not value:
        return "not set"
    if len(value) <= 8:
        return "***"
    return f"{value[:4]}...{value[-4:]}"


def save_new_disease_image(image: Image.Image, diagnosis: dict, original_name: str, dataset_dir: Path) -> Path:
    plant_folder = safe_folder_name(diagnosis.get("plant", "Unknown"))
    disease_folder = safe_folder_name(diagnosis.get("disease", "Unknown"))
    target_dir = dataset_dir / plant_folder / disease_folder
    target_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    original_stem = safe_folder_name(Path(original_name).stem)
    target_path = target_dir / f"{timestamp}_{original_stem}.jpg"
    image.convert("RGB").save(target_path, format="JPEG", quality=95)
    return target_path
