"""
Configuration settings for InsightGenie
Centralized configuration for easy maintenance and customization.
"""

import os
from typing import Final

# ==================== API Configuration ====================
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4")

# ==================== Application Settings ====================
APP_NAME: Final[str] = "InsightGenie"
APP_VERSION: Final[str] = "1.0.0"
HOST: str = os.getenv("HOST", "0.0.0.0")
PORT: int = int(os.getenv("PORT", "7860"))

# ==================== File Processing ====================
MAX_FILE_SIZE_MB: int = int(os.getenv("MAX_FILE_SIZE_MB", "50"))
MAX_FILE_SIZE_BYTES: int = MAX_FILE_SIZE_MB * 1024 * 1024
SUPPORTED_FILE_TYPES: Final[list] = [".xlsx", ".xls", ".csv"]
CHUNK_SIZE_ROWS: Final[int] = 10000  # For large file processing

# ==================== Data Analysis ====================
MAX_COLUMNS_TO_ANALYZE: Final[int] = 50
MAX_ROWS_FOR_PREVIEW: Final[int] = 100
NUMERIC_THRESHOLD: Final[float] = 0.8  # 80% numeric for numeric column detection
MISSING_DATA_THRESHOLD: Final[float] = 0.3  # 30% missing triggers warning

# ==================== AI Analysis ====================
AI_TEMPERATURE: Final[float] = 0.3  # Low for factual analysis
AI_MAX_TOKENS: Final[int] = 2000
AI_TIMEOUT_SECONDS: Final[int] = 60
AI_RETRY_ATTEMPTS: Final[int] = 3
AI_RETRY_DELAY: Final[float] = 2.0

# Analysis types
ANALYSIS_TYPES: Final[dict] = {
    "executive": "Executive Summary",
    "financial": "Financial Analysis",
    "operational": "Operational Insights",
    "generic": "General Analysis"
}

# ==================== Visualization ====================
CHART_WIDTH: Final[int] = 800
CHART_HEIGHT: Final[int] = 500
MAX_CATEGORIES_IN_CHART: Final[int] = 20
COLOR_SCHEME: Final[str] = "plotly"  # plotly, seaborn, or custom

# Chart types
CHART_TYPES: Final[list] = [
    "bar", "line", "scatter", "pie",
    "histogram", "box", "heatmap"
]

# ==================== Logging Configuration ====================
LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT: Final[str] = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_FILE: Final[str] = "insightgenie.log"

# ==================== Gradio UI Settings ====================
GRADIO_THEME: str = os.getenv("GRADIO_THEME", "default")
GRADIO_SHARE: bool = os.getenv("GRADIO_SHARE", "false").lower() == "true"
GRADIO_SERVER_NAME: str = HOST
GRADIO_SERVER_PORT: int = PORT

# UI Text
UI_TITLE: Final[str] = "🧠 InsightGenie - AI-Powered Excel Analytics"
UI_DESCRIPTION: Final[str] = """
Transform your Excel/CSV files into actionable insights with AI.
Upload your data and get executive summaries, visualizations, and recommendations.
"""

# ==================== Export Settings ====================
EXPORT_FORMATS: Final[list] = ["markdown", "html", "json"]
EXPORT_DIR: Final[str] = "exports"
INCLUDE_CHARTS_IN_EXPORT: Final[bool] = True

# ==================== Performance ====================
ENABLE_CACHING: Final[bool] = True
CACHE_SIZE_MB: Final[int] = 100
PARALLEL_PROCESSING: Final[bool] = True
MAX_WORKERS: Final[int] = 4

# ==================== Security ====================
ENABLE_RATE_LIMITING: Final[bool] = False
MAX_REQUESTS_PER_MINUTE: Final[int] = 10
SANITIZE_INPUTS: Final[bool] = True

# ==================== Feature Flags ====================
ENABLE_AI_ANALYSIS: bool = bool(OPENAI_API_KEY)
ENABLE_ADVANCED_VISUALIZATIONS: Final[bool] = True
ENABLE_ANOMALY_DETECTION: Final[bool] = True
ENABLE_EXPORT: Final[bool] = True


def validate_config() -> bool:
    """
    Validate configuration settings.
    Returns True if all required settings are valid, False otherwise.
    """
    errors = []

    # Check API key for AI features
    if not OPENAI_API_KEY and ENABLE_AI_ANALYSIS:
        errors.append("OPENAI_API_KEY is required for AI analysis features")

    # Validate file size limits
    if MAX_FILE_SIZE_MB <= 0:
        errors.append("MAX_FILE_SIZE_MB must be positive")

    # Validate port
    if not (1 <= PORT <= 65535):
        errors.append(f"Invalid PORT: {PORT}. Must be between 1-65535")

    # Validate AI settings
    if AI_TEMPERATURE < 0 or AI_TEMPERATURE > 2:
        errors.append(f"AI_TEMPERATURE must be between 0-2, got {AI_TEMPERATURE}")

    if errors:
        for error in errors:
            print(f"Config Error: {error}")
        return False

    return True


def get_config_summary() -> dict:
    """
    Get a summary of current configuration (for debugging/logging).
    Excludes sensitive information like API keys.
    """
    return {
        "app_name": APP_NAME,
        "app_version": APP_VERSION,
        "host": HOST,
        "port": PORT,
        "openai_model": OPENAI_MODEL,
        "max_file_size_mb": MAX_FILE_SIZE_MB,
        "ai_enabled": ENABLE_AI_ANALYSIS,
        "log_level": LOG_LEVEL,
        "supported_formats": SUPPORTED_FILE_TYPES,
    }


# Validate config on import
if __name__ != "__main__":
    validate_config()
