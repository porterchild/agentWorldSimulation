LOCAL_BASE_URL = "http://10.0.0.188:8080/v1"
LOCAL_API_KEY = "none"
LOCAL_MODEL_NAME = None  # auto-discovered from /v1/models if None

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_MODEL = "x-ai/grok-4.1-fast"

THINKING = False   # True = prepend reasoning scaffold to prompts (for non-thinking models)
SPAWN_BUDGET = 20  # global cap on total sub-entity spawns across the whole simulation
