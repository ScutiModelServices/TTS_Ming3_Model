"""Shared admin settings plus restart-only Ming runtime options."""
from engine import DEFAULT_REVISION
from settings import Setting

SETTINGS = [
    Setting('update_check_hours', 'Check for updates every (hours)', 'float', 6.0, 'SERVICE_UPDATE_CHECK_HOURS', 0, 720),
    Setting('max_inflight', 'Calls allowed to wait for the model', 'int', 0, 'SERVICE_MAX_INFLIGHT', 0, 10000,
            help='One active generation per replica; remaining admitted calls wait. 0 = 4; excess calls get 429'),
    Setting('cache_size', 'Answers kept per replica', 'int', 32, 'SERVICE_CACHE_SIZE', 0, 100000,
            help='Seeded calls only; one-call references are never cached. Each entry is at most 1 MB'),
    Setting('model_revision', 'Model revision', 'text', DEFAULT_REVISION, 'SERVICE_MODEL_REVISION', restart=True, blank=False,
            help='Only inclusionAI/Ming-3.0-tts-0.5b. Download explicitly on the Updates card'),
    Setting('device', 'Device', 'text', 'cuda:0', 'SERVICE_DEVICE', restart=True, blank=False,
            help='CUDA device; inference refuses to run without a visible NVIDIA GPU'),
    Setting('max_new_tokens', 'Most decode steps per fragment', 'int', 1000, 'SERVICE_MAX_NEW_TOKENS', 1, 8000, restart=True),
    Setting('warm', 'Load downloaded model at start', 'bool', True, 'SERVICE_WARM', restart=True),
]
