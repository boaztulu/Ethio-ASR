"""Minimal ToneColorConverter wrapper that avoids OpenVoice's broken
text-cleaner import chain (cn2an, eng_to_ipa, pypinyin, jieba).

We monkey-patch sys.modules so `from openvoice.api import ToneColorConverter`
works without installing all the locale text deps we'll never use.
"""
import sys
import types

# Stub the text-processing submodule so api.py's import succeeds
_stub_text = types.ModuleType("openvoice.text")
_stub_text.text_to_sequence = lambda *a, **k: []
sys.modules["openvoice.text"] = _stub_text
sys.modules["openvoice.text.symbols"] = types.ModuleType("openvoice.text.symbols")
sys.modules["openvoice.text.symbols"].symbols = []

_OV_REPO = "/blue/rcstudents/btulu/Projects/Ethio-TTS/external/OpenVoice"
if _OV_REPO not in sys.path:
    sys.path.insert(0, _OV_REPO)

# Now api.py imports cleanly
from openvoice.api import ToneColorConverter  # noqa: E402


def load_converter(ckpt_dir: str = "/blue/rcstudents/btulu/Projects/Ethio-TTS/openvoice_ckpts",
                   device: str = "cuda"):
    """Load OpenVoice v2 ToneColorConverter without watermark.

    The OpenVoice ToneColorConverter forwards `enable_watermark` to the
    base class which rejects it.  We instantiate without that kwarg and
    disable the watermark by setting watermark_model = None manually.
    """
    import os
    # Pre-stub wavmark so the import inside ToneColorConverter.__init__
    # is a no-op (we then overwrite watermark_model below).
    import types as _types
    if "wavmark" not in sys.modules:
        wm = _types.ModuleType("wavmark")
        class _DummyWM:
            def __init__(self): pass
            def to(self, _): return self
            def encode(self, signal, *_): return signal
        wm.load_model = lambda: _DummyWM()
        sys.modules["wavmark"] = wm
    converter = ToneColorConverter(
        os.path.join(ckpt_dir, "converter/config.json"),
        device=device,
    )
    converter.watermark_model = None   # ensure no watermark math runs
    converter.load_ckpt(os.path.join(ckpt_dir, "converter/checkpoint.pth"))
    return converter
