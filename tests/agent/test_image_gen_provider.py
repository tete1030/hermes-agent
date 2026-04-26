from __future__ import annotations

import base64


def test_save_b64_image_sanitizes_prefix_separators(monkeypatch, tmp_path):
    from agent import image_gen_provider as mod

    monkeypatch.setattr(mod, "_images_cache_dir", lambda: tmp_path)
    b64 = base64.b64encode(b"img").decode("ascii")

    path = mod.save_b64_image(b64, prefix="vendor/model v1")

    assert path.parent == tmp_path
    assert path.exists()
    assert path.read_bytes() == b"img"
    assert path.name.startswith("vendor_model_v1_")
    assert "/" not in path.name
    assert "\\" not in path.name


def test_save_b64_image_prefix_falls_back_when_empty_after_sanitize(monkeypatch, tmp_path):
    from agent import image_gen_provider as mod

    monkeypatch.setattr(mod, "_images_cache_dir", lambda: tmp_path)
    b64 = base64.b64encode(b"img").decode("ascii")

    path = mod.save_b64_image(b64, prefix="///")

    assert path.parent == tmp_path
    assert path.name.startswith("image_")
