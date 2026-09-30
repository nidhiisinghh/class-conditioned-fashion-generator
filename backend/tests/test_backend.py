"""
Weave Backend Integration Test Suite
Validates all PRD-required API endpoints, AI inference services, and database persistence.
"""

import io
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from backend.app.main import app
from backend.app.core.database import init_db

# Ensure schema initialized
init_db()
client = TestClient(app)


def create_dummy_png():
    img = Image.new("RGB", (64, 64), (200, 100, 50))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf


def test_health_and_root():
    """Verifies server health and checkpoint presence."""
    r_root = client.get("/")
    assert r_root.status_code == 200
    assert r_root.json()["status"] == "online"

    r_health = client.get("/health")
    assert r_health.status_code == 200
    data = r_health.json()
    assert data["status"] == "healthy"
    assert data["database"] == "connected"
    assert data["checkpoints_present"]["category_classifier"] is True
    assert data["checkpoints_present"]["fashion_lora"] is True


def test_auth_flow():
    """Tests signup, login, and current designer profile retrieval."""
    test_email = "designer_test_01@weave.studio"
    signup_payload = {
        "email": test_email,
        "password": "SecurePassword123!",
        "display_name": "Haute Couture Lead"
    }

    # Signup
    r_signup = client.post("/api/v1/auth/signup", json=signup_payload)
    if r_signup.status_code == 409:
        # Already registered in previous test run
        r_login = client.post("/api/v1/auth/login", json={"email": test_email, "password": "SecurePassword123!"})
        assert r_login.status_code == 200
        token = r_login.json()["access_token"]
    else:
        assert r_signup.status_code == 201
        token = r_signup.json()["access_token"]

    assert token is not None

    # Get /me
    headers = {"Authorization": f"Bearer {token}"}
    r_me = client.get("/api/v1/auth/me", headers=headers)
    assert r_me.status_code == 200
    assert r_me.json()["email"] == test_email
    assert r_me.json()["role"] == "designer"


def test_generation_pipeline():
    """Tests multimodal generation with brief extraction and AI evaluation scoring."""
    payload = {
        "brief_text": "An asymmetrical structured terracotta silk evening dress with sculptural drape",
        "batch_size": 2
    }
    r = client.post("/api/v1/generations", data=payload)
    assert r.status_code == 201
    data = r.json()

    assert "generation_id" in data
    assert data["structured_brief"]["category"].lower() in ["dress", "gown"]
    assert len(data["concepts"]) == 2

    concept = data["concepts"][0]
    assert "concept_image_id" in concept
    assert concept["url"].startswith("/media/generations/")
    assert concept["category_consistency_prob"] >= 0.85
    assert concept["style_alignment_score"] > 0.0
    assert concept["diversity_score"] > 0.0

    gen_id = data["generation_id"]
    concept_id = concept["concept_image_id"]

    # Test conversational feedback routing
    fb_payload = {"feedback_text": "Change the silhouette to be more voluminous with a high collar"}
    r_fb = client.post(f"/api/v1/generations/{gen_id}/feedback", json=fb_payload)
    assert r_fb.status_code == 200
    fb_data = r_fb.json()
    assert fb_data["routed_operation"] in ["generate", "inpaint", "variation"]
    assert len(fb_data["concepts"]) > 0

    # Test variations endpoint
    var_payload = {"concept_image_id": concept_id, "strength": 0.35, "count": 2}
    r_var = client.post(f"/api/v1/generations/{gen_id}/variations", data=var_payload)
    assert r_var.status_code == 200
    assert len(r_var.json()["concepts"]) == 2

    # Test inpainting endpoint
    mask_file = create_dummy_png()
    files = {"mask": ("mask.png", mask_file, "image/png")}
    inpaint_data = {"concept_image_id": concept_id, "instruction": "deepen the waist belt to obsidian black"}
    r_inp = client.post(f"/api/v1/generations/{gen_id}/inpaint", data=inpaint_data, files=files)
    assert r_inp.status_code == 200
    assert r_inp.json()["operation_type"] == "inpaint"


def test_moodboard_and_export_flow():
    """Tests board creation, pinning concept items, layout update, and PDF export."""
    # 1. Create Generation first to get a concept
    r_gen = client.post(
        "/api/v1/generations",
        data={"brief_text": "Tailored emerald green wool blazer with peak lapels", "batch_size": 1}
    )
    assert r_gen.status_code == 201
    concept_id = r_gen.json()["concepts"][0]["concept_image_id"]

    # 2. Create Board
    r_board = client.post("/api/v1/boards", json={"title": "Autumn Haute Couture 2026"})
    assert r_board.status_code == 201
    board_id = r_board.json()["id"]

    # 3. Pin concept to board
    pin_payload = {
        "concept_image_id": concept_id,
        "position_x": 150.0,
        "position_y": 200.0,
        "annotation": "Key runway showstopper silhouette"
    }
    r_pin = client.post(f"/api/v1/boards/{board_id}/items", json=pin_payload)
    assert r_pin.status_code == 201
    item_id = r_pin.json()["id"]

    # 4. Update item layout position
    patch_payload = {"position_x": 300.0, "position_y": 420.0}
    r_patch = client.patch(f"/api/v1/boards/{board_id}/items/{item_id}", json=patch_payload)
    assert r_patch.status_code == 200
    assert r_patch.json()["position_x"] == 300.0

    # 5. Fetch Board details
    r_get_board = client.get(f"/api/v1/boards/{board_id}")
    assert r_get_board.status_code == 200
    assert len(r_get_board.json()["items"]) == 1

    # 6. Test Design Brief PDF Export
    r_export = client.post(f"/api/v1/concepts/{concept_id}/export")
    assert r_export.status_code == 201
    export_data = r_export.json()
    assert "pdf_url" in export_data
    assert export_data["pdf_url"].startswith("/media/exports/")
    assert len(export_data["prompt_lineage"]) >= 1
