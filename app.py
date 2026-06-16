"""
Cloud Wardrobe
--------------
Upload a clothing photo -> remove background -> AI-tag it with Gemini -> store in Supabase.

Required secrets (in .streamlit/secrets.toml or Streamlit Cloud settings):
    SUPABASE_URL = "..."
    SUPABASE_KEY = "..."
    GEMINI_API_KEY = "..."

Required packages (requirements.txt):
    streamlit
    rembg
    pillow
    supabase
    google-generativeai
    pydantic
    onnxruntime   # rembg dependency, often needs to be listed explicitly
"""

import io
import json
import uuid

import streamlit as st
from PIL import Image
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# 1. Page config MUST be the first Streamlit call
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Cloud Wardrobe", layout="centered")


# ---------------------------------------------------------------------------
# 2. Output schema for Gemini's structured response
# ---------------------------------------------------------------------------
class ClothingMetadata(BaseModel):
    category: str
    color: str
    style: str


# ---------------------------------------------------------------------------
# 3. Initialize API connections (cached so this doesn't re-run every rerun)
# ---------------------------------------------------------------------------
@st.cache_resource
def init_clients():
    """Create and cache the Supabase + Gemini clients. Raises on failure
    so callers can stop the app cleanly instead of limping on with None."""
    from supabase import create_client
    import google.generativeai as genai

    missing = [
        k for k in ("SUPABASE_URL", "SUPABASE_KEY", "GEMINI_API_KEY")
        if k not in st.secrets
    ]
    if missing:
        raise RuntimeError(f"Missing secrets: {', '.join(missing)}")

    supabase_client = create_client(st.secrets["SUPABASE_URL"], st.secrets["SUPABASE_KEY"])
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    return supabase_client, genai


try:
    supabase, genai = init_clients()
except Exception as e:
    st.error(f"Configuration error: {e}\n\nPlease check that your Streamlit Secrets are correctly filled.")
    st.stop()  # Hard stop -- prevents NameErrors later in the script


# ---------------------------------------------------------------------------
# 4. Cache the background-removal engine so it loads once, not per-upload
# ---------------------------------------------------------------------------
@st.cache_resource
def get_rembg_session():
    import rembg
    return rembg.new_session()


# ---------------------------------------------------------------------------
# 5. Helpers
# ---------------------------------------------------------------------------
def remove_background(pil_image: Image.Image, session) -> Image.Image:
    """Run rembg on a PIL image via raw bytes (most version-compatible path)."""
    from rembg import remove

    buf = io.BytesIO()
    pil_image.convert("RGB").save(buf, format="PNG")
    result_bytes = remove(buf.getvalue(), session=session)
    return Image.open(io.BytesIO(result_bytes)).convert("RGBA")


def classify_clothing(model, image_for_ai: Image.Image) -> dict:
    """Ask Gemini to tag the item, returning a plain dict that matches ClothingMetadata."""
    generation_config = {
        "response_mime_type": "application/json",
        "response_schema": ClothingMetadata,
    }

    response = model.generate_content(
        ["Analyze this clothing item precisely. Identify its category, "
         "dominant color, and style.", image_for_ai],
        generation_config=generation_config,
    )

    # Prefer the SDK's parsed object if available; fall back to manual JSON parsing.
    parsed = getattr(response, "parsed", None)
    if parsed is not None:
        data = parsed.model_dump() if isinstance(parsed, BaseModel) else parsed
    else:
        text = response.text.strip()
        # Defensive: strip accidental markdown code fences
        if text.startswith("```"):
            text = text.strip("`")
            text = text.split("\n", 1)[-1] if "\n" in text else text
        data = json.loads(text)

    # Validate against the schema so bad/missing fields fail loudly, not silently
    return ClothingMetadata(**data).model_dump()


def upload_to_supabase(client, file_bytes: bytes, original_filename: str) -> str:
    """Upload the processed PNG to Supabase storage and return its public URL."""
    safe_stem = original_filename.rsplit(".", 1)[0] if "." in original_filename else original_filename
    unique_name = f"{safe_stem}_{uuid.uuid4().hex[:8]}_processed.png"

    client.storage.from_("wardrobe").upload(
        unique_name,
        file_bytes,
        file_options={"content-type": "image/png"},
    )

    url_result = client.storage.from_("wardrobe").get_public_url(unique_name)
    # Different supabase-py versions return either a plain string or a dict
    if isinstance(url_result, dict):
        public_url = url_result.get("publicURL") or url_result.get("publicUrl") or str(url_result)
    else:
        public_url = url_result

    return public_url


# ---------------------------------------------------------------------------
# 6. UI
# ---------------------------------------------------------------------------
st.title("👗 My Cloud Wardrobe")
st.write("Upload an item to remove its background, auto-tag it with AI, and sync it to your cloud storage.")

uploaded_file = st.file_uploader("Snap or upload a photo of your clothing item", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    status_box = st.empty()

    if uploaded_file.size == 0:
        st.error("The uploaded file is empty. Please choose a different photo.")
        st.stop()

    try:
        status_box.info("⚡ Compressing image to save cloud space...")
        raw_img = Image.open(uploaded_file)
        raw_img.thumbnail((800, 800))

        status_box.info("✂️ Extracting clothing item from background...")
        session = get_rembg_session()
        clean_img = remove_background(raw_img, session)

        img_byte_arr = io.BytesIO()
        clean_img.save(img_byte_arr, format="PNG")
        img_bytes = img_byte_arr.getvalue()

        status_box.info("🤖 AI is analyzing features and color schemas...")
        model = genai.GenerativeModel("gemini-2.0-flash")
        # Gemini's vision input expects RGB, not RGBA cutouts
        metadata = classify_clothing(model, clean_img.convert("RGB"))

        status_box.info("☁️ Syncing assets to Supabase storage bucket...")
        public_url = upload_to_supabase(supabase, img_bytes, uploaded_file.name)

        db_data = {
            "category": metadata["category"],
            "color": metadata["color"],
            "style": metadata["style"],
            "image_url": public_url,
        }
        supabase.table("clothes").insert(db_data).execute()

        status_box.empty()
        st.success("🎉 Item processing complete! Added directly to cloud closet.")

        col1, col2 = st.columns(2)
        with col1:
            st.image(clean_img, caption="Processed Cutout")
        with col2:
            st.markdown(f"**Category:** {metadata['category'].title()}")
            st.markdown(f"**Color Profile:** {metadata['color'].title()}")
            st.markdown(f"**Style Tag:** {metadata['style'].title()}")

    except Exception as error:
        status_box.empty()
        st.error(f"Execution failed: {str(error)}")
