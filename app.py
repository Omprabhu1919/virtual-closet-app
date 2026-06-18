import streamlit as st
from PIL import Image
import io
from supabase import create_client
import google.generativeai as genai
from pydantic import BaseModel
import json
import uuid

# 1. Page Config
st.set_page_config(page_title="Cloud Wardrobe", layout="centered")

# 2. Output Schema
class ClothingMetadata(BaseModel):
    category: str
    color: str
    style: str

# 3. Initialize Clients
@st.cache_resource
def init_clients():
    supabase_client = create_client(st.secrets["SUPABASE_URL"], st.secrets["SUPABASE_KEY"])
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    return supabase_client, genai

try:
    supabase, genai = init_clients()
except Exception as e:
    st.error("Configuration error. Please check your Streamlit Secrets.")
    st.stop()

# 4. Helpers
def classify_clothing(model, image_for_ai: Image.Image) -> dict:
    generation_config = {
        "response_mime_type": "application/json",
        "response_schema": ClothingMetadata,
    }
    response = model.generate_content(
        ["Analyze this clothing item precisely. Identify its category, dominant color, and style.", image_for_ai],
        generation_config=generation_config,
    )
    
    parsed = getattr(response, "parsed", None)
    if parsed is not None:
        data = parsed.model_dump() if isinstance(parsed, BaseModel) else parsed
    else:
        text = response.text.strip()
        if text.startswith("```"):
            text = text.strip("`")
            text = text.split("\n", 1)[-1] if "\n" in text else text
        data = json.loads(text)
    return ClothingMetadata(**data).model_dump()

def upload_to_supabase(client, file_bytes: bytes, original_filename: str) -> str:
    safe_stem = original_filename.rsplit(".", 1)[0] if "." in original_filename else original_filename
    unique_name = f"{safe_stem}_{uuid.uuid4().hex[:8]}.png"
    client.storage.from_("wardrobe").upload(unique_name, file_bytes, file_options={"content-type": "image/png"})
    
    url_result = client.storage.from_("wardrobe").get_public_url(unique_name)
    if isinstance(url_result, dict):
        return url_result.get("publicURL") or url_result.get("publicUrl") or str(url_result)
    return url_result

# 5. UI
st.title("👗 My Cloud Wardrobe")
st.write("Upload an item to auto-tag it with AI and sync it to your cloud storage.")

uploaded_file = st.file_uploader("Snap or upload a photo of your clothing item", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    status_box = st.empty()
    if uploaded_file.size == 0:
        st.error("The uploaded file is empty. Please choose a different photo.")
        st.stop()

    try:
        status_box.info("⚡ Preparing image...")
        raw_img = Image.open(uploaded_file)
        raw_img.thumbnail((800, 800))
        
        img_byte_arr = io.BytesIO()
        raw_img.save(img_byte_arr, format="PNG")
        img_bytes = img_byte_arr.getvalue()

        status_box.info("🤖 AI is analyzing features and color schemas...")
        model = genai.GenerativeModel("gemini-2.0-flash")
        metadata = classify_clothing(model, raw_img.convert("RGB"))

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
            st.image(raw_img, caption="Uploaded Item")
        with col2:
            st.markdown(f"**Category:** {metadata['category'].title()}")
            st.markdown(f"**Color Profile:** {metadata['color'].title()}")
            st.markdown(f"**Style Tag:** {metadata['style'].title()}")

    except Exception as error:
        status_box.empty()
        st.error(f"Execution failed: {str(error)}")
