import streamlit as st
from PIL import Image # To handle image display for the logo
import requests # For making API calls
import os
import google.generativeai as genai # Import the Google Gemini SDK
import base64 # For handling image data from Stability AI if it's base64 encoded
import io # For handling image bytes

# --- Access API Keys from Streamlit Secrets ---
# For Streamlit Cloud, you'll set these in your secrets.toml file
# Example secrets.toml content:
# GEMINI_API_KEY = "your_actual_gemini_api_key"
# ELEVENLABS_API_KEY = "your_actual_elevenlabs_api_key"
# STABILITY_API_KEY = "your_actual_stability_ai_api_key"

GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY", "")
ELEVENLABS_API_KEY = st.secrets.get("ELEVENLABS_API_KEY", "")
STABILITY_API_KEY = st.secrets.get("STABILITY_API_KEY", "")
# --- End of API Key Access ---


# --- Stability AI Configuration ---
# You MUST choose a specific engine/model ID for Stability AI.
# Check Stability AI documentation for available and suitable engines.
# Examples: "stable-diffusion-v1-6", "stable-diffusion-xl-1024-v1-0" (for SDXL), etc.
STABILITY_ENGINE_ID = "stable-diffusion-xl-1024-v1-0" # <<< EXAMPLE: CHOOSE YOUR ENGINE
STABILITY_API_HOST = os.getenv('STABILITY_API_HOST', 'https://api.stability.ai')
STABILITY_API_ENDPOINT_TEXT_TO_IMAGE = f"{STABILITY_API_HOST}/v1/generation/{STABILITY_ENGINE_ID}/text-to-image"


# --- Configure Text Engine API (Gemini) ---
GEMINI_CONFIGURED = False # Renamed for clarity, though variable name is internal
if GEMINI_API_KEY: # Check if the key was successfully fetched from secrets
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        # Using a recent and capable model. You can choose other models like 'gemini-pro'.
        gemini_text_model = genai.GenerativeModel('gemini-1.5-flash-latest')
        GEMINI_CONFIGURED = True
        # st.sidebar.success("Text Engine API Configured.") # Optionally enable for debugging
    except Exception as e:
        st.sidebar.error(f"Text Engine API Config Failed: {e}")
else:
    st.sidebar.warning("Text Engine API Key not found in Secrets. Text generation may not function.")

# --- Configure Image Engine API (Stability AI) ---
STABILITY_AI_CONFIGURED = False # Renamed for clarity, though variable name is internal
if STABILITY_API_KEY: # Check if the key was successfully fetched from secrets
    STABILITY_AI_CONFIGURED = True
    # st.sidebar.success("Image Engine API Key Present.") # Optionally enable for debugging
else:
    st.sidebar.warning("Image Engine API Key not found in Secrets. Image generation may not function.")

# --- Configure Voice Engine API (ElevenLabs) ---
ELEVENLABS_CONFIGURED = False
if ELEVENLABS_API_KEY:
    ELEVENLABS_CONFIGURED = True
    # st.sidebar.success("Voice Engine API Key Present.")
else:
    st.sidebar.warning("Voice Engine API Key not found in Secrets. Voice generation may not function.")


# --- 0. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="KPMG Content Generation Suite",
    page_icon="kpmg_logo.png" if os.path.exists("kpmg_logo.png") else "🖼️", # Use logo as icon if available
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- 1. LOGO AND APPLICATION TITLE ---
try:
    # Ensure your KPMG logo file is in the same directory as your script, or provide the full path.
    # For Streamlit Cloud, make sure this file is in your GitHub repository.
    logo = Image.open("kpmg_logo.png")
    st.sidebar.image(logo, use_column_width=True)
except FileNotFoundError:
    st.sidebar.warning("KPMG logo (kpmg_logo.png) not found. Make sure it's in your repository.")

st.sidebar.title("KPMG Content Suite") 
st.sidebar.markdown("Powered by AI")
st.sidebar.markdown("---") 

# --- 2. SIDEBAR NAVIGATION ---
app_mode = st.sidebar.selectbox(
    "Choose a Tool:", 
    [
        "Text Generation (Social Posts, Captions, Taglines)",
        "Text-to-Image Creation (AI Image Engine)",
        "Video Creation (Text/Image to Video with Audio)",
        "Adaptation Generation (Platform-specific Content)",
    ],
    key="app_mode_selector" 
)
st.sidebar.markdown("---")
st.sidebar.info( 
    "Ensure API keys are set in Streamlit Cloud Secrets for full functionality."
)


# --- 3. HELPER FUNCTIONS ---

def ai_text_generation(prompt: str, use_case: str) -> str: # Renamed function
    """
    Generates text using the AI Text Engine.
    Args:
        prompt (str): The user's input prompt.
        use_case (str): The specific use case (e.g., "social media post").
    Returns:
        str: The generated text or an error/placeholder message.
    """
    if not GEMINI_CONFIGURED: # Internal flag name can remain
        return f"Placeholder: Generated {use_case} for: '{prompt}' (Text Engine API Key not configured in Secrets)"
    try:
        # Constructing a more detailed and persona-driven prompt for KPMG
        full_prompt = (
            f"You are an AI assistant for KPMG, a leading global network of professional firms "
            f"providing Audit, Tax, and Advisory services. Your task is to generate content that is "
            f"professional, insightful, clear, concise, and aligns with KPMG's brand values of "
            f"integrity, excellence, courage, togetherness, and for better. "
            f"Avoid overly casual language or jargon where possible, unless appropriate for the specific platform. "
            f"The request is to create a {use_case} based on the following: {prompt}"
        )
        response = gemini_text_model.generate_content(full_prompt)
        return response.text.strip() if response.text else "AI Text Engine returned an empty response."
    except Exception as e:
        st.error(f"AI Text Generation Error: {e}")
        return f"Error generating text with AI Text Engine. Details: {str(e)}"

def ai_image_generation(prompt: str, engine_id: str, height: int = 512, width: int = 512, cfg_scale: float = 7, steps: int = 30, samples: int = 1) -> bytes: # Renamed function
    """
    Generates an image using the AI Image Engine.
    Args:
        prompt (str): The text prompt for image generation.
        engine_id (str): The specific AI Image engine model to use (internal config).
        height (int): Height of the generated image.
        width (int): Width of the generated image.
        cfg_scale (float): Classifier-Free Guidance scale.
        steps (int): Number of diffusion steps.
        samples (int): Number of images to generate.
    Returns:
        bytes: The generated image bytes, or None if an error occurs.
    """
    if not STABILITY_AI_CONFIGURED: # Internal flag name can remain
        st.warning("AI Image Engine API Key not configured in Secrets. Returning placeholder.")
        try: # Return bytes of a placeholder image
            placeholder_url = f"https://placehold.co/{width}x{height}/00338D/FFFFFF?text=AI+Image+Engine+Secret+Missing"
            response = requests.get(placeholder_url)
            response.raise_for_status()
            return response.content
        except Exception as e:
            st.error(f"Failed to fetch placeholder image: {e}")
            return None

    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "Authorization": f"Bearer {STABILITY_API_KEY}", # Internal variable name
    }
    body = {
        "text_prompts": [{"text": f"KPMG professional style, high quality, corporate visual: {prompt}"}],
        "cfg_scale": cfg_scale,
        "height": height,
        "width": width,
        "samples": samples,
        "steps": steps,
    }
    if "xl" in engine_id.lower(): 
        body["height"] = 1024
        body["width"] = 1024
        # body["style_preset"] = "photographic" 

    api_endpoint = f"{STABILITY_API_HOST}/v1/generation/{engine_id}/text-to-image" # Internal variable

    try:
        st.info(f"Sending prompt to AI Image Engine...") # Made generic
        api_response = requests.post(api_endpoint, headers=headers, json=body)
        api_response.raise_for_status() 

        response_json = api_response.json()
        if response_json.get("artifacts") and len(response_json["artifacts"]) > 0:
            image_artifact = response_json["artifacts"][0] 
            if image_artifact.get("base64"):
                st.success("Image generated successfully by AI Image Engine.")
                image_bytes = base64.b64decode(image_artifact["base64"])
                return image_bytes
            else:
                st.error("AI Image Engine: No base64 image data found in the artifact.")
                return None
        else:
            st.error(f"AI Image Engine: No artifacts found in response. Full response: {response_json}")
            return None

    except requests.exceptions.RequestException as e:
        st.error(f"AI Image Engine API Request Error: {e}")
        if 'api_response' in locals() and api_response is not None: 
            st.error(f"Response status: {api_response.status_code}, content: {api_response.text}")
        return None
    except Exception as e: 
        st.error(f"An unexpected error occurred during AI Image Generation: {e}")
        return None


def ai_text_to_speech(text: str, voice_id: str = "21m00Tcm4TlvDq8ikWAM") -> bytes: # Renamed function
    """
    Converts text to speech using the AI Voice Engine.
    Args:
        text (str): The text to convert.
        voice_id (str): The ID of the AI voice to use (internal config).
    Returns:
        bytes: The audio data in bytes, or a placeholder/error message.
    """
    if not ELEVENLABS_CONFIGURED: # Internal flag name can remain
        st.warning("AI Voice Engine API Key not configured in Secrets.")
        return b"Placeholder: Audio (AI Voice Engine Secret Missing)" 

    XI_API_KEY = ELEVENLABS_API_KEY # Internal variable name
    CHUNK_SIZE = 1024 
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}" # Internal URL

    headers = {
        "Accept": "audio/mpeg", 
        "Content-Type": "application/json",
        "xi-api-key": XI_API_KEY
    }
    data = {
        "text": text,
        "model_id": "eleven_multilingual_v2", 
        "voice_settings": {
            "stability": 0.5,
            "similarity_boost": 0.75,
        }
    }
    try:
        api_response = requests.post(url, json=data, headers=headers)
        api_response.raise_for_status() 
        
        audio_bytes = b''
        for chunk in api_response.iter_content(chunk_size=CHUNK_SIZE):
            if chunk:
                audio_bytes += chunk
        return audio_bytes
    except requests.exceptions.RequestException as e:
        st.error(f"AI Voice Engine API Request Error: {e}")
        if 'api_response' in locals() and api_response is not None:
            st.error(f"Response status: {api_response.status_code}, content: {api_response.text}")
        return None 
    except Exception as e:
        st.error(f"An unexpected error occurred during AI Voice Generation: {e}")
        return None

def generate_video_from_elements(text_prompt: str, image_bytes_list: list = None, audio_bytes: bytes = None) -> str:
    st.info("Video Generation Logic (e.g., using MoviePy) is a placeholder.")
    st.info(f"Received text prompt for video context: {text_prompt[:100]}...")
    if image_bytes_list:
        st.info(f"Received {len(image_bytes_list)} image(s) (as bytes data) for the video.")
    if audio_bytes:
        st.info(f"Received audio data (length: {len(audio_bytes)} bytes) for the video.")

    if audio_bytes and image_bytes_list:
        st.success("Placeholder: Video elements received. Simulating video generation.")
        # TODO: Implement actual video generation using MoviePy
        return "https://www.w3schools.com/html/mov_bbb.mp4" 
    else:
        st.warning("Not enough elements to generate video (requires images and audio).")
        return None

def adapt_content_for_platform_ai(content: str, platform: str) -> str: # Renamed function
    """
    Adapts content for a specific platform using AI Text Engine.
    Args:
        content (str): The original content to adapt.
        platform (str): The target platform (e.g., "LinkedIn Post").
    Returns:
        str: The adapted content or an error/placeholder message.
    """
    use_case_for_ai = f"content adaptation for a {platform}"
    prompt_for_ai = (
        f"Please adapt the following original content to be perfectly suited for a {platform}. "
        f"Consider the typical audience, tone, length, and formatting conventions of {platform}.\n\n"
        f"Original Content:\n---\n{content}\n---\n\n"
        f"Adapted Content for {platform}:"
    )
    return ai_text_generation(prompt_for_ai, use_case_for_ai)


# --- 4. MAIN APPLICATION SECTIONS ---

# --- 4.1 Text Generation ---
if app_mode == "Text Generation (Social Posts, Captions, Taglines)":
    st.header("Text Generation") 
    st.markdown("Generate compelling text for various marketing needs, maintaining KPMG's professional tone.")

    text_use_case_options = [
        "LinkedIn Post", "Twitter (X) Post (max 280 characters)", "Instagram Caption",
        "Product/Service Tagline (short and catchy)", "Email Subject Line (concise and engaging)",
        "Blog Post Introduction (approx. 100-150 words)", "Executive Summary (brief overview)"
    ]
    text_use_case = st.selectbox(
        "Select Use Case:", 
        text_use_case_options,
        key="text_use_case_selector" 
    )
    text_prompt_input = st.text_area(
        "Enter your topic, keywords, or a brief description:",
        height=150,
        key="text_gen_prompt_input", 
        placeholder="e.g., Key insights from our latest cybersecurity report for financial institutions..."
    )

    if st.button("Generate Text", key="text_gen_button", type="primary"): 
        if text_prompt_input:
            with st.spinner(f"Generating {text_use_case} with AI Text Engine..."): 
                generated_text = ai_text_generation(text_prompt_input, text_use_case)
                st.subheader("Generated Text:")
                st.markdown(f"> {generated_text}") 
                st.download_button("Download Text", generated_text, file_name=f"{text_use_case.replace(' ', '_')}.txt")
        else:
            st.warning("Please enter a prompt for text generation.")


# --- 4.2 Text-to-Image Creation ---
elif app_mode == "Text-to-Image Creation (AI Image Engine)": # Updated title
    st.header("🖼️ Text-to-Image Creation with AI Image Engine") 
    st.markdown("Create unique visuals using our advanced AI image generation model. "
                "Ensure the required API key is set in Streamlit Cloud Secrets.") 

    image_prompt_input = st.text_area(
        "Describe the image you want to create:",
        height=100,
        key="image_gen_prompt_stability", 
        placeholder="e.g., A diverse team of professionals collaborating in a bright, modern KPMG office..."
    )
    
    with st.expander("Advanced Image Creation Settings (Optional)"): 
        cfg_scale = st.slider("CFG Scale (Adherence to Prompt)", min_value=1.0, max_value=20.0, value=7.0, step=0.5, key="cfg_scale_stability")
        steps = st.slider("Diffusion Steps (Quality vs. Speed)", min_value=10, max_value=100, value=30, step=5, key="steps_stability") 


    if st.button("Generate Image with AI Image Engine", key="image_gen_button_stability", type="primary"): 
        if image_prompt_input:
            if not STABILITY_AI_CONFIGURED: # Internal flag
                st.error("AI Image Engine API Key is not configured in Streamlit Secrets.") 
            else:
                with st.spinner(f"Generating image with AI Image Engine... This may take a moment."): 
                    image_bytes = ai_image_generation(
                        image_prompt_input,
                        engine_id=STABILITY_ENGINE_ID, # Internal variable
                        cfg_scale=cfg_scale,
                        steps=steps
                    )
                    st.subheader("Generated Image:")
                    if image_bytes:
                        st.image(image_bytes, caption=f"Generated by AI Image Engine for: '{image_prompt_input}'")
                        st.download_button(
                            label="Download Image",
                            data=image_bytes,
                            file_name=f"ai_generated_image_{image_prompt_input[:20].replace(' ','_')}.png",
                            mime="image/png"
                        )
                    else:
                        st.error("Image generation failed or no image was returned by AI Image Engine. Check console for errors.")
        else:
            st.warning("Please enter a prompt for image generation.")

# --- 4.3 Video Creation ---
elif app_mode == "Video Creation (Text/Image to Video with Audio)":
    st.header("🎬 Video Creation for KPMG")
    st.markdown(
        "**Step 1:** Generate script/narration using AI Text Engine. \n" 
        "**Step 2:** Generate images for scenes using AI Image Engine. \n"
        "**Step 3:** Generate voiceover for the script using AI Voice Engine. \n"
        "**Step 4:** (Placeholder) Combine elements into a video."
    )

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Step 1: Script (AI Text Engine)")
        video_topic = st.text_input("Enter video topic/main message:", key="video_topic_input", placeholder="e.g., The future of AI in auditing")
        if 'video_script' not in st.session_state:
            st.session_state.video_script = ""

        if st.button("Generate Script", key="video_script_gen_button") and video_topic:
            with st.spinner("Generating script with AI Text Engine..."):
                script_prompt_detail = (
                    f"Create a concise and engaging video script suitable for a professional KPMG audience. "
                    f"The script should be approximately 30-60 seconds when narrated. "
                    f"Topic: {video_topic}. Break it into 2-4 short paragraphs or scenes."
                )
                st.session_state.video_script = ai_text_generation(script_prompt_detail, "video script")

        st.text_area("Generated Script:", value=st.session_state.video_script, height=200, key="video_script_display_area", disabled=not st.session_state.video_script)

    with col2:
        st.subheader("Step 2: Visuals (AI Image Engine)")
        if 'image_bytes_for_video' not in st.session_state:
            st.session_state.image_bytes_for_video = []

        num_images = st.slider("Number of images for video scenes:", 1, 5, 2, key="video_num_images_slider", help="Images will be generated based on script segments.")
        
        if st.button("Generate Scene Images", key="video_gen_images_button") and st.session_state.video_script:
            if not STABILITY_AI_CONFIGURED: # Internal flag
                st.error("AI Image Engine API Key is not configured in Streamlit Secrets. Cannot generate images.")
            else:
                with st.spinner(f"Generating {num_images} images with AI Image Engine..."):
                    script_parts = [part.strip() for part in st.session_state.video_script.split('\n') if part.strip()]
                    prompts_for_images = [f"KPMG style professional visual, cinematic, for video scene about: {part[:120]}" for part in script_parts[:num_images]]
                    
                    if not prompts_for_images and num_images > 0: 
                        prompts_for_images = [f"KPMG professional visual, abstract theme {i+1} for video" for i in range(num_images)]
                    
                    temp_image_bytes_list = []
                    for i, p_img in enumerate(prompts_for_images):
                        st.write(f"Generating image {i+1}/{len(prompts_for_images)}: {p_img[:70]}...")
                        img_bytes = ai_image_generation(p_img, engine_id=STABILITY_ENGINE_ID) 
                        if img_bytes:
                            temp_image_bytes_list.append(img_bytes)
                        else:
                            st.error(f"Failed to generate image {i+1} with AI Image Engine.")
                    st.session_state.image_bytes_for_video = temp_image_bytes_list
        
        uploaded_files_video = st.file_uploader(
            "Or upload your own images for the video:",
            type=["png", "jpg", "jpeg"],
            accept_multiple_files=True,
            key="video_image_uploader"
        )
        if uploaded_files_video:
            st.session_state.image_bytes_for_video = [] 
            for uploaded_file_v in uploaded_files_video:
                st.session_state.image_bytes_for_video.append(uploaded_file_v.getvalue())
        
        if st.session_state.image_bytes_for_video:
            st.write("Images queued for video:")
            cols_img_preview = st.columns(min(len(st.session_state.image_bytes_for_video), 4))
            for i, img_bytes_data_v in enumerate(st.session_state.image_bytes_for_video):
                cols_img_preview[i % 4].image(img_bytes_data_v, width=100, caption=f"Scene {i+1}")


    st.markdown("---") 
    st.subheader("Step 3: Voiceover (AI Voice Engine)") 
    if 'video_audio_bytes' not in st.session_state:
        st.session_state.video_audio_bytes = None

    if st.button("Generate Voiceover from Script", key="video_audio_gen_button") and st.session_state.video_script:
        with st.spinner("Generating voiceover with AI Voice Engine..."): 
            audio_data = ai_text_to_speech(st.session_state.video_script)
            if audio_data and not (isinstance(audio_data, bytes) and audio_data.startswith(b"Placeholder")):
                st.session_state.video_audio_bytes = audio_data
                st.audio(st.session_state.video_audio_bytes, format="audio/mpeg")
            elif isinstance(audio_data, bytes) and audio_data.startswith(b"Placeholder"):
                 st.warning(audio_data.decode()) 
            else:
                st.error("Voiceover generation failed or returned no valid data.")
    elif st.session_state.video_audio_bytes: 
        st.audio(st.session_state.video_audio_bytes, format="audio/mpeg")


    st.markdown("---")
    st.subheader("Step 4: Generate Video (Placeholder)")
    if st.button("Combine Elements into Video", key="video_final_gen_button", type="primary"):
        if st.session_state.get('image_bytes_for_video') and st.session_state.get('video_audio_bytes') and st.session_state.get('video_script'):
            with st.spinner("Generating video (placeholder)... This will show a sample video."):
                video_path_result = generate_video_from_elements(
                    st.session_state.video_script,
                    image_bytes_list=st.session_state.image_bytes_for_video,
                    audio_bytes=st.session_state.video_audio_bytes
                )
                if video_path_result:
                    st.video(video_path_result)
                else:
                    st.error("Video generation placeholder failed or prerequisites not met.")
        else:
            missing_elements = []
            if not st.session_state.get('video_script'): missing_elements.append("Script")
            if not st.session_state.get('image_bytes_for_video'): missing_elements.append("Images")
            if not st.session_state.get('video_audio_bytes'): missing_elements.append("Audio")
            st.warning(f"Please generate/provide all elements before combining: {', '.join(missing_elements)} are missing.")

# --- 4.4 Adaptation Generation ---
elif app_mode == "Adaptation Generation (Platform-specific Content)":
    st.header("🔄 Content Adaptation with AI") 
    st.markdown("Tailor existing content for different platforms using our AI Text Engine, maintaining KPMG's professional tone.")

    original_content = st.text_area(
        "Paste your original content here:",
        height=200,
        key="adapt_orig_content_input", 
        placeholder="Enter the content you wish to adapt..."
    )
    target_platform_options = [
        "LinkedIn Post (professional, insightful)", "Twitter (X) Post (concise, engaging, with hashtags)",
        "YouTube Video Description (SEO-friendly, with chapters if applicable)",
        "Internal Company Announcement (clear, direct)", "Website Blog Snippet (engaging summary)"
    ]
    target_platform = st.selectbox(
        "Select Target Platform:", 
        target_platform_options,
        key="adapt_platform_selector" 
    )

    if st.button("Adapt Content with AI", key="adapt_gen_button", type="primary"): 
        if original_content and target_platform:
            with st.spinner(f"Adapting content for {target_platform.split(' (')[0]} with AI Text Engine..."):
                adapted_content = adapt_content_for_platform_ai(original_content, target_platform.split(' (')[0])
                st.subheader(f"Adapted Content for {target_platform.split(' (')[0]}:")
                st.markdown(f"> {adapted_content}")
                st.download_button("Download Adapted Text", adapted_content, file_name=f"adapted_for_{target_platform.split(' (')[0].replace(' ', '_')}.txt")
        else:
            st.warning("Please provide the original content and select a target platform.")

# --- 5. FOOTER (Optional) ---
st.markdown("---") 
st.markdown(
    "<div style='text-align: center; color: #555; font-size: 0.9em; padding: 10px;'>" 
    "KPMG Content Suite | AI-Powered Content Generation and Adaptation<br>" 
    "&copy; KPMG 2025. For internal use and demonstration purposes." 
    "</div>",
    unsafe_allow_html=True
)
