import gradio as gr
from PIL import Image
import base64
import io
import asyncio
import os
from emergentintegrations.llm.chat import LlmChat, UserMessage, ImageContent

# Load API KEY from environment (MUCH safer)
API_KEY = os.getenv("EMERGENT_API_KEY")

async def classify_leaf_async(image):
    if image is None:
        return "❌ Please upload a leaf image first."

    if API_KEY is None:
        return "❌ ERROR: No API key found. Please add EMERGENT_API_KEY in HuggingFace Secrets."

    try:
        # Convert image to base64
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode()

        # Initialize chatbot
        chat = LlmChat(
            api_key=API_KEY,
            session_id="leaf-classify",
            system_message="You are an expert botanist and plant pathologist."
        ).with_model("openai", "gpt-4o")

        # Prompt + image
        image_content = ImageContent(image_base64=img_base64)

        prompt = """Analyze this leaf image and return:

- Common & Scientific name
- Medicinal or Regular plant
- Healthy or Diseased
- If diseased → name, treatment, prevention
- If medicinal → uses, benefits, preparation

Be clear and accurate."""

        user_message = UserMessage(text=prompt, file_contents=[image_content])

        response = await chat.send_message(user_message)
        return response

    except Exception as e:
        return f"❌ Error: {str(e)}"


def classify_leaf(image):
    return asyncio.run(classify_leaf_async(image))


with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🌿 LeafID - AI Leaf Classifier")
    gr.Markdown("### Powered by GPT-4o Vision through Emergent API")

    with gr.Row():
        with gr.Column():
            image_input = gr.Image(type="pil", label="📸 Upload Leaf Image")
            classify_btn = gr.Button("🔍 Classify Leaf", variant="primary")

        with gr.Column():
            output = gr.Textbox(label="📊 Results", lines=20)

    classify_btn.click(fn=classify_leaf, inputs=image_input, outputs=output)

demo.launch()
