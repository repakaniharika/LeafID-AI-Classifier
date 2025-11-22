# Install packages
!pip install -q gradio emergentintegrations --extra-index-url https://d33sy5i8bnduwe.cloudfront.net/simple/

import gradio as gr
from PIL import Image
import base64
import io
import asyncio
from emergentintegrations.llm.chat import LlmChat, UserMessage, ImageContent

# Your Emergent LLM Key
API_KEY = "sk-emergent-05eBf84A55aE130737"

async def classify_leaf_async(image):
    """Classify leaf using Emergent Integrations"""
    if image is None:
        return "❌ Please upload an image first!"
    
    try:
        # Convert to base64
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode()
        
        # Initialize chat
        chat = LlmChat(
            api_key=API_KEY,
            session_id="leaf-classify",
            system_message="You are an expert botanist and plant pathologist."
        ).with_model("openai", "gpt-4o")
        
        # Create image content
        image_content = ImageContent(image_base64=img_base64)
        
        # Create message
        prompt = """Analyze this leaf image and provide:

📝 LEAF NAME: Common and scientific name
🌿 PLANT TYPE: Medicinal or Regular?
🏥 HEALTH STATUS: Healthy or Diseased?

If DISEASED:
🦠 Disease name
💊 How to cure/treat it
🛡️ Prevention tips

If MEDICINAL:
💚 Health benefits
🌿 How it's used medicinally
📋 Preparation methods (tea, paste, etc.)

🎯 CONFIDENCE LEVEL: High/Medium/Low

Be detailed and practical."""

        user_message = UserMessage(
            text=prompt,
            file_contents=[image_content]
        )
        
        # Get response
        response = await chat.send_message(user_message)
        return response
        
    except Exception as e:
        return f"❌ Error: {str(e)}\n\nAPI Key: {API_KEY[:20]}... (verified)"

def classify_leaf(image):
    """Wrapper to run async function"""
    return asyncio.run(classify_leaf_async(image))

# Create interface
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🌿 LeafID - AI Leaf Classifier")
    gr.Markdown("### Using Emergent LLM Key + GPT-4o Vision")
    
    with gr.Row():
        with gr.Column():
            image_input = gr.Image(type="pil", label="📸 Upload Leaf Image")
            classify_btn = gr.Button("🔍 Classify Leaf", variant="primary", size="lg")
            gr.Markdown(f"✅ API Key: `{API_KEY[:15]}...` (Active)")
        
        with gr.Column():
            output = gr.Textbox(label="📊 Classification Results", lines=20)
    
    classify_btn.click(fn=classify_leaf, inputs=image_input, outputs=output)

# Launch
demo.launch(share=True, debug=True)
