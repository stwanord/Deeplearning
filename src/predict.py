import torch
from torchvision import transforms
from PIL import Image
from model import FireDetectionCNN
import gradio as gr
import os


MODEL_PATH = "fire_model.pth"
CLASS_NAMES = ['Fire', 'Neutral', 'Smoke'] 

def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = FireDetectionCNN(num_classes=len(CLASS_NAMES))
    
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        print("Model loaded successfully.")
    else:
        print(f"Model file '{MODEL_PATH}' not found. Please train the model first.")
     
        
    model.to(device)
    model.eval()
    return model, device

model, device = load_model()

def predict_image(image):
    if image is None:
        return "Please upload an image."

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    img_tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        
        top_prob, top_class = torch.topk(probabilities, 1)
        class_idx = top_class.item()
        score = top_prob.item()
        
        result_text = f"Prediction: {CLASS_NAMES[class_idx]} ({score*100:.2f}%)"
       
        details = {CLASS_NAMES[i]: float(probabilities[0][i]) for i in range(len(CLASS_NAMES))}
        
    return details

if __name__ == "__main__":
    interface = gr.Interface(
        fn=predict_image,
        inputs=gr.Image(type="pil"),
        outputs=gr.Label(num_top_classes=3),
        title="Fire & Smoke Detection System",
        description="Upload an image to detect Fire, Smoke, or Neutral state.",
        examples=[] 
    )
    
    interface.launch(share=True)
