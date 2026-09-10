import os
import torch
import numpy as np
import cv2
import json
import gradio as gr
from PIL import Image

from ultralytics import YOLO
from model import HOIDetector
from config import CONFIG
from torchvision import transforms

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 1. Load the HOI Model
print("Loading HOI Model...")
hoi_model = HOIDetector().to(device)
ckpt = torch.load("best.pt", map_location=device)
hoi_model.load_state_dict(ckpt['state_dict'])
hoi_model.eval()

# 2. Load YOLOv11
print("Loading YOLO Model...")
yolo_model = YOLO("yolo11x.pt") # It will download automatically if missing

# 3. Load HOI List
with open("hoi_list.json", "r") as f:
    hoi_list = json.load(f)

# Image preprocessor for CLIP
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=CONFIG['clip_mean'], std=CONFIG['clip_std'])
])

def predict_hoi(image):
    # image is a PIL Image
    image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    H, W, _ = image_cv.shape

    # 1. Run YOLO
    results = yolo_model(image_cv, verbose=False)[0]
    
    humans = []
    objects = []
    
    for box in results.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        if conf < 0.3: continue
            
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        
        # Normalize to [0, 1]
        norm_box = [x1/W, y1/H, x2/W, y2/H]
        
        if cls_id == 0: # Person
            humans.append(norm_box)
        else:
            objects.append(norm_box)
            
    if len(humans) == 0 or len(objects) == 0:
        return image, "No human-object pairs found by YOLO."

    # 2. Prepare inputs for HOI model
    # We will pair every human with every object
    hb_list = []
    ob_list = []
    for h in humans:
        for o in objects:
            hb_list.append(h)
            ob_list.append(o)
            
    hb_tensor = torch.tensor(hb_list, dtype=torch.float32).to(device)
    ob_tensor = torch.tensor(ob_list, dtype=torch.float32).to(device)
    
    img_tensor = preprocess(image).unsqueeze(0).to(device)
    # Expand image tensor to match number of pairs
    img_tensor = img_tensor.expand(len(hb_list), -1, -1, -1)

    # 3. Run HOI Model
    with torch.no_grad():
        with torch.amp.autocast('cuda'):
            out = hoi_model(img_tensor, hb_tensor, ob_tensor, use_gt_boxes=True)
            
    preds = torch.sigmoid(out['final_hoi']).cpu().numpy() # (N_pairs, 600)

    # 4. Draw results
    draw_img = np.array(image).copy()
    
    top_predictions_text = []
    
    for i in range(len(hb_list)):
        pred_class = np.argmax(preds[i])
        pred_score = preds[i, pred_class]
        
        if pred_score > 0.15: # Confidence threshold
            verb = hoi_list[pred_class]['verb']
            obj_name = hoi_list[pred_class]['object']
            label = f"{verb} {obj_name} ({pred_score*100:.1f}%)"
            
            # Un-normalize boxes for drawing
            hx1, hy1, hx2, hy2 = [int(v * d) for v, d in zip(hb_list[i], [W, H, W, H])]
            ox1, oy1, ox2, oy2 = [int(v * d) for v, d in zip(ob_list[i], [W, H, W, H])]
            
            # Draw Human (Red)
            cv2.rectangle(draw_img, (hx1, hy1), (hx2, hy2), (255, 0, 0), 2)
            cv2.putText(draw_img, "Human", (hx1, max(0, hy1-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            
            # Draw Object (Blue)
            cv2.rectangle(draw_img, (ox1, oy1), (ox2, oy2), (0, 0, 255), 2)
            cv2.putText(draw_img, label, (ox1, max(0, oy1-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            # Draw line connecting them
            hcx, hcy = (hx1+hx2)//2, (hy1+hy2)//2
            ocx, ocy = (ox1+ox2)//2, (oy1+oy2)//2
            cv2.line(draw_img, (hcx, hcy), (ocx, ocy), (0, 255, 0), 2)
            
            top_predictions_text.append(label)

    if len(top_predictions_text) == 0:
        return draw_img, "No confident interactions detected."
        
    return draw_img, "\n".join(top_predictions_text)

# 5. Build Gradio Interface
demo = gr.Interface(
    fn=predict_hoi,
    inputs=gr.Image(type="pil"),
    outputs=[gr.Image(type="numpy"), gr.Textbox(label="Detected Interactions")],
    title="Multimodal Human-Object Interaction (HOI) Detection",
    description="Upload an image to see the model detect humans, objects, and their interactions!"
)

if __name__ == "__main__":
    demo.launch(share=True)
