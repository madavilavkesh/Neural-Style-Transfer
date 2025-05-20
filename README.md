# Neural-Style-Transfer

This project demonstrates **Neural Style Transfer (NST)** using both **Classical (VGG-based)** and **Fast (TF Hub pre-trained)** methods. It allows you to blend the content of one image with the style of another — turning your photos into artwork!

---

## 🚀 Live Demo

Try the app here (Fast Style Transfer):

👉 [Gradio App on Hugging Face](https://huggingface.co/spaces/madavilavkesh/Neural-Style-Transfer)

---

## 📌 Project Highlights

- ✅ Classical NST using VGG19 with optimization
- ✅ Fast NST using TensorFlow Hub (Magenta model)
- ✅ Gradio UI for interactive image generation
- ✅ Live demo hosted on Hugging Face Spaces
- ✅ GPU support (optional)

---

## 📂 Repository Structure

| File | Description |
|------|-------------|
| `app.py` | Gradio app for fast style transfer with GPU toggle |
| `classic_NST.py` | Classic NST with loss optimization (content/style/TV loss) |
| `fast_NST.py` | Fast NST using pre-trained model from TF Hub |
| `requirements.txt` | Required Python dependencies |
| `screenshots/` | UI and stylization result previews |

---

## 🖼️ Screenshots

| Gradio UI | Style Transfer Demo |
|-----------|---------------------|
| ![UI](screenshots/Screenshot_App_UI.png) | ![Demo1](screenshots/Screenshot_Demo_1.png) |
|  | ![Demo2](screenshots/Screenshot_Demo_2.png) |

---

## 🧪 Methodology

### 🎯 Goal
Blend two images — a **content image** and a **style image** — to generate a new image that maintains the content of the first and the artistic style of the second.

### 1. **Classical Style Transfer (VGG19-based)**
- Uses pre-trained **VGG19** model
- Optimizes total loss = content loss + style loss + total variation loss
- Highly tunable with weights and iteration count
- Slower but customizable and research-oriented

### 2. **Fast Style Transfer (TF Hub Model)**
- Uses pre-trained feed-forward model from **TensorFlow Hub**
- Real-time stylization
- Faster, best for live demos & web apps
- Limited control over fine-tuning

---

## ⚙️ How to Run

### 🔧 Requirements

```bash
pip install -r requirements.txt

