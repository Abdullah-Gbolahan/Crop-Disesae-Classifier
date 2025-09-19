import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import efficientnet_b0
import numpy as np
from PIL import Image
import plotly.express as px
import pandas as pd
import json
import io

# Configure page
st.set_page_config(
    page_title="🌱 Crop Disease Classifier",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configuration
MODEL_PATH = 'model/crop_disease_model.pth'
IMG_SIZE = (224, 224)  # EfficientNetB0 input size
MAX_FILE_SIZE = 5 * 1024 * 1024  # 5MB

# Your actual crop disease classes
DEFAULT_CLASSES = [
    'cashew anthracnose', 'cashew gumosis', 'cashew healthy', 'cashew leaf miner', 'cashew red rust', 
    'cassava bacterial blight', 'cassava brown spot', 'cassava green mite', 'cassava healthy', 'cassava mosaic', 
    'maize fall armyworm', 'maize grasshoper', 'maize healthy', 'maize leaf beetle', 'maize leaf blight', 
    'maize leaf spot', 'maize streak virus', 'tomato healthy', 'tomato leaf blight', 'tomato leaf curl', 
    'tomato septoria leaf spot', 'tomato verticulium wilt'
]

# Image preprocessing transforms for EfficientNetB0
transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225])
])

@st.cache_resource
def load_model():
    """Load the PyTorch crop disease classification model"""
    try:
        # Set device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load class names to get number of classes
        class_names = load_class_names()
        num_classes = len(class_names)
        
        # Load checkpoint first to inspect the architecture
        checkpoint = torch.load(MODEL_PATH, map_location=device)
        
        # Handle different checkpoint formats
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint
        
        # Create model architecture (EfficientNetB0)
        model = efficientnet_b0(weights=None)  # Updated from pretrained=False
        
        # Recreate your exact classifier architecture
        # (0): Linear(in_features=1280, out_features=256, bias=True)
        # (1): ReLU()
        # (2): Dropout(p=0.5, inplace=False)
        # (3): Linear(in_features=256, out_features=22, bias=True)
        model.classifier = nn.Sequential(
            nn.Linear(1280, 256),  # classifier.0
            nn.ReLU(),             # classifier.1
            nn.Dropout(p=0.5, inplace=False),  # classifier.2
            nn.Linear(256, num_classes)  # classifier.3
        )
        
        # Load the state dict
        model.load_state_dict(state_dict)
        
        model.to(device)
        model.eval()
        
        return model, device
    
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.info("Make sure your model file is saved as 'model/crop_disease_model.pth'")
        return None, None

@st.cache_data
def load_class_names():
    """Load class names - customize this based on your actual classes"""
    try:
        # Try to load from JSON file if available
        with open('model/class_names.json', 'r') as f:
            class_names = json.load(f)
        return class_names
    except:
        # Fallback to default classes - UPDATE THESE WITH YOUR ACTUAL CLASSES
        return DEFAULT_CLASSES

def preprocess_image(uploaded_file):
    """Preprocess uploaded crop image for PyTorch EfficientNetB0 model"""
    try:
        # Load and convert image
        image = Image.open(uploaded_file)
        
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Store original for display
        original_image = image.copy()
        
        # Apply transforms
        processed_image = transform(image)
        
        # Add batch dimension
        processed_image = processed_image.unsqueeze(0)
        
        return processed_image, original_image
    
    except Exception as e:
        st.error(f"Error preprocessing image: {str(e)}")
        return None, None

def make_prediction(model, processed_image, class_names, device):
    """Make prediction on crop disease using PyTorch model"""
    try:
        # Move image to device
        processed_image = processed_image.to(device)
        
        # Make prediction
        with torch.no_grad():
            outputs = model(processed_image)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            predicted_class_idx = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_class_idx].item()
        
        # Create results dictionary
        results = {
            'predicted_class': class_names[predicted_class_idx],
            'confidence': float(confidence),
            'all_predictions': {
                class_names[i]: float(probabilities[0][i].item()) 
                for i in range(len(class_names))
            }
        }
        
        return results
    
    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")
        return None

def get_disease_info(disease_name):
    """Get comprehensive information about the detected disease"""
    disease_info = {
        # CASHEW DISEASES
        "cashew anthracnose": {
            "description": "Fungal disease causing dark brown spots on leaves, flowers, and young fruits.",
            "symptoms": ["Dark brown irregular spots on leaves", "Flower and fruit drop", "Premature leaf fall"],
            "causes": "Colletotrichum gloeosporioides fungus",
            "action": "Apply copper-based fungicides during wet season. Prune affected branches and ensure good air circulation.",
            "prevention": "Plant resistant varieties, improve drainage, avoid overhead irrigation during flowering",
            "severity": "Moderate to High",
            "color": "orange",
            "economic_impact": "20-40% yield loss if untreated"
        },
        "cashew gumosis": {
            "description": "Bark disease causing gum exudation from trunk and branches.",
            "symptoms": ["Gum oozing from bark", "Bark cracking and peeling", "Branch dieback"],
            "causes": "Various fungi and environmental stress",
            "action": "Remove affected bark, apply fungicide paste, improve tree nutrition and drainage.",
            "prevention": "Avoid mechanical injuries, proper pruning practices, balanced fertilization",
            "severity": "High",
            "color": "red",
            "economic_impact": "Can cause tree death if severe"
        },
        "cashew healthy": {
            "description": "Cashew plant appears healthy with no visible disease symptoms.",
            "symptoms": ["Green, vigorous leaves", "Normal growth pattern", "No spots or lesions"],
            "causes": "No disease present",
            "action": "Continue regular monitoring and maintain good agricultural practices.",
            "prevention": "Regular inspection, proper nutrition, and integrated pest management",
            "severity": "None",
            "color": "green",
            "economic_impact": "Optimal productivity expected"
        },
        "cashew leaf miner": {
            "description": "Insect pest creating serpentine mines in cashew leaves.",
            "symptoms": ["Serpentine tunnels in leaves", "Leaf yellowing", "Premature leaf drop"],
            "causes": "Leaf miner larvae (Acrocercops syngramma)",
            "action": "Apply systemic insecticides, remove and destroy affected leaves, use pheromone traps.",
            "prevention": "Regular monitoring, maintain field hygiene, encourage natural predators",
            "severity": "Moderate",
            "color": "orange",
            "economic_impact": "10-25% yield reduction due to reduced photosynthesis"
        },
        "cashew red rust": {
            "description": "Fungal disease causing reddish-brown pustules on leaf undersides.",
            "symptoms": ["Reddish-brown pustules on leaf undersides", "Yellow spots on upper surface", "Leaf drop"],
            "causes": "Cephaleuros virescens algae",
            "action": "Apply copper-based fungicides, improve air circulation, reduce humidity around plants.",
            "prevention": "Proper spacing, pruning for air circulation, avoid excessive moisture",
            "severity": "Moderate",
            "color": "orange",
            "economic_impact": "15-30% yield loss in severe cases"
        },
        
        # CASSAVA DISEASES
        "cassava bacterial blight": {
            "description": "Bacterial disease causing angular leaf spots and stem cankers.",
            "symptoms": ["Angular leaf spots with yellow halos", "Stem cankers", "Leaf blight", "Plant wilting"],
            "causes": "Xanthomonas axonopodis bacteria",
            "action": "Remove infected plants, apply copper-based bactericides, use resistant varieties.",
            "prevention": "Use certified disease-free planting material, crop rotation, field sanitation",
            "severity": "High",
            "color": "red",
            "economic_impact": "50-100% yield loss in susceptible varieties"
        },
        "cassava brown spot": {
            "description": "Fungal disease causing brown spots with yellow halos on leaves.",
            "symptoms": ["Brown circular spots with yellow margins", "Premature leaf drop", "Reduced tuber yield"],
            "causes": "Cercospora henningsii fungus",
            "action": "Apply fungicides, remove infected leaves, ensure proper plant spacing.",
            "prevention": "Use resistant varieties, improve air circulation, balanced fertilization",
            "severity": "Moderate",
            "color": "orange",
            "economic_impact": "20-40% yield reduction"
        },
        "cassava green mite": {
            "description": "Microscopic pest causing leaf yellowing and bronzing.",
            "symptoms": ["Leaf yellowing and bronzing", "Fine webbing on leaves", "Stunted growth"],
            "causes": "Mononychellus tanajoa (green spider mite)",
            "action": "Apply miticides, increase humidity around plants, release predatory mites.",
            "prevention": "Regular monitoring, maintain field moisture, encourage natural enemies",
            "severity": "Moderate to High",
            "color": "orange",
            "economic_impact": "30-80% yield loss in severe infestations"
        },
        "cassava healthy": {
            "description": "Cassava plant appears healthy with no visible disease symptoms.",
            "symptoms": ["Dark green, vigorous leaves", "Normal stem development", "Good root formation"],
            "causes": "No disease present",
            "action": "Continue regular monitoring and maintain good agricultural practices.",
            "prevention": "Use healthy planting material, proper spacing, integrated pest management",
            "severity": "None",
            "color": "green",
            "economic_impact": "Optimal tuber yield expected"
        },
        "cassava mosaic": {
            "description": "Viral disease causing mosaic patterns and leaf distortion.",
            "symptoms": ["Mosaic pattern on leaves", "Leaf distortion", "Stunted growth", "Reduced tuber size"],
            "causes": "Cassava mosaic virus (transmitted by whiteflies)",
            "action": "Remove infected plants immediately, control whitefly vectors, use resistant varieties.",
            "prevention": "Use certified virus-free planting material, control whiteflies, field sanitation",
            "severity": "Very High",
            "color": "red",
            "economic_impact": "70-100% yield loss in susceptible varieties"
        },
        
        # MAIZE DISEASES AND PESTS
        "maize fall armyworm": {
            "description": "Destructive pest caterpillar feeding on maize leaves and stems.",
            "symptoms": ["Irregular holes in leaves", "Window-pane feeding pattern", "Frass in leaf whorls"],
            "causes": "Spodoptera frugiperda larvae",
            "action": "Apply appropriate insecticides, use biological control agents, handpick larvae when possible.",
            "prevention": "Early planting, crop rotation, intercropping, pheromone traps",
            "severity": "High",
            "color": "red",
            "economic_impact": "20-70% yield loss depending on infestation level"
        },
        "maize grasshoper": {
            "description": "Grasshopper pest causing defoliation and stem damage.",
            "symptoms": ["Irregular holes in leaves", "Defoliation", "Stem cutting", "Reduced plant vigor"],
            "causes": "Various grasshopper species",
            "action": "Apply contact insecticides, use biopesticides, encourage natural predators.",
            "prevention": "Field sanitation, remove alternate hosts, biological control",
            "severity": "Moderate",
            "color": "orange",
            "economic_impact": "10-30% yield loss in severe infestations"
        },
        "maize healthy": {
            "description": "Maize plant appears healthy with no visible disease or pest damage.",
            "symptoms": ["Vibrant green leaves", "Normal growth rate", "No pest damage", "Good ear development"],
            "causes": "No disease or pest present",
            "action": "Continue regular monitoring and maintain good agricultural practices.",
            "prevention": "Balanced nutrition, proper spacing, integrated pest and disease management",
            "severity": "None",
            "color": "green",
            "economic_impact": "Maximum yield potential"
        },
        "maize leaf beetle": {
            "description": "Beetle pest causing characteristic hole patterns in maize leaves.",
            "symptoms": ["Small round holes in leaves", "Skeletonized leaves", "Reduced photosynthesis"],
            "causes": "Diabrotica species (leaf beetles)",
            "action": "Apply insecticides, use sticky traps, encourage beneficial insects.",
            "prevention": "Crop rotation, field sanitation, resistant varieties",
            "severity": "Moderate",
            "color": "orange",
            "economic_impact": "15-25% yield reduction"
        },
        "maize leaf blight": {
            "description": "Fungal disease causing elongated lesions on maize leaves.",
            "symptoms": ["Elliptical gray-green lesions", "Lesions with dark borders", "Premature leaf death"],
            "causes": "Exserohilum turcicum fungus",
            "action": "Apply fungicides, plant resistant varieties, remove crop residues.",
            "prevention": "Crop rotation, balanced fertilization, proper plant spacing",
            "severity": "Moderate to High",
            "color": "orange",
            "economic_impact": "30-70% yield loss in susceptible varieties"
        },
        "maize leaf spot": {
            "description": "Fungal disease causing circular to oval spots on leaves.",
            "symptoms": ["Circular brown spots with light centers", "Yellow halos around spots", "Premature senescence"],
            "causes": "Bipolaris maydis fungus",
            "action": "Apply fungicides, ensure good air circulation, remove infected debris.",
            "prevention": "Resistant varieties, crop rotation, balanced nutrition",
            "severity": "Moderate",
            "color": "orange",
            "economic_impact": "20-50% yield loss"
        },
        "maize streak virus": {
            "description": "Viral disease causing characteristic streaking on maize leaves.",
            "symptoms": ["Yellow-white streaks parallel to leaf veins", "Stunted growth", "Poor ear development"],
            "causes": "Maize streak virus (transmitted by leafhoppers)",
            "action": "Remove infected plants, control leafhopper vectors, use resistant varieties.",
            "prevention": "Plant resistant varieties, control vectors, avoid late planting",
            "severity": "High",
            "color": "red",
            "economic_impact": "50-100% yield loss in susceptible varieties"
        },
        
        # TOMATO DISEASES
        "tomato healthy": {
            "description": "Tomato plant appears healthy with no visible disease symptoms.",
            "symptoms": ["Deep green foliage", "Normal fruit development", "Vigorous growth"],
            "causes": "No disease present",
            "action": "Continue regular monitoring and maintain good agricultural practices.",
            "prevention": "Proper nutrition, adequate spacing, integrated disease management",
            "severity": "None",
            "color": "green",
            "economic_impact": "Optimal fruit yield and quality"
        },
        "tomato leaf blight": {
            "description": "Fungal disease causing brown spots and leaf death in tomatoes.",
            "symptoms": ["Brown spots with concentric rings", "Yellow halos", "Rapid leaf death"],
            "causes": "Alternaria solani fungus",
            "action": "Apply fungicides, remove infected leaves, improve air circulation.",
            "prevention": "Drip irrigation, mulching, crop rotation, resistant varieties",
            "severity": "High",
            "color": "red",
            "economic_impact": "40-80% yield loss if untreated"
        },
        "tomato leaf curl": {
            "description": "Viral disease causing upward curling and thickening of tomato leaves.",
            "symptoms": ["Upward leaf curling", "Leaf thickening", "Stunted growth", "Reduced fruit set"],
            "causes": "Tomato leaf curl virus (transmitted by whiteflies)",
            "action": "Remove infected plants, control whitefly vectors, use reflective mulch.",
            "prevention": "Use virus-free seedlings, control whiteflies, resistant varieties",
            "severity": "Very High",
            "color": "red",
            "economic_impact": "70-100% yield loss"
        },
        "tomato septoria leaf spot": {
            "description": "Fungal disease causing small circular spots with dark borders.",
            "symptoms": ["Small circular spots with dark borders", "Yellow halos", "Progressive defoliation"],
            "causes": "Septoria lycopersici fungus",
            "action": "Apply fungicides, remove lower leaves, ensure good air circulation.",
            "prevention": "Avoid overhead watering, mulch plants, crop rotation",
            "severity": "Moderate to High",
            "color": "orange",
            "economic_impact": "30-60% yield reduction"
        },
        "tomato verticulium wilt": {
            "description": "Soil-borne fungal disease causing wilting and yellowing.",
            "symptoms": ["Yellowing of lower leaves", "Wilting during day", "Brown vascular discoloration"],
            "causes": "Verticillium dahliae fungus",
            "action": "No cure available - remove infected plants, improve drainage, soil solarization.",
            "prevention": "Use resistant varieties, soil sterilization, crop rotation with non-hosts",
            "severity": "Very High",
            "color": "red",
            "economic_impact": "Complete plant loss in severe cases"
        }
    }
    
    return disease_info.get(disease_name, {
        "description": "Disease information not available in database.",
        "symptoms": ["Consult agricultural expert for identification"],
        "causes": "Unknown",
        "action": "Consult with agricultural experts for proper diagnosis and treatment.",
        "prevention": "Follow integrated pest and disease management practices",
        "severity": "Unknown",
        "color": "gray",
        "economic_impact": "Consult expert for assessment"
    })

def display_results(results, image):
    """Display prediction results with agricultural context"""
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📸 Analyzed Crop Image")
        st.image(image, caption="Uploaded Crop Image", use_container_width=True)
        
        # Image details
        st.info(f"**Image Size:** {image.size[0]} x {image.size[1]} pixels")
    
    with col2:
        st.subheader("🔍 Disease Analysis Results")
        
        disease_info = get_disease_info(results['predicted_class'])
        
        # Main prediction with color coding
        if disease_info['color'] == 'green':
            st.success(f"✅ **Diagnosis:** {results['predicted_class']}")
        elif disease_info['color'] == 'orange':
            st.warning(f"⚠️ **Diagnosis:** {results['predicted_class']}")
        else:
            st.error(f"🚨 **Diagnosis:** {results['predicted_class']}")
        
        st.metric("Confidence Level", f"{results['confidence']:.1%}")
        
        # Disease information with enhanced details
        st.subheader("📋 Detailed Disease Information")
        st.write(f"**Description:** {disease_info['description']}")
        
        # Show symptoms
        if 'symptoms' in disease_info:
            st.write("**Symptoms:**")
            for symptom in disease_info['symptoms']:
                st.write(f"• {symptom}")
        
        st.write(f"**Cause:** {disease_info['causes']}")
        st.write(f"**Severity:** {disease_info['severity']}")
        st.write(f"**Economic Impact:** {disease_info['economic_impact']}")
        
        # Action plan
        st.subheader("🎯 Recommended Action Plan")
        st.write(f"**Treatment:** {disease_info['action']}")
        st.write(f"**Prevention:** {disease_info['prevention']}")
        
        # Urgency indicator
        if disease_info['severity'] in ['Very High', 'High']:
            st.error("🚨 **URGENT ACTION REQUIRED** - Contact agricultural expert immediately!")
        elif disease_info['severity'] == 'Moderate':
            st.warning("⚠️ **Monitor closely** - Take preventive measures")
        elif disease_info['severity'] == 'Moderate to High':
            st.warning("⚠️ **Action needed** - Apply treatments promptly")
        
        # Confidence threshold warning
        if results['confidence'] < 0.75:
            st.warning("⚠️ Low confidence prediction. Consider consulting an agricultural expert.")

    # All predictions visualization
    st.subheader("📊 Disease Probability Distribution")
    
    # Create DataFrame for plotting
    df = pd.DataFrame(
        list(results['all_predictions'].items()),
        columns=['Disease/Condition', 'Probability']
    )
    df = df.sort_values('Probability', ascending=False)
    
    # Create horizontal bar chart
    fig = px.bar(
        df, y='Disease/Condition', x='Probability',
        title="Probability of Each Disease/Condition",
        orientation='h',
        color='Probability',
        color_continuous_scale='RdYlGn_r',  # Red to Green (reverse)
        text='Probability'
    )
    
    fig.update_traces(texttemplate='%{text:.1%}', textposition='outside')
    fig.update_layout(height=400, showlegend=False)
    fig.update_xaxes(tickformat=',.0%')  # Changed from update_xaxis to update_xaxes
    
    st.plotly_chart(fig, use_container_width=True)

def main():
    # App header
    st.title("🌾 Multi-Crop Disease Classifier")
    st.markdown("**AI-Powered Disease Detection for Cashew, Cassava, Maize & Tomato | EfficientNetB0 PyTorch**")
    st.markdown("---")
    
    # Load class names at the very beginning
    class_names = load_class_names()
    
    if class_names is None:
        st.error("❌ Failed to load class names. Please check your class_names.json file.")
        return
    
    # Device info
    device_info = "🖥️ CPU" if not torch.cuda.is_available() else f"🚀 GPU ({torch.cuda.get_device_name(0)})"
    st.sidebar.success(f"**Computing Device:** {device_info}")
    
    # Sidebar information
    with st.sidebar:
        st.header("🌾 Multi-Crop Disease Detector")
        st.markdown("""
        This advanced AI tool identifies diseases and pests across four major crops:
        
        **🥜 Cashew:** Anthracnose, Gumosis, Leaf Miner, Red Rust
        **🍠 Cassava:** Bacterial Blight, Brown Spot, Green Mite, Mosaic
        **🌽 Maize:** Fall Armyworm, Leaf Blight, Streak Virus, Pests
        **🍅 Tomato:** Leaf Blight, Leaf Curl, Septoria, Verticillium Wilt
        
        **Key Features:**
        - Multi-crop disease detection
        - Pest identification capabilities  
        - Economic impact assessment
        - Treatment recommendations
        """)
        
        st.header("📱 How to Use")
        st.markdown("""
        1. **Upload** a clear image of the crop leaf
        2. **Wait** for AI analysis (few seconds)
        3. **Review** the diagnosis and confidence
        4. **Follow** the recommended actions
        5. **Consult** experts for severe cases
        """)
        
        st.header("⚠️ Important Notes")
        st.markdown("""
        - Use clear, well-lit images
        - Focus on affected leaf areas
        - This is a diagnostic aid, not replacement for expert consultation
        - For severe diseases, contact agricultural experts immediately
        """)
        
        st.header("🔧 Technical Specs")
        st.markdown(f"""
        - **Model:** EfficientNetB0 (PyTorch)
        - **Classes:** {len(class_names)} conditions across 4 crops
        - **Input Size:** 224x224 pixels  
        - **Framework:** PyTorch + Torchvision
        - **Preprocessing:** ImageNet normalization
        """)
    
    # Load model 
    with st.spinner("🔄 Loading PyTorch AI model..."):
        model_data = load_model()
    
    if model_data[0] is None:
        st.error("❌ Failed to load the PyTorch model. Please check your model file.")
        st.info("""
        **Setup Instructions:**
        1. Save your trained PyTorch model as `model/crop_disease_model.pth`
        2. Create a `model/class_names.json` file with your disease classes
        3. Ensure the model architecture matches (EfficientNetB0)
        4. Restart the application
        """)
        return
    
    model, device = model_data
    
    st.success("✅ PyTorch AI model loaded successfully!")
    st.info(f"📊 Model can detect {len(class_names)} different conditions: {', '.join(class_names)}")
    
    # File uploader
    st.subheader("📁 Upload Crop Image for Analysis")
    
    uploaded_file = st.file_uploader(
        "Choose a crop/leaf image file",
        type=['png', 'jpg', 'jpeg'],
        help="Upload a clear image of crop leaves (PNG, JPG, JPEG format, max 5MB)"
    )
    
    # Sample images section
    with st.expander("📷 Tips for Best Results"):
        st.markdown("""
        **For accurate disease detection:**
        - Use good lighting (natural daylight preferred)
        - Focus on the affected leaf areas
        - Avoid blurry or low-resolution images
        - Include the entire leaf when possible
        - Take photos straight-on (not at extreme angles)
        """)
    
    if uploaded_file is not None:
        # Validate file size
        if uploaded_file.size > MAX_FILE_SIZE:
            st.error(f"❌ File size ({uploaded_file.size/1024/1024:.1f}MB) exceeds 5MB limit")
            return
        
        # Process and predict
        with st.spinner("🔍 Analyzing crop image for disease detection..."):
            processed_image, original_image = preprocess_image(uploaded_file)
            
            if processed_image is not None:
                results = make_prediction(model, processed_image, class_names, device)
                
                if results is not None:
                    display_results(results, original_image)
                    
                    # Export results option
                    st.subheader("📄 Export Analysis Report")
                    
                    report_data = {
                        "timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "filename": uploaded_file.name,
                        "diagnosis": results['predicted_class'],
                        "confidence": f"{results['confidence']:.2%}",
                        "severity": get_disease_info(results['predicted_class'])['severity'],
                        "recommended_action": get_disease_info(results['predicted_class'])['action']
                    }
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("📋 Generate Report"):
                            report_df = pd.DataFrame([report_data])
                            st.dataframe(report_df, use_container_width=True)
                    
                    with col2:
                        # Technical details
                        with st.expander("🔧 Technical Details"):
                            st.json({
                                "Model Architecture": "EfficientNetB0 (PyTorch)",
                                "Input Shape": list(processed_image.shape),
                                "Number of Classes": len(class_names),
                                "Prediction Confidence": f"{results['confidence']:.4f}",
                                "Processing Device": str(device).upper(),
                                "File Size": f"{uploaded_file.size / 1024:.2f} KB",
                                "Image Dimensions": f"{original_image.size[0]}x{original_image.size[1]}"
                            })

if __name__ == "__main__":
    main()