import os
import logging
from flask import Flask, render_template, request, jsonify
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F
from typing import Tuple, Dict, Any
import time

# -------------------------
# Configuration
# -------------------------
class Config:
    MODEL_PATH = "C:/Models/my_model"
    CONFIDENCE_THRESHOLD = 0.4
    MAX_LENGTH = 512
    DEBUG = True

# -------------------------
# Flask App
# -------------------------
app = Flask(__name__)
app.config.from_object(Config)

# -------------------------
# Logging Setup
# -------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -------------------------
# Model Manager (Singleton Pattern)
# -------------------------
class ModelManager:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelManager, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance
    
    def _initialize(self):
        """Initialize model and tokenizer"""
        try:
            logger.info("Loading model and tokenizer...")
            start_time = time.time()
            
            self.tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_PATH)
            self.model = AutoModelForSequenceClassification.from_pretrained(Config.MODEL_PATH)
            
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(self.device)
            self.model.eval()
            
            load_time = time.time() - start_time
            logger.info(f"Model loaded successfully in {load_time:.2f} seconds")
            logger.info(f"Using device: {self.device}")
            
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise
    
    def predict(self, text: str) -> Tuple[str, float, Dict[str, float]]:
        """
        Predict category with detailed probabilities
        
        Returns:
            Tuple of (predicted_label, confidence_score, all_probabilities)
        """
        try:
            # Tokenization
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=Config.MAX_LENGTH
            ).to(self.device)
            
            # Model inference
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            # Convert logits to probabilities
            logits = outputs.logits
            probs = F.softmax(logits, dim=1)
            
            # Get confidence and prediction
            confidence, pred_id = torch.max(probs, dim=1)
            confidence = confidence.item()
            pred_id = pred_id.item()
            
            # Get label
            label = self.model.config.id2label[pred_id]
            
            # Get all probabilities
            all_probs = {self.model.config.id2label[i]: probs[0][i].item() 
                        for i in range(len(self.model.config.id2label))}
            
            # Check confidence threshold
            if confidence < Config.CONFIDENCE_THRESHOLD:
                label = "UNCERTAIN"
            
            return label, confidence, all_probs
            
        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            raise

# Initialize model manager
model_manager = ModelManager()

# -------------------------
# API Routes
# -------------------------
@app.route("/", methods=["GET"])
def index():
    """Render main page"""
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    """API endpoint for predictions"""
    try:
        data = request.get_json()
        text = data.get("text", "").strip()
        
        if not text:
            return jsonify({
                "error": "No text provided",
                "success": False
            }), 400
        
        if len(text) > 2000:
            return jsonify({
                "error": "Text too long. Maximum 2000 characters.",
                "success": False
            }), 400
        
        # Perform prediction
        label, confidence, all_probs = model_manager.predict(text)
        
        response = {
            "success": True,
            "prediction": label,
            "confidence": confidence,
            "probabilities": all_probs,
            "is_confident": confidence >= Config.CONFIDENCE_THRESHOLD,
            "timestamp": time.time()
        }
        
        logger.info(f"Prediction: {label} (confidence: {confidence:.2f})")
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"API error: {str(e)}")
        return jsonify({
            "error": "Internal server error",
            "success": False
        }), 500

@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "model_loaded": True,
        "device": str(model_manager.device)
    })

@app.route("/api/classify", methods=["POST"])
def classify():
    """Alternative endpoint for form submissions"""
    try:
        text = request.form.get("text", "").strip()
        
        if not text:
            return render_template(
                "index.html",
                error="Please enter some text",
                text=text
            )
        
        label, confidence, all_probs = model_manager.predict(text)
        
        return render_template(
            "index.html",
            prediction=label,
            confidence=confidence,
            probabilities=all_probs,
            is_confident=confidence >= Config.CONFIDENCE_THRESHOLD,
            text=text
        )
        
    except Exception as e:
        logger.error(f"Classification error: {str(e)}")
        return render_template(
            "index.html",
            error="An error occurred during classification",
            text=request.form.get("text", "")
        )

# -------------------------
# Error Handlers
# -------------------------
@app.errorhandler(404)
def not_found(error):
    return render_template('error.html', error="Page not found"), 404

@app.errorhandler(500)
def internal_error(error):
    return render_template('error.html', error="Internal server error"), 500

# -------------------------
# Run Application
# -------------------------
if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=Config.DEBUG,
        threaded=True
    )