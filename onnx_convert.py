#!/usr/bin/env python3
"""
Script to convert quantized Kosmos-2.5 model to ONNX for browser deployment.

This script takes a quantized Kosmos-2.5 model and converts it to ONNX format,
optimizes it for browser deployment, and generates web integration files.

Requirements:
- torch
- transformers
- optimum[onnxruntime]
- onnx
- onnxruntime
- numpy
- Pillow

Install with:
pip install torch transformers optimum[onnxruntime] onnx onnxruntime numpy Pillow
"""

import os
import torch
import numpy as np
import json
from transformers import AutoProcessor, AutoModelForVision2Seq
from pathlib import Path
import logging
import onnx
from PIL import Image

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class KosmosONNXConverter:
    def __init__(self, quantized_model_dir="./kosmos_quantized", output_dir="./kosmos_onnx"):
        self.quantized_model_dir = Path(quantized_model_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Set device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")
        
        # Model configuration
        self.model_config = None
        self.quantization_stats = None
        
    def load_quantized_model(self):
        """Load the quantized model and its configuration."""
        logger.info(f"Loading quantized model from: {self.quantized_model_dir}")
        
        try:
            # Load model configuration
            config_path = self.quantized_model_dir / "model_config.json"
            if not config_path.exists():
                raise FileNotFoundError(f"Model configuration not found: {config_path}")
            
            with open(config_path, 'r') as f:
                self.model_config = json.load(f)
            
            # Load quantization statistics
            stats_path = self.quantized_model_dir / "quantization_stats.json"
            if stats_path.exists():
                with open(stats_path, 'r') as f:
                    self.quantization_stats = json.load(f)
            
            # Load processor
            processor_path = Path(self.model_config['processor_path'])
            self.processor = AutoProcessor.from_pretrained(processor_path)
            
            # Load quantized model
            model_path = Path(self.model_config['model_path'])
            self.model = AutoModelForVision2Seq.from_pretrained(
                model_path,
                torch_dtype=torch.float16,  # Quantized model is in FP16
                trust_remote_code=True
            )
            
            # Move to device
            self.model = self.model.to(self.device)
            self.model.eval()
            
            logger.info("Quantized model loaded successfully")
            logger.info(f"Original model: {self.model_config['model_name']}")
            logger.info(f"Quantization method: {self.model_config['quantization_method']}")
            
            if self.quantization_stats:
                logger.info(f"Model size: {self.quantization_stats.get('size_after_mb', 0):.1f}MB")
                logger.info(f"Compression ratio: {self.quantization_stats.get('compression_ratio', 0):.2f}x")
            
    
    def _generate_html_demo(self, model_filename):
        """Generate HTML demo page for browser testing."""
        return f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Kosmos-2.5 FP8 Quantized - Browser Demo</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            min-height: 100vh;
        }}
        .container {{
            background: rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(10px);
            border-radius: 20px;
            padding: 30px;
            box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
        }}
        .header {{
            text-align: center;
            margin-bottom: 30px;
        }}
        .model-info {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        .info-card {{
            background: rgba(255, 255, 255, 0.2);
            padding: 20px;
            border-radius: 15px;
            text-align: center;
        }}
        .demo-section {{
            background: rgba(255, 255, 255, 0.1);
            padding: 25px;
            border-radius: 15px;
            margin-bottom: 20px;
        }}
        .upload-area {{
            border: 2px dashed rgba(255, 255, 255, 0.5);
            border-radius: 10px;
            padding: 40px;
            text-align: center;
            margin: 20px 0;
            transition: all 0.3s ease;
        }}
        .upload-area:hover {{
            border-color: rgba(255, 255, 255, 0.8);
            background: rgba(255, 255, 255, 0.1);
        }}
        .btn {{
            background: linear-gradient(45deg, #ff6b6b, #ee5a24);
            color: white;
            border: none;
            padding: 12px 25px;
            border-radius: 25px;
            cursor: pointer;
            font-size: 16px;
            font-weight: bold;
            transition: all 0.3s ease;
            margin: 5px;
        }}
        .btn:hover {{
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(0,0,0,0.3);
        }}
        .btn:disabled {{
            background: #666;
            cursor: not-allowed;
            transform: none;
        }}
        .results {{
            background: rgba(0, 0, 0, 0.3);
            padding: 20px;
            border-radius: 10px;
            margin-top: 20px;
            font-family: 'Courier New', monospace;
        }}
        .loading {{
            display: none;
            text-align: center;
            padding: 20px;
        }}
        .spinner {{
            border: 4px solid rgba(255,255,255,0.3);
            border-radius: 50%;
            border-top: 4px solid white;
            width: 40px;
            height: 40px;
            animation: spin 2s linear infinite;
            margin: 0 auto 10px;
        }}
        @keyframes spin {{
            0% {{ transform: rotate(0deg); }}
            100% {{ transform: rotate(360deg); }}
        }}
        canvas {{
            border: 1px solid rgba(255,255,255,0.3);
            border-radius: 5px;
            margin: 10px 0;
        }}
    </style>
    <script src="https://cdn.jsdelivr.net/npm/onnxruntime-web@1.16.0/dist/ort.min.js"></script>
    <script src="kosmos_web_config.js"></script>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🤖 Kosmos-2.5 FP8 Quantized</h1>
            <p>Vision-Language Model Running in Browser</p>
        </div>
        
        <div class="model-info">
            <div class="info-card">
                <h3>📊 Model Size</h3>
                <p id="model-size">Loading...</p>
            </div>
            <div class="info-card">
                <h3>⚡ Quantization</h3>
                <p id="quantization">Loading...</p>
            </div>
            <div class="info-card">
                <h3>🗜️ Compression</h3>
                <p id="compression">Loading...</p>
            </div>
            <div class="info-card">
                <h3>🎯 Status</h3>
                <p id="model-status">Loading model...</p>
            </div>
        </div>
        
        <div class="demo-section">
            <h2>🖼️ Image Analysis Demo</h2>
            
            <div class="upload-area" id="uploadArea">
                <p>📁 Click to select an image or drag & drop</p>
                <input type="file" id="imageInput" accept="image/*" style="display: none;">
                <br>
                <button class="btn" onclick="document.getElementById('imageInput').click()">
                    Select Image
                </button>
            </div>
            
            <div>
                <label for="promptInput">💬 Text Prompt:</label><br>
                <input type="text" id="promptInput" 
                       value="&lt;grounding&gt;Describe this image in detail."
                       style="width: 100%; padding: 10px; margin: 10px 0; border-radius: 5px; border: 1px solid #ccc;">
            </div>
            
            <div style="text-align: center; margin: 20px 0;">
                <button class="btn" id="analyzeBtn" onclick="runAnalysis()" disabled>
                    🔍 Analyze Image
                </button>
                <button class="btn" onclick="clearResults()">
                    🗑️ Clear
                </button>
            </div>
            
            <div id="imagePreview" style="text-align: center;"></div>
            
            <div class="loading" id="loading">
                <div class="spinner"></div>
                <p>Processing image with Kosmos-2.5...</p>
            </div>
            
            <div id="results" class="results" style="display: none;">
                <h3>📝 Analysis Results:</h3>
                <div id="output"></div>
            </div>
        </div>
        
        <div class="demo-section">
            <h2>🛠️ Technical Details</h2>
            <ul>
                <li><strong>Model:</strong> microsoft/kosmos-2.5 (FP8 Quantized)</li>
                <li><strong>Framework:</strong> ONNX Runtime Web</li>
                <li><strong>Quantization:</strong> Simulated FP8 precision</li>
                <li><strong>Browser:</strong> WebAssembly execution</li>
                <li><strong>Input:</strong> 224x224 RGB images + text prompts</li>
            </ul>
        </div>
    </div>

    <script>
        let session = null;
        let modelLoaded = false;

        // Initialize model on page load
        window.addEventListener('load', async () => {{
            await initializeModel();
            setupEventListeners();
            updateModelInfo();
        }});

        async function initializeModel() {{
            try {{
                document.getElementById('model-status').textContent = 'Loading...';
                
                // Load ONNX model
                session = await ort.InferenceSession.create('{model_filename}');
                modelLoaded = true;
                
                document.getElementById('model-status').textContent = '✅ Ready';
                document.getElementById('analyzeBtn').disabled = false;
                
                console.log('✅ Kosmos-2.5 model loaded successfully');
            }} catch (error) {{
                console.error('❌ Error loading model:', error);
                document.getElementById('model-status').textContent = '❌ Failed to load';
            }}
        }}

        function setupEventListeners() {{
            const imageInput = document.getElementById('imageInput');
            const uploadArea = document.getElementById('uploadArea');

            imageInput.addEventListener('change', handleImageUpload);
            
            uploadArea.addEventListener('dragover', (e) => {{
                e.preventDefault();
                uploadArea.style.borderColor = 'rgba(255, 255, 255, 0.8)';
            }});
            
            uploadArea.addEventListener('dragleave', (e) => {{
                e.preventDefault();
                uploadArea.style.borderColor = 'rgba(255, 255, 255, 0.5)';
            }});
            
            uploadArea.addEventListener('drop', (e) => {{
                e.preventDefault();
                uploadArea.style.borderColor = 'rgba(255, 255, 255, 0.5)';
                const files = e.dataTransfer.files;
                if (files.length > 0) {{
                    handleImageFile(files[0]);
                }}
            }});
        }}

        function updateModelInfo() {{
            if (typeof KOSMOS_CONFIG !== 'undefined') {{
                document.getElementById('model-size').textContent = 
                    KOSMOS_CONFIG.modelSizeMB.toFixed(1) + ' MB';
                document.getElementById('quantization').textContent = 
                    KOSMOS_CONFIG.quantization;
                document.getElementById('compression').textContent = 
                    KOSMOS_CONFIG.compressionRatio.toFixed(1) + 'x';
            }}
        }}

        function handleImageUpload(event) {{
            const file = event.target.files[0];
            if (file) {{
                handleImageFile(file);
            }}
        }}

        function handleImageFile(file) {{
            if (!file.type.startsWith('image/')) {{
                alert('Please select an image file');
                return;
            }}

            const reader = new FileReader();
            reader.onload = (e) => {{
                const img = new Image();
                img.onload = () => {{
                    displayImagePreview(img);
                }};
                img.src = e.target.result;
            }};
            reader.readAsDataURL(file);
        }}

        function displayImagePreview(img) {{
            const preview = document.getElementById('imagePreview');
            preview.innerHTML = '';
            
            const canvas = document.createElement('canvas');
            const ctx = canvas.getContext('2d');
            
            // Set display size
            const maxSize = 300;
            const scale = Math.min(maxSize / img.width, maxSize / img.height);
            canvas.style.width = (img.width * scale) + 'px';
            canvas.style.height = (img.height * scale) + 'px';
            
            // Set actual canvas size for processing
            canvas.width = img.width;
            canvas.height = img.height;
            
            ctx.drawImage(img, 0, 0);
            preview.appendChild(canvas);
            
            // Store canvas for processing
            window.currentImage = canvas;
        }}

        async function runAnalysis() {{
            if (!modelLoaded || !window.currentImage) {{
                alert('Please load an image and wait for the model to initialize');
                return;
            }}

            const loading = document.getElementById('loading');
            const results = document.getElementById('results');
            const analyzeBtn = document.getElementById('analyzeBtn');

            try {{
                // Show loading state
                loading.style.display = 'block';
                results.style.display = 'none';
                analyzeBtn.disabled = true;

                // Preprocess image
                const processedImage = KOSMOS_CONFIG.preprocessImage(
                    window.currentImage, 
                    document.createElement('canvas')
                );

                // Preprocess text
                const promptText = document.getElementById('promptInput').value;
                const processedText = KOSMOS_CONFIG.preprocessText(promptText);

                // Create input tensors
                const inputs = {{
                    'pixel_values': new ort.Tensor('float32', processedImage, [1, 3, 224, 224]),
                    'input_ids': new ort.Tensor('int64', processedText.map(x => BigInt(x)), [1, processedText.length])
                }};

                // Run inference
                const outputs = await session.run(inputs);

                // Process results
                const logits = outputs.logits.data;
                const tokens = KOSMOS_CONFIG.postprocessOutput([Array.from(logits)]);

                // Display results
                document.getElementById('output').innerHTML = `
                    <p><strong>Prompt:</strong> ${{promptText}}</p>
                    <p><strong>Generated Tokens:</strong> ${{tokens.slice(0, 20).join(', ')}}...</p>
                    <p><strong>Output Shape:</strong> ${{outputs.logits.dims.join(' × ')}}</p>
                    <p><em>Note: This is a simplified demo. Full text decoding requires the tokenizer.</em></p>
                `;

                results.style.display = 'block';

            }} catch (error) {{
                console.error('Analysis error:', error);
                document.getElementById('output').innerHTML = 
                    `<p style="color: #ff6b6b;"><strong>Error:</strong> ${{error.message}}</p>`;
                results.style.display = 'block';
            }} finally {{
                loading.style.display = 'none';
                analyzeBtn.disabled = false;
            }}
        }}

        function clearResults() {{
            document.getElementById('results').style.display = 'none';
            document.getElementById('imagePreview').innerHTML = '';
            document.getElementById('imageInput').value = '';
            window.currentImage = null;
        }}
    </script>
</body>
</html>'''
            
        except Exception as e:
            logger.error(f"Error loading quantized model: {e}")
            return False
    
    def create_dummy_inputs(self, batch_size=1, image_size=(224, 224), sequence_length=77):
        """Create dummy inputs for ONNX export with proper shapes."""
        logger.info("Creating dummy inputs for ONNX export...")
        
        try:
            # Create dummy image
            dummy_image = Image.new('RGB', image_size, color=(128, 128, 128))
            
            # Create dummy text prompt
            dummy_text = "<grounding>Describe this image in detail."
            
            # Process inputs
            inputs = self.processor(
                text=dummy_text,
                images=dummy_image,
                return_tensors="pt"
            )
            
            # Ensure consistent batch size
            for key in inputs:
                if inputs[key].dim() > 0:
                    current_batch = inputs[key].shape[0]
                    if current_batch != batch_size:
                        inputs[key] = inputs[key].repeat(batch_size, *[1] * (inputs[key].dim() - 1))
            
            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            logger.info("Dummy inputs created successfully")
            for key, value in inputs.items():
                logger.info(f"  {key}: {value.shape} ({value.dtype})")
            
            return inputs
            
        except Exception as e:
            logger.error(f"Error creating dummy inputs: {e}")
            return None
    
    def test_model_inference(self, inputs):
        """Test the model with dummy inputs before ONNX conversion."""
        logger.info("Testing model inference...")
        
        try:
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            logger.info("Model inference test passed")
            
            if hasattr(outputs, 'logits'):
                logger.info(f"Output logits shape: {outputs.logits.shape}")
            if hasattr(outputs, 'last_hidden_state'):
                logger.info(f"Last hidden state shape: {outputs.last_hidden_state.shape}")
                
            return outputs
            
        except Exception as e:
            logger.error(f"Model inference test failed: {e}")
            return None
    
    def convert_to_onnx(self, opset_version=14):
        """Convert the quantized model to ONNX format."""
        logger.info(f"Converting model to ONNX (opset version {opset_version})...")
        
        try:
            # Create dummy inputs
            dummy_inputs = self.create_dummy_inputs()
            if dummy_inputs is None:
                return None
            
            # Test model inference first
            test_outputs = self.test_model_inference(dummy_inputs)
            if test_outputs is None:
                return None
            
            # Prepare ONNX export
            onnx_path = self.output_dir / "kosmos_quantized.onnx"
            
            # Extract input tensors for export
            input_names = list(dummy_inputs.keys())
            input_tensors = [dummy_inputs[name] for name in input_names]
            
            # Define dynamic axes
            dynamic_axes = {}
            for name in input_names:
                if 'pixel_values' in name:
                    dynamic_axes[name] = {0: 'batch_size'}
                elif 'input_ids' in name or 'attention_mask' in name:
                    dynamic_axes[name] = {0: 'batch_size', 1: 'sequence_length'}
            
            # Add output dynamic axes
            output_names = ['logits']
            dynamic_axes['logits'] = {0: 'batch_size', 1: 'sequence_length'}
            
            logger.info(f"Input names: {input_names}")
            logger.info(f"Output names: {output_names}")
            logger.info(f"Dynamic axes: {dynamic_axes}")
            
            # Export to ONNX
            torch.onnx.export(
                self.model,
                tuple(input_tensors),
                str(onnx_path),
                input_names=input_names,
                output_names=output_names,
                dynamic_axes=dynamic_axes,
                opset_version=opset_version,
                do_constant_folding=True,
                verbose=False,
                export_params=True
            )
            
            logger.info(f"ONNX model exported to: {onnx_path}")
            
            # Verify ONNX model
            self.verify_onnx_model(onnx_path)
            
            return str(onnx_path)
            
        except Exception as e:
            logger.error(f"Error during ONNX conversion: {e}")
            return None
    
    def verify_onnx_model(self, onnx_path):
        """Verify the ONNX model structure and run basic checks."""
        logger.info("Verifying ONNX model...")
        
        try:
            # Load and check ONNX model
            onnx_model = onnx.load(str(onnx_path))
            onnx.checker.check_model(onnx_model)
            
            # Get model info
            model_size = os.path.getsize(onnx_path) / (1024 * 1024)  # MB
            
            logger.info("ONNX model verification passed")
            logger.info(f"ONNX model size: {model_size:.1f}MB")
            logger.info(f"ONNX opset version: {onnx_model.opset_import[0].version}")
            
            # Print model inputs and outputs
            logger.info("Model inputs:")
            for inp in onnx_model.graph.input:
                logger.info(f"  {inp.name}: {[d.dim_value for d in inp.type.tensor_type.shape.dim]}")
            
            logger.info("Model outputs:")
            for out in onnx_model.graph.output:
                logger.info(f"  {out.name}: {[d.dim_value for d in out.type.tensor_type.shape.dim]}")
            
            return True
            
        except Exception as e:
            logger.error(f"ONNX model verification failed: {e}")
            return False
    
    def optimize_for_browser(self, onnx_path):
        """Optimize ONNX model for browser deployment."""
        logger.info("Optimizing ONNX model for browser deployment...")
        
        try:
            import onnxruntime as ort
            
            # Create optimized model path
            optimized_path = self.output_dir / "kosmos_optimized.onnx"
            
            # Set up optimization options
            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            sess_options.optimized_model_filepath = str(optimized_path)
            
            # Enable additional optimizations for web
            sess_options.enable_cpu_mem_arena = False
            sess_options.enable_mem_pattern = False
            
            # Create session to trigger optimization
            providers = ['CPUExecutionProvider']  # Use CPU for web compatibility
            session = ort.InferenceSession(onnx_path, sess_options, providers=providers)
            
            # Verify optimized model exists
            if optimized_path.exists():
                original_size = os.path.getsize(onnx_path) / (1024 * 1024)
                optimized_size = os.path.getsize(optimized_path) / (1024 * 1024)
                
                logger.info(f"Optimization completed")
                logger.info(f"Original size: {original_size:.1f}MB")
                logger.info(f"Optimized size: {optimized_size:.1f}MB")
                logger.info(f"Size reduction: {((original_size - optimized_size) / original_size) * 100:.1f}%")
                
                return str(optimized_path)
            else:
                logger.warning("Optimized model not created, using original")
                return onnx_path
                
        except Exception as e:
            logger.error(f"Error during browser optimization: {e}")
            logger.info("Continuing with unoptimized model...")
            return onnx_path
    
    def generate_web_integration_files(self, onnx_model_path):
        """Generate files needed for web browser integration."""
        logger.info("Generating web integration files...")
        
        try:
            model_filename = os.path.basename(onnx_model_path)
            
            # Generate model metadata
            model_metadata = {
                "model_name": "kosmos-2.5-fp8-quantized",
                "model_file": model_filename,
                "input_size": [224, 224, 3],
                "quantization": self.model_config.get('quantization_method', 'FP8_simulated'),
                "compression_ratio": self.quantization_stats.get('compression_ratio', 1.0) if self.quantization_stats else 1.0,
                "model_size_mb": os.path.getsize(onnx_model_path) / (1024 * 1024),
                "preprocessing": {
                    "mean": [0.485, 0.456, 0.406],
                    "std": [0.229, 0.224, 0.225],
                    "image_size": 224
                }
            }
            
            # Save metadata
            metadata_path = self.output_dir / "model_metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(model_metadata, f, indent=2)
            
            # Generate JavaScript configuration
            js_config = self._generate_js_config(model_metadata)
            js_config_path = self.output_dir / "kosmos_web_config.js"
            with open(js_config_path, 'w') as f:
                f.write(js_config)
            
            # Generate HTML demo
            html_demo = self._generate_html_demo(model_filename)
            html_demo_path = self.output_dir / "index.html"
            with open(html_demo_path, 'w') as f:
                f.write(html_demo)
            
            # Generate README
            readme = self._generate_readme(model_metadata)
            readme_path = self.output_dir / "README.md"
            with open(readme_path, 'w') as f:
                f.write(readme)
            
            logger.info("Web integration files generated:")
            logger.info(f"  Metadata: {metadata_path}")
            logger.info(f"  JS Config: {js_config_path}")
            logger.info(f"  HTML Demo: {html_demo_path}")
            logger.info(f"  README: {readme_path}")
            
    
    def _generate_readme(self, metadata):
        """Generate README file with deployment instructions."""
        return f'''# Kosmos-2.5 FP8 Quantized - Browser Deployment

This directory contains a quantized version of Microsoft's Kosmos-2.5 vision-language model optimized for browser deployment.

## 📋 Model Information

- **Original Model**: microsoft/kosmos-2.5
- **Quantization**: {metadata["quantization"]}
- **Model Size**: {metadata["model_size_mb"]:.1f} MB
- **Compression Ratio**: {metadata["compression_ratio"]:.2f}x
- **Format**: ONNX (optimized for web)

## 🚀 Quick Start

1. **Serve the files from a web server** (required for ONNX Runtime Web):
   ```bash
   # Using Python
   python -m http.server 8000
   
   # Using Node.js
   npx serve .
   
   # Using any web server of your choice
   ```

2. **Open your browser** and navigate to:
   ```
   http://localhost:8000
   ```

3. **Upload an image** and enter a text prompt to analyze the image.

## 📁 Files Description

- `{metadata["model_file"]}` - Quantized ONNX model
- `model_metadata.json` - Model configuration and metadata
- `kosmos_web_config.js` - JavaScript configuration and utilities
- `index.html` - Interactive demo webpage
- `README.md` - This file

## 🔧 Integration Guide

### Basic Usage

```javascript
// Load the model
const session = await ort.InferenceSession.create('{metadata["model_file"]}');

// Preprocess your image
const imageData = KOSMOS_CONFIG.preprocessImage(yourImage, canvas);
const textData = KOSMOS_CONFIG.preprocessText("Describe this image");

// Create input tensors
const inputs = {{
    'pixel_values': new ort.Tensor('float32', imageData, [1, 3, 224, 224]),
    'input_ids': new ort.Tensor('int64', textData.map(x => BigInt(x)), [1, textData.length])
}};

// Run inference
const outputs = await session.run(inputs);
```

### Requirements

- Modern web browser with WebAssembly support
- HTTPS or localhost (required for some browser security features)
- Sufficient memory (model size: {metadata["model_size_mb"]:.1f} MB)

## ⚠️ Limitations

- **Quantization**: This uses simulated FP8 quantization, not hardware FP8
- **Tokenizer**: Simplified text processing (full tokenizer integration needed for production)
- **Performance**: Browser inference is slower than native GPU inference
- **Memory**: Large model requires substantial browser memory

## 🛠️ Technical Details

### Input Format
- **Images**: 224×224 RGB, normalized with ImageNet statistics
- **Text**: Tokenized text prompts (simplified tokenization in demo)

### Output Format
- **Logits**: Raw model outputs requiring post-processing for text generation

### Browser Compatibility
- Chrome/Chromium: ✅ Full support
- Firefox: ✅ Full support  
- Safari: ✅ WebAssembly support
- Edge: ✅ Full support

## 🎯 Performance Tips

1. **Use smaller images** when possible to reduce preprocessing time
2. **Cache the model** - it only needs to be loaded once
3. **Optimize batch size** for your use case
4. **Consider Web Workers** for heavy preprocessing

## 🔍 Troubleshooting

### Model fails to load
- Ensure you're serving files from a web server (not file://)
- Check browser console for specific error messages
- Verify all files are in the same directory

### Out of memory errors
- Close other browser tabs
- Try on a device with more RAM
- Consider using a smaller model variant

### Slow performance
- This is expected - browser inference is slower than native
- Consider preprocessing images on the server side
- Use Web Workers for non-blocking execution

## 📚 Additional Resources

- [ONNX Runtime Web Documentation](https://onnxruntime.ai/docs/get-started/with-javascript.html)
- [Original Kosmos-2.5 Paper](https://arxiv.org/abs/2309.11419)
- [Hugging Face Model Page](https://huggingface.co/microsoft/kosmos-2.5)

## 📄 License

This quantized model follows the same license as the original microsoft/kosmos-2.5 model.

---

*Generated by Kosmos ONNX Converter - FP8 Quantization Pipeline*
'''
    
    def run_conversion_pipeline(self):
        """Run the complete ONNX conversion pipeline."""
        logger.info("Starting Kosmos-2.5 ONNX conversion pipeline...")
        
        # Step 1: Load quantized model
        if not self.load_quantized_model():
            return False
        
        # Step 2: Convert to ONNX
        onnx_path = self.convert_to_onnx()
        if not onnx_path:
            return False
        
        # Step 3: Optimize for browser
        optimized_path = self.optimize_for_browser(onnx_path)
        
        # Step 4: Generate web integration files
        if not self.generate_web_integration_files(optimized_path):
            logger.warning("Web integration files generation failed, but ONNX conversion succeeded")
        
        logger.info("="*60)
        logger.info("ONNX CONVERSION PIPELINE COMPLETED SUCCESSFULLY!")
        logger.info("="*60)
        logger.info(f"Input directory: {self.quantized_model_dir}")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"ONNX model: {optimized_path}")
        logger.info(f"Web demo: {self.output_dir}/index.html")
        logger.info("="*60)
        logger.info("🌐 To test the model:")
        logger.info(f"   cd {self.output_dir}")
        logger.info("   python -m http.server 8000")
        logger.info("   Open http://localhost:8000 in your browser")
        logger.info("="*60)
        
        return True


def main():
    """Main execution function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Convert quantized Kosmos-2.5 to ONNX')
    parser.add_argument('--input', default='./kosmos_quantized',
                       help='Directory containing quantized model')
    parser.add_argument('--output', default='./kosmos_onnx',
                       help='Output directory for ONNX model and web files')
    parser.add_argument('--opset', type=int, default=14,
                       help='ONNX opset version (default: 14)')
    
    args = parser.parse_args()
    
    try:
        # Initialize converter
        converter = KosmosONNXConverter(
            quantized_model_dir=args.input,
            output_dir=args.output
        )
        
        # Run conversion pipeline
        success = converter.run_conversion_pipeline()
        
        if success:
            print("\n✅ ONNX conversion completed successfully!")
            print(f"📁 Check the output directory: {converter.output_dir}")
            print("🌐 Serve the files with a web server and open index.html")
            print("\n🚀 Quick start:")
            print(f"   cd {converter.output_dir}")
            print("   python -m http.server 8000")
            print("   Open http://localhost:8000 in your browser")
        else:
            print("\n❌ ONNX conversion failed. Check the logs above for details.")
            
    except Exception as e:
        logger.error(f"Unexpected error in main: {e}")
        print(f"\n❌ Unexpected error: {e}")


if __name__ == "__main__":
    main()
    
    def _generate_js_config(self, metadata):
        """Generate JavaScript configuration for web integration."""
        return f'''// Kosmos-2.5 FP8 Quantized Model Configuration
const KOSMOS_CONFIG = {{
    // Model information
    modelFile: '{metadata["model_file"]}',
    modelName: '{metadata["model_name"]}',
    modelSizeMB: {metadata["model_size_mb"]:.1f},
    quantization: '{metadata["quantization"]}',
    compressionRatio: {metadata["compression_ratio"]:.2f},
    
    // Preprocessing parameters
    preprocessing: {{
        mean: {metadata["preprocessing"]["mean"]},
        std: {metadata["preprocessing"]["std"]},
        imageSize: {metadata["preprocessing"]["image_size"]}
    }},
    
    // Input/Output configuration
    inputNames: ['pixel_values', 'input_ids'],
    outputNames: ['logits'],
    
    // Utility functions
    preprocessImage: function(imageData, canvas) {{
        const ctx = canvas.getContext('2d');
        const imageSize = this.preprocessing.imageSize;
        
        // Resize canvas to model input size
        canvas.width = imageSize;
        canvas.height = imageSize;
        
        // Draw and resize image
        ctx.drawImage(imageData, 0, 0, imageSize, imageSize);
        
        // Get pixel data
        const imageDataArray = ctx.getImageData(0, 0, imageSize, imageSize);
        const pixels = imageDataArray.data;
        
        // Convert to normalized float array [C, H, W]
        const normalized = new Float32Array(3 * imageSize * imageSize);
        const mean = this.preprocessing.mean;
        const std = this.preprocessing.std;
        
        for (let i = 0; i < imageSize * imageSize; i++) {{
            const pixelIndex = i * 4; // RGBA
            const outputIndex = i;
            
            // Normalize RGB channels
            normalized[outputIndex] = (pixels[pixelIndex] / 255.0 - mean[0]) / std[0]; // R
            normalized[imageSize * imageSize + outputIndex] = (pixels[pixelIndex + 1] / 255.0 - mean[1]) / std[1]; // G
            normalized[2 * imageSize * imageSize + outputIndex] = (pixels[pixelIndex + 2] / 255.0 - mean[2]) / std[2]; // B
        }}
        
        return normalized;
    }},
    
    preprocessText: function(text) {{
        // Simple tokenization - in production, use the actual tokenizer
        const tokens = text.split(' ').map(word => word.charCodeAt(0) % 1000);
        const maxLength = 77;
        
        // Pad or truncate to max length
        const paddedTokens = new Array(maxLength).fill(0);
        for (let i = 0; i < Math.min(tokens.length, maxLength); i++) {{
            paddedTokens[i] = tokens[i];
        }}
        
        return new Int32Array(paddedTokens);
    }},
    
    postprocessOutput: function(logits) {{
        // Convert logits to text tokens (simplified)
        const tokens = [];
        for (let i = 0; i < logits.length; i++) {{
            const maxIndex = logits[i].indexOf(Math.max(...logits[i]));
            tokens.push(maxIndex);
        }}
        return tokens;
    }}
}};

// Export for use in different environments
if (typeof module !== 'undefined' && module.exports) {{
    module.exports = KOSMOS_CONFIG;
}} else if (typeof window !== 'undefined') {{
    window.KOSMOS_CONFIG = KOSMOS_CONFIG;
}}
'''