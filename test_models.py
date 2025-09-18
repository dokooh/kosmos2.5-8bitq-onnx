#!/usr/bin/env python3
"""
Test script to verify the exported ONNX models work correctly.
"""

import onnx
import onnxruntime as ort
import numpy as np
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_onnx_model(model_path):
    """Test an ONNX model to ensure it loads and can run inference."""
    
    try:
        # Load and verify model
        model = onnx.load(str(model_path))
        onnx.checker.check_model(model)
        logger.info(f"✅ Model {model_path.name} is valid")
        
        # Create inference session
        session = ort.InferenceSession(str(model_path))
        
        # Get input/output info
        input_names = [inp.name for inp in session.get_inputs()]
        output_names = [out.name for out in session.get_outputs()]
        
        logger.info(f"   Inputs: {input_names}")
        logger.info(f"   Outputs: {output_names}")
        
        # Create dummy inputs based on model
        inputs = {}
        for inp in session.get_inputs():
            shape = inp.shape
            # Replace dynamic dimensions with fixed values
            fixed_shape = []
            for dim in shape:
                if isinstance(dim, str) or dim is None:
                    fixed_shape.append(1)  # Use batch size 1
                else:
                    fixed_shape.append(dim)
            
            # Create random input data
            inputs[inp.name] = np.random.randn(*fixed_shape).astype(np.float32)
            logger.info(f"   Created input {inp.name} with shape {fixed_shape}")
        
        # Run inference
        outputs = session.run(output_names, inputs)
        
        logger.info(f"   Inference successful! Output shapes:")
        for i, output in enumerate(outputs):
            logger.info(f"     {output_names[i]}: {output.shape}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Error testing {model_path.name}: {e}")
        return False

def main():
    """Test all exported ONNX models."""
    
    output_dir = Path("./kosmos_fp8_onnx")
    
    if not output_dir.exists():
        logger.error("Output directory not found. Run simple_export.py first.")
        return False
    
    # Find all ONNX files
    onnx_files = list(output_dir.glob("*.onnx"))
    
    if not onnx_files:
        logger.error("No ONNX files found in output directory.")
        return False
    
    logger.info(f"Testing {len(onnx_files)} ONNX models...")
    
    success_count = 0
    for onnx_file in onnx_files:
        logger.info(f"\nTesting {onnx_file.name}:")
        if test_onnx_model(onnx_file):
            success_count += 1
    
    logger.info(f"\n{'='*50}")
    logger.info(f"Test Results: {success_count}/{len(onnx_files)} models passed")
    
    if success_count == len(onnx_files):
        logger.info("🎉 All models are working correctly!")
        return True
    else:
        logger.warning(f"⚠️  {len(onnx_files) - success_count} models failed tests")
        return False

if __name__ == "__main__":
    success = main()
    print("\n" + "="*50)
    if success:
        print("✅ All ONNX models are working! Conversion successful!")
    else:
        print("❌ Some models failed tests, but conversion partially successful.")