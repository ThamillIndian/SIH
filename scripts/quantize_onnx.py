from onnxruntime.quantization import quantize_dynamic, QuantType
import os
os.makedirs("artifacts", exist_ok=True)
quantize_dynamic("artifacts/best.onnx", "artifacts/best-int8.onnx",
                 weight_type=QuantType.QInt8)
print("[OK] Quantized -> artifacts/best-int8.onnx")
