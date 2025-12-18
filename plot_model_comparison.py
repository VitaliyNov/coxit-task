import matplotlib.pyplot as plt

# Model names
models = [
    "YOLOv8n-cls",
    "YOLOv11n-cls",
    "YOLOv8m-cls",
    "EfficientNet-B2",
    "ConvNeXt-Tiny",
]

# Test accuracy (from your metrics)
accuracy = [
    0.8597,  # YOLOv8n
    0.8801,  # YOLOv11n
    0.8552,  # YOLOv8m
    0.9729,  # EfficientNet-B2
    0.7760,  # ConvNeXt-Tiny
]

# Approximate latency in ms / image (FP16, public info)
latency_ms = [
    1.1,   # YOLOv8n
    1.2,   # YOLOv11n
    3.0,   # YOLOv8m
    4.5,   # EfficientNet-B2
    5.0,   # ConvNeXt-Tiny
]

plt.figure(figsize=(8, 6))
plt.scatter(latency_ms, accuracy)

for x, y, name in zip(latency_ms, accuracy, models):
    plt.text(x + 0.05, y, name, fontsize=9)

plt.xlabel("Latency (ms / image)")
plt.ylabel("Test Accuracy")
plt.title("Accuracy vs Latency (Cabinet Classification)")
plt.grid(True)
plt.tight_layout()
plt.savefig("model_accuracy_latency_comparison.png", dpi=200)
plt.show()