
# ============================================================
# Mangrove Degradation and Health Assessment in the Niger Delta
# Using NDVI, NDWI, and Machine Learning (Synthetic Data)
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# ------------------------------------------------------------
# 1. Synthetic Multispectral Data Generation
# ------------------------------------------------------------

def generate_synthetic_bands(size=100):
    """
    Generates synthetic reflectance bands:
    - RED, NIR, GREEN
    Mimics mangrove vegetation and degraded areas
    """

    nir = np.random.normal(0.65, 0.08, (size, size))
    red = np.random.normal(0.25, 0.05, (size, size))
    green = np.random.normal(0.35, 0.06, (size, size))

    degradation_mask = np.zeros((size, size))
    cx, cy = np.random.randint(25, size - 25, 2)
    radius = np.random.randint(12, 25)

    y, x = np.ogrid[:size, :size]
    degraded_area = (x - cx)**2 + (y - cy)**2 <= radius**2
    degradation_mask[degraded_area] = 1

    nir[degraded_area] *= 0.4
    red[degraded_area] *= 1.4
    green[degraded_area] *= 1.2

    return red, nir, green, degradation_mask

# ------------------------------------------------------------
# 2. NDVI & NDWI Computation
# ------------------------------------------------------------

def compute_ndvi(nir, red):
    return (nir - red) / (nir + red + 1e-6)

def compute_ndwi(green, nir):
    return (green - nir) / (green + nir + 1e-6)

# ------------------------------------------------------------
# 3. Dataset Creation
# ------------------------------------------------------------

def create_dataset(samples=400, size=100):
    features = []
    labels = []

    for _ in range(samples):
        red, nir, green, mask = generate_synthetic_bands(size)
        ndvi = compute_ndvi(nir, red)
        ndwi = compute_ndwi(green, nir)

        for i in range(size):
            for j in range(size):
                features.append([ndvi[i, j], ndwi[i, j]])
                labels.append(mask[i, j])

    return np.array(features), np.array(labels)

X, y = create_dataset(samples=120)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# ------------------------------------------------------------
# 4. Machine Learning Model (Random Forest)
# ------------------------------------------------------------

model = RandomForestClassifier(
    n_estimators=150,
    max_depth=12,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# ------------------------------------------------------------
# 5. Evaluation
# ------------------------------------------------------------

y_pred = model.predict(X_test)

print("Classification Report:")
print(classification_report(
    y_test, y_pred,
    target_names=["Healthy Mangrove", "Degraded Mangrove"]
))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# ------------------------------------------------------------
# 6. Spatial Health Assessment Visualization
# ------------------------------------------------------------

def visualize_health_map():
    red, nir, green, ground_truth = generate_synthetic_bands(size=100)

    ndvi = compute_ndvi(nir, red)
    ndwi = compute_ndwi(green, nir)

    feature_stack = np.stack([ndvi, ndwi], axis=-1)
    reshaped = feature_stack.reshape(-1, 2)

    prediction = model.predict(reshaped)
    prediction_map = prediction.reshape(100, 100)

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.title("NDVI Map")
    plt.imshow(ndvi, cmap="YlGn")
    plt.colorbar()
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.title("NDWI Map")
    plt.imshow(ndwi, cmap="Blues")
    plt.colorbar()
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.title("Predicted Mangrove Health")
    plt.imshow(prediction_map, cmap="RdYlGn")
    plt.axis("off")

    plt.tight_layout()
    plt.show()

visualize_health_map()
