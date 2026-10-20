# Face Filter CLI Tool

A lightweight, high-performance command-line tool that filters and organizes photo libraries by matching faces using deep learning facial embeddings. Instead of relying on raw pixel matching, this tool extracts deep facial features via the **Facenet512** architecture and performs distance metric matching in a dense vector space.

---

## 🚀 How It Works

1. **Embedding Extraction:** Passes your reference target face(s) through a deep convolutional neural network to map facial features into a dense 512-dimensional vector space.
2. **Batch Scanning:** Iterates through candidate images in your source directory, detects all faces present using OpenCV, and generates their respective embeddings.
3. **Distance Metric Matching:** Computes the cosine distance between the target vectors and candidate vectors. If any face falls below your configured threshold, the image is identified as a match and automatically copied to your output folder.

---

## 🛠️ Prerequisites & Installation

Make sure you have Python 3.8+ installed, then install the required dependencies:

```bash
pip install deepface tf-keras opencv-python scikit-learn numpy
```

---

## 📖 Usage

Run the tool from your terminal by specifying your targets, source directory, and output destination:

```bash
python face_filter.py --targets ./reference_face.jpg --source ./my_vacation_photos --output ./matched_results --threshold 0.40
```

### Command-Line Arguments

* `--targets` *(Required)*: Path to a single target photo file or a folder containing multiple target photos.
* `--source` *(Required)*: Path to the source directory of candidate images you want to scan.
* `--output` *(Required)*: Path to the destination folder where matching images will be copied.
* `--threshold` *(Optional)*: Cosine distance threshold for `Facenet512` (default: `0.40`). Lower values are stricter; higher values are more lenient.

---

## ⚙️ Troubleshooting & Tips

* **No Faces Detected:** Ensure your reference and source photos have clear, well-lit frontal or near-frontal faces. If a photo fails detection, the script will safely skip it and log a warning.
* **Tuning Accuracy:** If you are getting false positives (wrong people matching), lower your threshold to `0.35` or `0.30`. If you are missing true matches due to lighting or head tilt, try increasing it slightly to `0.45`.