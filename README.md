# face-filter 🔍

A lightweight, high-performance command-line tool that filters and organizes photo libraries by matching faces using deep learning facial embeddings. Give it either a single reference photo or an entire folder of target people, point it at a source directory of images, and it will automatically extract and copy all matching photos into your output folder.

## 🚀 How It Works

`face-filter` bypasses raw pixel matching by leveraging modern computer vision pipelines:
1. **Embedding Extraction:** Passes your reference target face(s) through a deep convolutional neural network (`Facenet512` via DeepFace) to map facial features into a dense 512-dimensional vector space.
2. **Batch Scanning:** Iterates through candidate images, detects all faces present, and generates their respective embeddings.
3. **Distance Metric Matching:** Computes the **cosine distance** between the target vectors and candidate vectors. If any face falls below your configured threshold, the image is identified as a match and saved.

---

## 🛠️ Prerequisites & Installation

### Requirements
* Python 3.8+
* OpenCV, DeepFace, Click, NumPy, TensorFlow

### Installation

Clone the repository and install the required dependencies:

```bash
git clone [https://github.com/your-username/face-filter.git](https://github.com/your-username/face-filter.git)
cd face-filter
pip install -r requirements.txt