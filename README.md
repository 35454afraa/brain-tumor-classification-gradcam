# Explainable Brain Tumor Classification from MRI using EfficientNetB0 and Grad-CAM
**Author:** FATHIMATHUL AFRA ISMAIL
**Date:** 4 October 2026

A deep learning project that classifies brain MRI images into four classes and uses **Grad-CAM** to highlight the image regions that influenced each prediction.

**Classes:** glioma, meningioma, pituitary, no tumor

## Objective

To improve the understanding of model predictions by using Grad-CAM to highlight the important regions of an MRI image.

## Dataset

Brain Tumor MRI Dataset (Kaggle), 4 classes: glioma, meningioma, pituitary, no tumor. Images are split into `Training` and `Testing` folders.

## Method

1. **Model:** EfficientNetB0 (ImageNet weights) with a global average pooling layer, dropout, and a 4-class softmax output. Input size is 224x224.
2. **Training:** first the base model was frozen and only the new classification layers were trained (10 epochs). Then the last 30 layers were unfrozen (BatchNorm layers kept frozen) and the model was fine-tuned for 5 epochs with a learning rate of 1e-5.
3. **Explainability:** Grad-CAM computed on the `block4c_add` layer, which gave heatmaps that were more localised than the final `top_activation` layer (7x7).
4. **Preprocessing note:** images are kept in the 0-255 range. EfficientNetB0 rescales inputs internally, so images must **not** be divided by 255.

## Results
| Model | Test accuracy |
|---|---|
| EfficientNetB0 (frozen base) | 80.31% |
| EfficientNetB0 (fine-tuned, final model) | **84.81%** |
| EfficientNetB0 (fine-tuned with augmentation, experiment) | 85.88% |

Per-class results of the final model:

| Class | Precision | Recall |
|---|---|---|
| glioma | 0.92 | 0.66 |
| meningioma | 0.73 | 0.80 |
| no tumor | 0.86 | 0.99 |
| pituitary | 0.91 | 0.94 |

The augmented model (v2) had slightly higher accuracy, but its Grad-CAM heatmaps were less focused on the lesion (they often highlighted the skull edge and background), so the fine-tuned model was kept as the final model.

The confusion matrix and Grad-CAM examples are saved in the `RESULTS` folder.

## Observations and limitations

- Grad-CAM highlights the lesion clearly in some cases (for example, a pituitary example where the strongest activation sits on the pituitary region).
- In other cases the model also attends to the skull edge, the image background, or text watermarks, so the model sometimes relies on regions that are not the tumour.
- In the sample test, one glioma image was predicted as pituitary with low confidence (40%).
- Grad-CAM shows where the model looked. It does not prove that the prediction is correct.
- This project is for learning and demonstration only. It is **not** a medical diagnosis tool.
- Glioma recall is 0.66, so about one third of glioma images are missed, and meningioma precision is 0.73, so some other tumours are labelled as meningioma.

## Project structure

```
DL MINI/
├── Brain_Tumor_Classification.ipynb   # training, evaluation, Grad-CAM
├── Brain_Tumor_Web_App/
│   ├── app.py                         # Flask app
│   ├── model/                         # saved .keras model
│   ├── static/ (uploads, results, style.css)
│   └── templates/index.html
├── DATASETS/ (Training, Testing)      # one folder per class
├── MODELS/                            # saved models
├── RESULTS/                           # confusion matrix, Grad-CAM images
├── SAMPLE_IMAGES/test.jpg
├── requirements.txt
└── README.md
```

## How to run

1. Install Python 3.11 and the requirements:
   ```
   pip install -r requirements.txt
   ```
2. **Notebook:** open `Brain_Tumor_Classification.ipynb` in VS Code or Jupyter and choose **Run All**. Set `TRAIN = True` to retrain the model, or keep `TRAIN = False` to load the saved model.
3. **Web app:**
   ```
   cd Brain_Tumor_Web_App
   python app.py
   ```
   Open `http://127.0.0.1:5000`, upload an MRI image, and the page shows the predicted class, confidence, Grad-CAM heatmap, and overlay.

## Tools

Python, TensorFlow/Keras, OpenCV, NumPy, Matplotlib, scikit-learn, Flask
