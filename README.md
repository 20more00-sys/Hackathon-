# Brain Tumor MRI Classification

An educational four-class MRI image-classification prototype using ResNet50 transfer learning, held-out evaluation, optional Grad-CAM visualization, and a small Streamlit demo.

> **Medical disclaimer:** This project is not a medical device or diagnostic tool. It has not been clinically validated and must not be used for screening, diagnosis, treatment, triage, or patient-care decisions. Model scores and Grad-CAM visualizations are experimental only.

## For Judges – How to Run

**Fastest demo:** Open [`Hackathon_Colab.ipynb`](./Hackathon_Colab.ipynb) in Colab and select **Runtime → Run all**. It installs dependencies, downloads the dataset, compares a majority baseline with ResNet50, evaluates the provided test split, and saves `results/models/brain_tumor_model.keras`.

### Run the web app (Streamlit)

1. **Get the code** (clone it, or open the repo in GitHub Codespaces):
   ```bash
   git clone https://github.com/20more00-sys/Hackathon-.git
   cd Hackathon-
   ```

2. **Download the model** from the link in the [Model weights](#model-weights) section and save it as: results/models/brain_tumor_model.keras
(The file name and folder must match exactly.)

3. **Install the dependencies:**
```bash
python -m pip install -r requirements.txt
```

4. **Start the app** from the project root folder:
```bash
python -m streamlit run app/app.py
```

5. **Open it in the browser.** Streamlit prints a local address (usually `http://localhost:8501`).
- In **GitHub Codespaces**, click **Open in Browser** in the pop-up, or open the **Ports** tab and click the globe icon next to port 8501.

6. **Try it:** click **Browse files** and upload an MRI image (for example `my_images/image.png` from this repo). The app shows the predicted class, the probability scores, and a Grad-CAM heatmap.

> The first start can take a minute while TensorFlow loads. The app is an educational prototype, not a medical device.

The notebook trains a short fallback if project weights are not present in Colab. After downloading/saving the model into `results/models/` in your local checkout, the app uses it by default. See **Model weights** below.

The notebook evaluates the Kaggle-provided `Testing` folder; patient identifiers are unavailable, so patient-independent generalization is not established.

Grad-CAM visualizations are available for example errors in the notebook and after image uploads in the Streamlit app; they are qualitative only.

## Model weights

Download the trained model here (Colab-trained, 4 epochs, 85.4% test accuracy):https://drive.google.com/file/d/1GJrwvVoyRnvxhvMrw32OO5NkmJ49TpxD/view?usp=drive_link

### Dataset credit

The notebook downloads [Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset) by Masoud Nickparvar from Kaggle using `kagglehub`. The project records dataset version 2 and its CC BY 4.0 attribution. Do not redistribute MRI images without following the dataset license and source terms.

## Results in this repository

The existing validation-selected checkpoint reached **87.1% accuracy** and **0.868 macro F1** on the provided 1,600-image test split (400 images per class). A constant majority-class prediction scores **25.0%** on this balanced test set. Glioma recall was 73.0%, and meningioma recall was 77.8%.

These are dataset-specific experimental measurements from a **partial training run**; the configured full training schedule was not completed. The dataset audit also flagged 2,740 cross-split perceptual near-match candidates (not confirmed duplicates), and patient identifiers were unavailable. Do not interpret these results as clinical performance. See [the evaluation report](./results/reports/REAL_EVALUATION_REPORT.md), [training report](./results/reports/REAL_TRAINING_REPORT.md), and [one-page write-up](./docs/WRITEUP.md).

## Results note
- Saved project checkpoint (CPU run, partial training): 87.1% test accuracy.
- Colab-trained model (4 epochs): 85.4% test accuracy.

These two numbers come from different training runs and should not be mixed.

## Reproduce training locally

Download or otherwise provide the dataset with this layout:

```text
brain_tumor_data/
  Training/
    glioma/ meningioma/ notumor/ pituitary/
  Testing/
    glioma/ meningioma/ notumor/ pituitary/
```

Audit it and prepare exact-duplicate-safe train/validation splits while preserving the original test set:

```bash
python src/real_dataset_audit.py --data-dir brain_tumor_data --output-dir results
python src/prepare_real_splits.py \
  --data-dir brain_tumor_data \
  --output-dir results/prepared_splits \
  --report-dir results/reports \
  --seed 42
```

Train and evaluate the ResNet50 experiment:

```bash
python src/run_real_experiment.py \
  --data-dir brain_tumor_data \
  --prepared-dir results/prepared_splits \
  --output-dir results \
  --epochs 8 \
  --fine-tune-epochs 6 \
  --batch-size 32 \
  --img-size 224 \
  --seed 42
```

To run single-image inference or create a Grad-CAM visualization:

```bash
python src/predict.py \
  --model results/models/brain_tumor_model.keras \
  --image path/to/image.png \
  --gradcam-dir results/gradcam
```

## Project files

- `Hackathon_Colab.ipynb` — standalone Colab data, baseline, training, evaluation, and error-inspection workflow
- `src/` — reusable training, evaluation, prediction, dataset-audit, and Grad-CAM code
- `app/app.py` — Streamlit image upload demo
- `results/models/` — saved Keras checkpoints
- `results/plots/` and `results/reports/` — metrics, figures, and experiment documentation

Run the test suite with:

```bash
python -m pytest -q
```
