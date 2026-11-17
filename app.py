
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Render plots without requiring a display

import matplotlib.pyplot as plt
import pandas as pd
import gradio as gr

from src.predict import load_model, predict
from src.gradcam import gradcam_overlay

ROOT = Path(__file__).resolve().parent
ART = ROOT / "artifacts_v2"


# ---------- Load project resources once at startup ----------

model, classes, model_name = load_model(ART)

# Confidence levels for the messages shown to the user
NOT_PET_BELOW = 0.40     # all 28 non-pet test photos scored below 40%
UNCLEAR_BELOW = 0.50     # 50–60%: show the guesses, but warn the user

with open(ART / "metrics.json") as f:
    metrics = json.load(f)

per_class = pd.read_csv(ART / "per_class.csv")
pairs = pd.read_csv(ART / "confused_pairs.csv")
by_source = pd.read_csv(ART / "accuracy_by_source.csv").set_index("source")

example_files = sorted(
    str(p) for p in (ART / "examples").glob("*.jpg")
)


# ---------- Tab 1: Predict ----------

def classify(image):

    # Handle cases where no image has been uploaded
    if image is None:
        return {}, None, "Upload a photo of a cat or dog."

    # Generate the top-5 breed predictions
    top5 = predict(model, classes, image, k=5)
    best_breed, best_prob = top5[0]

    # Generate a Grad-CAM visualization for the top prediction
    heatmap = gradcam_overlay(
        model,
        model_name,
        image,
        classes.index(best_breed)
    )

    # Choose the message based on how confident the model is
    if best_prob < NOT_PET_BELOW:
        message = (
            f"🚫 **This might not be a photo of a cat or a dog** "
            f"(top score {best_prob:.0%})."
        )
    elif best_prob < UNCLEAR_BELOW:
        message = (
            f"⚠️ **This might not be a clear photo** (top score {best_prob:.0%}). "
            f"Best guesses: **{top5[0][0]}**, **{top5[1][0]}** or **{top5[2][0]}**. "
            "It may also be a mixed breed or a breed the model doesn't know yet."
        )
    else:
        message = f"### {best_breed}\n{best_prob:.0%} confidence"

    # Convert predictions into the dictionary format expected by gr.Label
    label_dict = {
        breed: prob for breed, prob in top5
    }

    return label_dict, heatmap, message


# ---------- Tab 2: Model performance ----------

def per_class_plot(choice):

    # Filter results according to the selected species
    if choice == "Cats":
        df = per_class[per_class["species"] == "cat"]
    elif choice == "Dogs":
        df = per_class[per_class["species"] == "dog"]
    else:
        df = per_class

    # Sort breeds by accuracy
    df = df.sort_values("accuracy")

    # Show only the 30 hardest breeds so the chart stays readable with 137 breeds
    if len(df) > 30:
        df = df.head(30)
        title = f"30 hardest breeds: {choice}"
    else:
        title = f"Per-breed accuracy: {choice}"

    # Create a horizontal bar chart
    fig, ax = plt.subplots(
        figsize=(8, 0.28 * len(df) + 1)
    )

    ax.barh(df["breed"], df["accuracy"] * 100)
    ax.set_xlabel("Test accuracy (%)")
    ax.set_xlim(0, 100)
    ax.set_title(title)

    fig.tight_layout()
    return fig


# Prepare a Markdown summary of the model's evaluation metrics
metrics_md = f"""
| Metric | Value |
|---|---|
| Test accuracy | **{metrics['test_acc']:.1%}** |
| Top-5 accuracy | {metrics['top5_acc']:.1%} |
| Macro F1 | {metrics['macro_f1']:.3f} |
| Cats / Dogs accuracy | {metrics['cat_acc']:.1%} / {metrics['dog_acc']:.1%} |
| Test images | {metrics['n_test']:,} |
| Oxford / Stanford / Indie accuracy | {by_source.loc['oxford', 'accuracy']:.1%} / {by_source.loc['stanford', 'accuracy']:.1%} / {by_source.loc['indie', 'accuracy']:.1%} |

**Baseline vs fine-tuned (test accuracy):**
Frozen EfficientNet-B0 **{metrics['baseline_test_acc']:.1%}**
→ fine-tuned {metrics['model']} **{metrics['test_acc']:.1%}**

**Confidence levels:** below 40% → "might not be a cat or dog" · 40–60% → "might not be a clear photo" · 60%+ → breed shown
(validation-tuned threshold: {metrics['threshold']:.2f})

Rejected {metrics['nonpet_rejected']}/{metrics['nonpet_total']}
non-pet photos and wrongly rejected
{metrics['pets_wrongly_rejected']:.1%} of correctly classified pets.
"""


# ---------- Tab 3: Error explorer ----------

# Create a lookup dictionary for confused breed pairs
pair_lookup = {}

for _, row in pairs.iterrows():

    label = (
        f"{row['true']} → {row['predicted']} "
        f"({row['count']} images)"
    )

    pair_lookup[label] = row


def show_pair(choice):

    # Retrieve information about the selected breed pair
    row = pair_lookup[choice]

    # Construct the filename prefix used for saved error images
    prefix = (
        f"{row['true']}__{row['predicted']}__"
    ).replace(" ", "_")

    # Find example images for the selected confusion pair
    images = sorted(
        str(p) for p in (ART / "errors").glob(prefix + "*.jpg")
    )

    # Prepare a short explanation
    text = (
        f"**{row['count']}** test images of **{row['true']}** "
        f"were predicted as **{row['predicted']}**. Examples:"
    )

    return text, images


# ---------- Gradio interface layout ----------

with gr.Blocks(title="Pet Breed Classifier") as demo:

    gr.Markdown(
        "# 🐾 Pet Breed Classifier\n"
        f"{len(classes)} cat and dog breeds · "
        f"fine-tuned {metrics['model']} · "
        f"test accuracy {metrics['test_acc']:.1%}"
    )

    # Tab 1: Upload an image and predict its breed
    with gr.Tab("Predict"):

        with gr.Row():

            with gr.Column():
                image_in = gr.Image(
                    type="pil",
                    sources=["upload", "webcam"],
                    label="Photo"
                )

                predict_btn = gr.Button(
                    "Predict",
                    variant="primary"
                )

                gr.Examples(
                    examples=[[f] for f in example_files],
                    inputs=[image_in]
                )

            with gr.Column():
                message_out = gr.Markdown()

                label_out = gr.Label(
                    num_top_classes=5,
                    label="Top 5 breeds"
                )

                cam_out = gr.Image(
                    label="Grad-CAM: where the model looked"
                )

        # Connect the Predict button to the classification function
        predict_btn.click(
            classify,
            inputs=image_in,
            outputs=[label_out, cam_out, message_out]
        )

    # Tab 2: Display model evaluation results
    with gr.Tab("Model performance"):

        gr.Markdown(metrics_md)

        # Allow users to filter breed accuracy by species
        species_radio = gr.Radio(
            ["All", "Cats", "Dogs"],
            value="All",
            label="Show"
        )

        bar_plot = gr.Plot(
            value=per_class_plot("All")
        )

        # Update the chart when the selected species changes
        species_radio.change(
            per_class_plot,
            inputs=species_radio,
            outputs=bar_plot
        )

        # Display the normalized confusion matrix
        gr.Image(
            value=str(ART / "confusion.png"),
            label="Confusion matrix (row-normalized)",
            interactive=False
        )

    # Tab 3: Explore commonly confused breed pairs
    with gr.Tab("Error explorer"):

        first_pair = list(pair_lookup)[0]

        pair_dd = gr.Dropdown(
            choices=list(pair_lookup),
            value=first_pair,
            label="Most-confused pairs"
        )

        pair_text = gr.Markdown()

        pair_gallery = gr.Gallery(
            columns=4,
            height="auto",
            label="Misclassified test images"
        )

        # Update the gallery when a different breed pair is selected
        pair_dd.change(
            show_pair,
            inputs=pair_dd,
            outputs=[pair_text, pair_gallery]
        )

        # Display examples when the page first loads
        demo.load(
            show_pair,
            inputs=pair_dd,
            outputs=[pair_text, pair_gallery]
        )


# Launch the Gradio application
if __name__ == "__main__":
    demo.launch()
