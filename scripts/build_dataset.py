from pathlib import Path
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from tqdm import tqdm
from PIL import Image

if __name__ == "__main__":
    main_dir = Path("dataset")
    main_dir.mkdir(parents = True, exist_ok = True)
    df = pd.read_csv("train.csv")
    print(f"[INFO] Len of the dataset: {len(df)}")

    resolutions = []
    for _, row in tqdm(df.iterrows(), total = len(df), desc = "Building dataset"):
        target_name = row["target_name"]
        image_path = row["image_path"]
        image_name = image_path.split("/")[-1]

        try:
            with Image.open(image_path) as img:
                width, height = img.size
                res_str = f"{width}x{height}"
                resolutions.append([target_name, res_str])
        except: continue

    # Count images per class
    class_counts = df["target_name"].value_counts().sort_index()

    # Plot
    plt.figure(figsize=(12, 6))
    class_counts.plot(kind="bar")
    plt.xlabel("Target Name")
    plt.ylabel("Number of Images")
    plt.title("Image Count per Class")
    plt.xticks(rotation=45, ha="right")

    # Save plot inside dataset directory
    output_path = main_dir / "class_distribution.png"
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)

    print(f"Saved bar chart at: {output_path}")

    # Convert to DataFrame
    res_df = pd.DataFrame(resolutions, columns=["target_name", "resolution"])

    # Count resolution occurrences per class
    resolution_counts = (
        res_df.groupby(["target_name", "resolution"])
              .size()
              .reset_index(name="count")
    )

    # Pivot to wide format for multi-bar chart
    pivot_df = resolution_counts.pivot(
        index="target_name",
        columns="resolution",
        values="count"
    ).fillna(0)

    # Plot
    plt.figure(figsize=(20, 7))
    pivot_df.plot(kind="bar", figsize=(14, 7))
    plt.xlabel("Target Name")
    plt.ylabel("Number of Images")
    plt.title("Image Resolution Distribution per Class")
    plt.xticks(rotation=45, ha="right")

    plt.tight_layout()
    plt.savefig(main_dir / "image_resolution_distribution.png", dpi=300)
    print("Saved image_resolution_distribution.png inside dataset/")

    # Plotting the distribution target wrt to target name
    plt.figure(figsize = (20, 7))
    sns.histplot(df, x = 'target', hue='target_name')
    plt.savefig(main_dir / "distribution_of_target_biomass.png", dpi = 300)
    print("Saved distribution_of_target_biomass.png inside dataset/")




