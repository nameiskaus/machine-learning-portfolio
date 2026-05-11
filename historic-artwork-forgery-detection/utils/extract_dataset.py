import zipfile
import os
import random

# 🔧 Configure paths
zip_path = "historic-art.zip"  # Update to your ZIP location
output_dir = "historic_art_dataset"  # All data will go here
num_images_to_extract = 2000  # Number of images to extract

def extract_sample_images(zip_ref, output_dir, num_images=2000):
    """Extract a random sample of images from the zip file"""

    # Find image files in the archive
    image_files = [
        f for f in zip_ref.namelist()
        if f.lower().endswith(('.jpg', '.jpeg', '.png')) and 'artwork' in f
    ]

    print(f"Found {len(image_files)} total images")

    # Random selection
    selected_images = (
        random.sample(image_files, num_images)
        if len(image_files) > num_images else image_files
    )

    print(f"Extracting {len(selected_images)} images...")

    # Output folder for images
    image_output_dir = os.path.join(output_dir, "images")
    os.makedirs(image_output_dir, exist_ok=True)

    for i, img_file in enumerate(selected_images):
        if i % 500 == 0:
            print(f"Extracted {i}/{len(selected_images)} images")

        filename = os.path.basename(img_file)
        target_path = os.path.join(image_output_dir, filename)

        # Skip if image already exists
        if not os.path.exists(target_path):
            zip_ref.extract(img_file, image_output_dir)

    print("Image extraction complete!")
    return selected_images


def extract_csv_files(zip_ref, output_dir):
    """Extract CSV metadata files from the zip"""
    csv_files = [f for f in zip_ref.namelist() if f.lower().endswith('.csv')]
    print(f"Found {len(csv_files)} CSV files.")

    for csv_file in csv_files:
        print(f"Extracting: {csv_file}")
        zip_ref.extract(csv_file, output_dir)

        extracted_path = os.path.join(output_dir, csv_file)
        final_path = os.path.join(output_dir, os.path.basename(csv_file))

        # Overwrite if destination already exists
        if os.path.exists(final_path):
            os.remove(final_path)

        os.rename(extracted_path, final_path)

        # Remove empty folder if needed
        subdir = os.path.dirname(extracted_path)
        if subdir != output_dir and os.path.isdir(subdir):
            try:
                os.rmdir(subdir)
            except OSError:
                pass

    print("CSV extraction complete!")
    return csv_files


# Main runner
if __name__ == "__main__":
    os.makedirs(output_dir, exist_ok=True)

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        extract_csv_files(zip_ref, output_dir)
        extract_sample_images(zip_ref, output_dir, num_images=num_images_to_extract)

    print(f"\nDone! All files extracted to: {output_dir}")
