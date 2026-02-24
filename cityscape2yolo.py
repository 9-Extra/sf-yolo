import os
import json
import tqdm
import argparse

from pathlib import Path

YAML_TEMPLATE = """
# Train/val/test sets as 1) dir: path/to/imgs, 2) file: path/to/imgs.txt, or 3) list: [path/to/imgs1, path/to/imgs2, ..]

path: {path}

train:
  - images/train

val:
  - images/val

test:
  - images/val

# Classes
nc: {nc}  # number of classes
names: {names}  # class names

"""

def polygon_to_normalized_coords(polygon, img_width, img_height):
    normalized_coords = []
    for point in polygon:
        x_norm = point[0] / img_width
        y_norm = point[1] / img_height
        normalized_coords.append((x_norm, y_norm))
    return normalized_coords

def convert_cityscapes_to_yolov8(json_path, output_file, class_mapping):
    data = json.load(open(json_path))

    img_width = data['imgWidth']
    img_height = data['imgHeight']
    
    annotations = []
    for obj in data['objects']:
        label = obj['label']
        if label not in class_mapping:
            continue
        class_id = class_mapping[label]
        polygon = obj['polygon']
        norm_coords = polygon_to_normalized_coords(polygon, img_width, img_height)
        flattened_coords = [coord for point in norm_coords for coord in point]
        annotations.append((class_id, *flattened_coords))
    
    with open(output_file, 'w') as out_file:
        for ann in annotations:
            out_file.write(' '.join(map(str, ann)) + '\n')

def process_folder(image_path, source_label_path, output_dir, suffix, class_mapping):
    source_image_dir = Path(image_path)
    source_label_dir = Path(source_label_path)
    output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    open(output_dir / "cityscapes.yaml", "w").write(YAML_TEMPLATE.format(path=output_dir.absolute(), nc=len(class_mapping), names=repr(list(class_mapping.keys()))))
    
    image_dir = output_dir / "images"
    label_dir = output_dir / "labels"
    
    json_post_fix = "_gtFine_polygons.json"
    foggy_img_post_fix = f"_leftImg8bit{suffix}.png"
    target_img_post_fix = ".png"
    
    clips = ["train", "val", "test"]
    
    for clip in clips:
        source_image_dir_clip = source_image_dir / clip
        source_label_dir_clip = source_label_dir / clip
        
        image_dir_clip = image_dir / clip
        label_dir_clip = label_dir / clip
        image_dir_clip.mkdir(parents=True, exist_ok=True)
        label_dir_clip.mkdir(parents=True, exist_ok=True)
        # leftImg8bit_foggy/train/jena/jena_000092_000019_leftImg8bit_foggy_beta_0.02.png
        # jena_000092_000019_leftImg8bit_foggy_beta_0.02.png
        for json_path in tqdm.tqdm(list(source_label_dir_clip.glob("**/*.json")), desc=f"Processing {clip}"):
            image_name = json_path.name.replace(json_post_fix, foggy_img_post_fix)
            target_image_name = json_path.name.replace(json_post_fix, target_img_post_fix)
            aachen = json_path.parent.name # city name
            image_path = source_image_dir_clip / aachen / image_name
            target_image_path = image_dir_clip / target_image_name
            output_file = label_dir_clip / target_image_name.replace(".png", ".txt")
            
            convert_cityscapes_to_yolov8(json_path, output_file, class_mapping)
            target_image_path.unlink(True)
            os.link(image_path, target_image_path)
            

# Example class mapping based on trainId from Cityscapes labels
class_mapping = {
    'unlabeled': 255,
    'ego vehicle': 255,
    'rectification border': 255,
    'out of roi': 255,
    'static': 255,
    'dynamic': 255,
    'ground': 255,
    'road': 255,
    'sidewalk': 255,
    'parking': 255,
    'rail track': 255,
    'building': 255,
    'wall': 255,
    'fence': 255,
    'guard rail': 255,
    'bridge': 255,
    'tunnel': 255,
    'pole':255,
    'polegroup': 255,
    'traffic light': 255,
    'traffic sign': 255,
    'vegetation': 255,
    'terrain': 255,
    'sky': 255,
    'person': 4,
    'rider': 5,
    'car': 2,
    'truck': 7,
    'bus': 0,
    'caravan': 255,
    'trailer': 255,
    'train': 6,
    'motorcycle': 3,
    'bicycle': 1,
    'license plate': 255,
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert Cityscapes dataset to YOLO format")
    parser.add_argument("--image_path", type=str, default="leftImg8bit", help="Path to the Cityscapes image folder")
    parser.add_argument("--source_label_path", type=str, default="gtFine", help="Path to the Cityscapes label folder")
    parser.add_argument("--output_dir", type=str, default="yolov5_format", help="Output directory for YOLO format")
    parser.add_argument("--foggy_beta", type=str, choices=["", "_foggy_beta_0.005", "_foggy_beta_0.01", "_foggy_beta_0.02"], default="", help="foggy beta value")
    
    args = parser.parse_args()
    
    class_mapping = {k: v for k, v in class_mapping.items() if v != 255}
    process_folder(args.image_path, args.source_label_path, args.output_dir, args.foggy_beta, class_mapping)

    # You can use the rename command to make the names of the original image files and annotation files the same.
    # rename 's/gtFine_polygons/leftImg8bit/' *.txt
