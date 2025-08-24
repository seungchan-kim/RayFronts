import json

target_classes = ['radio_tower', 'white_water_tower', 'fuel_tank', 'green_container', 'red_container', 'blue_container', 'tree']

# Load JSON
with open("fire_academy_manual_labels_pose_bbox_nopad_v3.json", "r") as f:
    data = json.load(f)

# To store filtered and processed items
processed = []

# To track seen bbox_world values (for removing duplicates)
seen_bboxes = set()

for item in data:
    if "label" in item and any(tc in item['label'] for tc in target_classes):
        if "asset_root" in item and "radio_tower" in item["asset_root"]:
            item['label'] = 'radio tower'
        elif "asset_root" in item and "white_water_tower" in item["asset_root"]:
            item['label'] = 'water tower'
        elif "asset_root" in item and "fuel_tank" in item["asset_root"]:
            item['label'] = 'fuel tank'
        elif "asset_root" in item and "green_container" in item["asset_root"]:
            item['label'] = 'green container'
        elif "asset_root" in item and "red_container" in item["asset_root"]:
            item['label'] = 'red container'
        elif "asset_root" in item and "blue_container" in item["asset_root"]:
            item['label'] = 'blue container'
        elif "asset_root" in item and "tree" in item["asset_root"]:
            item['label'] = 'tree'

        bbox = item.get("bbox_world", {})
        center = bbox.get("center_xyz")
        size = bbox.get("size_xyz")
        
        if center and size:
            # Round values to 1 decimal
            center_rounded = [round(c, 1) for c in center]
            size_rounded = [round(s, 1) for s in size]
            
            # Create a tuple to check for duplicates
            bbox_tuple = (tuple(center_rounded), tuple(size_rounded))
            
            if bbox_tuple not in seen_bboxes:
                seen_bboxes.add(bbox_tuple)
                
                # Update the item with rounded values
                item["bbox_world"]["center_xyz"] = center_rounded
                item["bbox_world"]["size_xyz"] = size_rounded
                
                processed.append({"class": item["label"], "bbox_world": {"center_xyz_m": center_rounded, "size_xyz_m": size_rounded}})


# Print the results
for item in processed:
    print({"class": item["class"], "bbox_world": {"center_xyz_m": item["bbox_world"]["center_xyz_m"], "size_xyz_m": item["bbox_world"]["size_xyz_m"]}})

# Optionally, save to a new JSON
with open("fireacademy_filtered.json", "w") as f:
    json.dump(processed, f, indent=2)

