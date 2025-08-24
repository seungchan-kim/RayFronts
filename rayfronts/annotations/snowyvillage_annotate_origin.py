import json

target_classes = ['water_tower', 'fisherman_animated_sequence', 'car', 'outhouse', 'water_ice_plane']

# Load JSON
with open("snowy_village_labels_pose_bbox_v2.json", "r") as f:
    data = json.load(f)

# To store filtered and processed items
processed = []

# To track seen bbox_world values (for removing duplicates)
seen_bboxes = set()

for item in data:
    if "class" in item and item["class"] in target_classes:
        if "asset_root" in item and "WaterTower" in item["asset_root"]:
            item['class'] = 'water tower'
        elif "asset_root" in item and "Fisherman03_Animated_Sequence" in item["asset_root"]:
            item['class'] = 'human'
        elif "asset_root" in item and "Car01" in item["asset_root"]:
            item['class'] = 'car'
        elif "asset_root" in item and "Outhouse" in item["asset_root"]:
            item['class'] = 'outhouse'
        elif "asset_root" in item and "WaterIcePlane" in item["asset_root"]:
            item['class'] = 'pond'

        bbox = item.get("bbox_world", {})
        center = bbox.get("center_xyz_m")
        size = bbox.get("size_xyz_m")
        
        if center and size:
            # Round values to 1 decimal
            center_rounded = [round(c, 1) for c in center]
            size_rounded = [round(s, 1) for s in size]
            
            # Create a tuple to check for duplicates
            bbox_tuple = (tuple(center_rounded), tuple(size_rounded))
            
            if bbox_tuple not in seen_bboxes:
                seen_bboxes.add(bbox_tuple)
                
                # Update the item with rounded values
                item["bbox_world"]["center_xyz_m"] = center_rounded
                item["bbox_world"]["size_xyz_m"] = size_rounded
                
                processed.append({"class": item["class"], "bbox_world": {"center_xyz_m": center_rounded, "size_xyz_m": size_rounded}})


# Print the results
for item in processed:
    print({"class": item["class"], "bbox_world": {"center_xyz_m": item["bbox_world"]["center_xyz_m"], "size_xyz_m": item["bbox_world"]["size_xyz_m"]}})

# Optionally, save to a new JSON
with open("snowyvillage_filtered.json", "w") as f:
    json.dump(processed, f, indent=2)

