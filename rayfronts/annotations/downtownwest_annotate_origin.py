import json

target_classes = ['fountain_fx', 'food_cart', 'prop_recycle_bin', 'fire_hydrant', 'light_streetlight_complete']

# Load JSON
with open("downtown_west_labels_pose_bbox_v3.json", "r") as f:
    data = json.load(f)

# To store filtered and processed items
processed = []

# To track seen bbox_world values (for removing duplicates)
seen_bboxes = set()

for item in data:
    if "class" in item and item["class"] in target_classes:
        if "asset_root" in item and "FountainFX" in item["asset_root"]:
            item['class'] = 'fountain'
        elif "asset_root" in item and "food_cart" in item["asset_root"]:
            item['class'] = 'food cart'
        elif "asset_root" in item and "recycle_bin" in item["asset_root"]:
            item['class'] = 'recycle bin'
        elif "asset_root" in item and "streetlight_complete" in item["asset_root"]:
            item['class'] = 'traffic light'
        elif "asset_root" in item and "Fire_Hydrant" in item["asset_root"]:
            item['class'] = 'fire hydrant'

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
                
                processed.append({"class": item["class"], "bbox_world": {"center_xyz_m": center_rounded, "size_xyz_m": size_rounded}})


# Print the results
for item in processed:
    print({"class": item["class"], "bbox_world": {"center_xyz_m": item["bbox_world"]["center_xyz_m"], "size_xyz_m": item["bbox_world"]["size_xyz_m"]}})

# Optionally, save to a new JSON
with open("neighbor_filtered.json", "w") as f:
    json.dump(processed, f, indent=2)

