import json

target_classes = ['house', 'shippingcontainer', 'porch', 'tunnel','fountain', 'tree', 'busstop', 'chimney', 'billboard']

# Load JSON
with open("neighborhood_asset_labels_pose_bbox_v1.json", "r") as f:
    data = json.load(f)

# To store filtered and processed items
processed = []

# To track seen bbox_world values (for removing duplicates)
seen_bboxes = set()

for item in data:
    if "class" in item and item["class"] in target_classes:
        if "asset_root" in item and "ShippingContainer" in item["asset_root"]:
            if "ShippingContainer3" in item["asset_root"] or "ShippingContainer7" in item["asset_root"]:
                item['class'] = 'blue container'
            elif "ShippingContainer9" in item["asset_root"]:
                item['class'] = 'red container'
            else:
                continue
        if "asset_root" in item and "Merc_Hovercar" in item["asset_root"]:
            item['class'] = 'yellow car'
        elif "asset_root" in item and "Nissan" in item["asset_root"]:
            item['class'] = 'red car'


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


#add red car
processed.append({"class": 'red car', "bbox_world": {"center_xyz_m": [90.5, -29.5, 0.5], "size_xyz_m": [3.0, 4.0, 2.0]}})

#add yellow car
processed.append({"class": 'yellow car', "bbox_world": {"center_xyz_m": [-1.0, 30.0, 0.5], "size_xyz_m": [5.0, 4.0, 2.0]}})


# Print the results
for item in processed:
    print({"class": item["class"], "bbox_world": {"center_xyz_m": item["bbox_world"]["center_xyz_m"], "size_xyz_m": item["bbox_world"]["size_xyz_m"]}})

# Optionally, save to a new JSON
with open("neighbor_filtered.json", "w") as f:
    json.dump(processed, f, indent=2)

