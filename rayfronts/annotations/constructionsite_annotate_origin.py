import json

target_classes = ['forklift', 'biotoilet', 'cargo_container', 'cablingwinches', 'asphaltroller']

# Load JSON
with open("construction_site_simple_labels_pose_bbox_v5.json", "r") as f:
    data = json.load(f)

# To store filtered and processed items
processed = []

# To track seen bbox_world values (for removing duplicates)
seen_bboxes = set()

for item in data:
    if "label" in item and item["label"] in target_classes:
        if "asset_root" in item and "Cargo_Container" in item["asset_root"]:
            #print(item["asset_root"])
            if "Cargo_Container16" in item["asset_root"] or "Cargo_Container53" in item["asset_root"] or "Cargo_Container56" in item["asset_root"] or "Cargo_Container63" in item["asset_root"] or "Cargo_Container64" in item["asset_root"] or "Cargo_Container65" in item["asset_root"] or "Cargo_Container66" in item["asset_root"] or "Cargo_Container67" in item["asset_root"] or "Cargo_Container68" in item["asset_root"]:
                item['label'] = 'red container'
            else:
                continue
        
        if "label" in item and "cablingwinches" in item["label"]:
            item['label'] = 'cabling winches'
        if "label" in item and "asphaltroller" in item["label"]:
            item['label'] = 'asphalt roller'

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
                
                processed.append({"class": item["label"], "bbox_world": {"center_xyz_m": center_rounded, "size_xyz_m": size_rounded}})


#add yellow towercrane
processed.append({"class": 'yellow towercrane', "bbox_world": {"center_xyz_m": [-3.0, 12.0, 25.0], "size_xyz_m": [5.0, 4.0, 50.0]}})

#add orange towercrane
processed.append({"class": 'orange towercrane', "bbox_world": {"center_xyz_m": [58.5, -42.0, 40.0], "size_xyz_m": [6.0, 4.0, 80.0]}})

#add construction lift 
processed.append({"class": 'construction lift', "bbox_world": {"center_xyz_m": [66.0, -35.0, 16.0], "size_xyz_m": [5.0, 5.0, 32.0]}})

#add blue tarp1
processed.append({"class": 'blue tarp', "bbox_world": {"center_xyz_m": [66.0, -18.0, 16.0], "size_xyz_m": [3.0, 20.0, 32.0]}})

#add blue tarp2
processed.append({"class": 'blue tarp', "bbox_world": {"center_xyz_m": [25.0, -48.0, 8.0], "size_xyz_m": [10.0, 3.0, 16.0]}})

#add orange tarp
processed.append({"class": 'orange tarp', "bbox_world": {"center_xyz_m": [-20.0, 22.0, 10.0], "size_xyz_m": [20.0, 3.0, 20.0]}})

# Print the results
for item in processed:
    print({"class": item["class"], "bbox_world": {"center_xyz_m": item["bbox_world"]["center_xyz_m"], "size_xyz_m": item["bbox_world"]["size_xyz_m"]}})

# Optionally, save to a new JSON
with open("constructionsite_filtered.json", "w") as f:
    json.dump(processed, f, indent=2)

