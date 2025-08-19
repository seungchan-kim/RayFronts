import json

# Load JSON
with open("military_base_extra_labels_pose_bbox_v7.json", "r") as f:
    data = json.load(f)

# To store filtered and processed items
processed = []

# To track seen bbox_world values (for removing duplicates)
seen_bboxes = set()

for item in data:
    if "class" in item:
        if 'radiotower' in item['class']:
            item['class'] = 'radio tower'
        elif 'guardtower' in item['class']:
            item['class'] = 'guard tower'
        elif 'airfield' == item['class']:
            item['class'] = 'airfield'
        elif 'mobileradar' in item['class']:
            item['class'] = 'mobile radar'
        else:
            continue

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
                item["bbox_world"]["center_xyz_m"] = center_rounded
                item["bbox_world"]["size_xyz_m"] = size_rounded
                
                processed.append({"class": item["class"], "bbox_world": {"center_xyz_m": center_rounded, "size_xyz_m": size_rounded}})


#add helicopter 1
processed.append({"class": 'helicopter', "bbox_world": {"center_xyz_m": [1160.0, 253.0, 2.5], "size_xyz_m": [8.0, 12.0, 5.0]}})

#add helicopter 2 
processed.append({"class": 'helicopter', "bbox_world": {"center_xyz_m": [1095.0, -42.0, 0.5], "size_xyz_m": [8.0, 12.0, 5.0]}})


#add ATV 1
processed.append({"class": 'ATV', "bbox_world": {"center_xyz_m": [1050.0, 270.0, 1.5], "size_xyz_m": [3.0, 4.0, 3.0]}})

#add ATV2
processed.append({"class": 'ATV', "bbox_world": {"center_xyz_m": [1054.0, 225.0, 1.5], "size_xyz_m": [3.0, 4.0, 3.0]}})

#add ATV3
processed.append({"class": 'ATV', "bbox_world": {"center_xyz_m": [1060.0, 202.0, 1.5], "size_xyz_m": [3.0, 4.0, 3.0]}})

#add ATV4
processed.append({"class": 'ATV', "bbox_world": {"center_xyz_m": [1094.0, 244.0, 1.5], "size_xyz_m": [4.0, 3.0, 3.0]}}) 

#add bridge
processed.append({"class": 'bridge', "bbox_world": {"center_xyz_m": [1118.0, 75.0, 2.5], "size_xyz_m": [20.0, 80.0, 5.0]}})

# Print the results
for item in processed:
    print({"class": item["class"], "bbox_world": {"center_xyz_m": item["bbox_world"]["center_xyz_m"], "size_xyz_m": item["bbox_world"]["size_xyz_m"]}})

# Optionally, save to a new JSON
with open("militarybase_filtered.json", "w") as f:
    json.dump(processed, f, indent=2)

