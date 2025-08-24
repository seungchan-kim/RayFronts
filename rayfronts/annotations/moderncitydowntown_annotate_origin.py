import json

processed = []

#add obelisk
processed.append({"class": 'obelisk', "bbox_world": {"center_xyz_m": [8.5, -35.0, 7.0], "size_xyz_m": [3.0, 3.0, 14.0]}})

#add yellow truck
processed.append({"class": 'yellow truck', "bbox_world": {"center_xyz_m": [-22.0, -10.0, 2.5], "size_xyz_m": [8.5, 4.0, 5.0]}})

#add cafe table 1
processed.append({"class": 'cafe table', "bbox_world": {"center_xyz_m": [-3.0, -73.5, 1.5], "size_xyz_m": [8.5, 7.0, 3.0]}})

#add cafe table 2
processed.append({"class": 'cafe table', "bbox_world": {"center_xyz_m": [9.0, -73.5, 1.5], "size_xyz_m": [8.5, 7.0, 3.0]}})

#add bankomat 1
processed.append({"class": 'bankomat', "bbox_world": {"center_xyz_m": [42.0, -39.0, 1.8], "size_xyz_m": [2.0, 6.0, 3.5]}})

#add bankomat 2
processed.append({"class": 'bankomat', "bbox_world": {"center_xyz_m": [24.5, -50.0, 1.8], "size_xyz_m": [2.0, 6.0, 3.5]}})

#add bankomat 3
processed.append({"class": 'bankomat', "bbox_world": {"center_xyz_m": [-25.5, -75.0, 1.8], "size_xyz_m": [2.0, 6.0, 3.5]}})


# Print the results
for item in processed:
    print({"class": item["class"], "bbox_world": {"center_xyz_m": item["bbox_world"]["center_xyz_m"], "size_xyz_m": item["bbox_world"]["size_xyz_m"]}})

# Optionally, save to a new JSON
with open("moderncitydowntown_filtered.json", "w") as f:
    json.dump(processed, f, indent=2)
