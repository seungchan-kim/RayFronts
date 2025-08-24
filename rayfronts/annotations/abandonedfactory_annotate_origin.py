import json

processed = []

#add water tower 1 
processed.append({"class": 'water tower', "bbox_world": {"center_xyz_m": [20.0, -82.0, 25.0], "size_xyz_m": [10.0, 10.0, 50.0]}})

#add water tower 2
processed.append({"class": 'water tower', "bbox_world": {"center_xyz_m": [-90.0, 0.0, 30.0], "size_xyz_m": [12.0, 12.0, 60.0]}})

#add pipe 1
processed.append({"class": 'pipe', "bbox_world": {"center_xyz_m": [-18.0, -5.0, 1.5], "size_xyz_m": [4.0, 35.0, 3.0]}})

#add pipe 2
processed.append({"class": 'pipe', "bbox_world": {"center_xyz_m": [-28.0, -10.0, 15.0], "size_xyz_m": [16.0, 45.0, 5.0]}})

#add pipe 3
processed.append({"class": 'pipe', "bbox_world": {"center_xyz_m": [-34.0, -10.0, 2.5], "size_xyz_m": [16.0, 45.0, 5.0]}})

#add pipe 4
processed.append({"class": 'pipe', "bbox_world": {"center_xyz_m": [-27.0, 15.0, 8.0], "size_xyz_m": [4.0, 10.0, 16.0]}})

#add white silo
processed.append({"class": 'white silo', "bbox_world": {"center_xyz_m": [-28.0, 63.0, 10.0], "size_xyz_m": [20.0, 32.0, 20.0]}})

#add building 1
processed.append({"class": 'building', "bbox_world": {"center_xyz_m": [-47.0, -44.0, 15.0], "size_xyz_m": [55.0, 30.0, 30.0]}})

#add building 2
processed.append({"class": 'building', "bbox_world": {"center_xyz_m": [18.0, 31.5, 12.5], "size_xyz_m": [23.0, 29.0, 25.0]}})

#add building 3
processed.append({"class": 'building', "bbox_world": {"center_xyz_m": [-30.0, 18.0, 14.0], "size_xyz_m": [30.0, 20.0, 28.0]}})

#add building 4
processed.append({"class": 'building', "bbox_world": {"center_xyz_m": [-48.0, 88.0, 16.0], "size_xyz_m": [15.0, 12.0, 32.0]}})

# Print the results
for item in processed:
    print({"class": item["class"], "bbox_world": {"center_xyz_m": item["bbox_world"]["center_xyz_m"], "size_xyz_m": item["bbox_world"]["size_xyz_m"]}})

# Optionally, save to a new JSON
with open("abandonedfactory_filtered.json", "w") as f:
    json.dump(processed, f, indent=2)

