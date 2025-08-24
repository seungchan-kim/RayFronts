import json
import os

envs = ['FireAcademy', 'Neighborhood', 'ConstructionSite', 'AbandonedFactory', 'MilitaryBase', 'SnowyVillage', 'AbandonedCity', 'DowntownWest', 'ModernCityDowntown']
env_start_dict = {}
env_start_dict['FireAcademy'] = [[0,0,0,0,0,0],[-15,0,0,0,0,90],[30,30,1,0,0,-90]]
env_start_dict['Neighborhood'] = [[0,0,0,0,0,0],[-20,-80,0,0,0,90],[160,-19,0,0,0,180]]
env_start_dict['ConstructionSite'] = [[-27,8.5,0.2,0,0,0],[60,-3,0.2,0,0,-90],[48,-39,0.2,0,0,90]]
env_start_dict['AbandonedFactory'] = [[0,0,0.5,0,0,0],[-5,35,0.5,0,0,-90],[-5,-15,0.5,0,0,90]]
env_start_dict['MilitaryBase'] = [[1100,200,0,0,0,90],[1070,300,0,0,0,-90],[1114,28,0,0,0,90]]
env_start_dict['SnowyVillage'] = [[-152,-80,-2,0,0,90],[-145,20,-2.5,0,0,-90],[-200,-80,-2,0,0,0]]
env_start_dict['AbandonedCity'] = [[0,0,0,0,0,0],[0,80,0,0,0,-90],[5,-60,0,0,0,90]]
env_start_dict['DowntownWest'] = [[0,0,0,0,0,0],[2,-60,0,0,0,90],[-120,0,0,0,0,0]]
env_start_dict['ModernCityDowntown'] = [[-4,20,0.2,0,0,-90],[36,-80,0.2,0,0,90],[-17,-69,0.2,0,0,0]]

os.makedirs("transformed_annotations", exist_ok=True)

for env in envs:
    start_pose_pairs = env_start_dict[env]
    for start_pose in start_pose_pairs:
        tx,ty,tz,ox,oy,oz = start_pose
        transformed = []
        with open(f'raw_annotations/{env}.json','r') as f:
            data = json.load(f)
            for item in data:
                label = item['class']
                cx,cy,cz = item['bbox_world']['center_xyz_m']
                sx,sy,sz = item['bbox_world']['size_xyz_m']

                cx = cx - tx
                cy = cy - ty
                cz = cz - tz

                if oz == 0:
                    cx_, cy_, cz_ = cx, cy, cz
                    sx_, sy_, sz_ = sx, sy, sz
                elif oz == 90:
                    cx_, cy_, cz_ = cy, -cx, cz
                    sx_, sy_, sz_ = sy, sx, sz
                elif oz == 180:
                    cx_, cy_, cz_ = -cx, -cy, cz
                    sx_, sy_, sz_ = sx, sy, sz
                elif oz == -90:
                    cx_, cy_, cz_ = -cy, cx, cz
                    sx_, sy_, sz_ = sy, sx, sz
                else:
                    raise ValueError(f"Unsupported yaw {oz}")

                transformed.append({"class": label, "bbox_world": {"center_xyz_m": [cx_, cy_, cz_], "size_xyz_m": [sx_, sy_, sz_]}})
            
        out_file = f'transformed_annotations/{env}_t_x{tx}_y{ty}_z{tz}_o_x{ox}_y{oy}_z{oz}.json'
        with open(out_file,'w') as ff:
            json.dump(transformed, ff, indent=2)


