import torch
from std_msgs.msg import Bool
from rayfronts.utils import compute_cos_sim
from rayfronts import geometry3d as g3d
import numpy as np
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path

class LvlmBehavior:
    def __init__(self, get_clock, publisher_dict, mapping_server_rosnode):
        self.get_clock = get_clock
        self.name = 'LVLM-guided'
        self.lvlm_trigger_pub = publisher_dict['lvlm_trigger']
        self.guiding_objects = []
        self.last_request_time = None
        self.request_interval = 20.0
        self.mapping_server_rosnode = mapping_server_rosnode

    def set_guiding_objects(self, objects_string: str):
        raw_objects = [o.strip().lower() for o in objects_string.split(",")]

        cleaned_objects = []
        seen = set()
        for obj in raw_objects:
            if obj.startswith("a "):
                obj = obj[2:]
            elif obj.startswith("an "):
                obj = obj[3:]
            elif obj.startswith("the "):
                obj = obj[4:]
            obj = obj.rstrip('.,').strip()
            if obj not in seen and obj != "":
                cleaned_objects.append(obj)
                seen.add(obj)
        self.guiding_objects = cleaned_objects
        print("self.guiding_objects", self.guiding_objects)
        print(f"[LvlmBehavior] Updated objects: {self.guiding_objects}")

        # if cleaned_objects != self.guiding_objects:
        #     mapping_server_rosnode.delete_queries(self.guiding_objects)
        #     self.guiding_objects = cleaned_objects
            
        #     mapping_server_rosnode.add_queries(self.guiding_objects)

    def condition_check(self, queries_labels, target_objects, queries_feats, mapper, publisher_dict, subscriber_dict):
        self.mapping_server_rosnode.add_queries(self.guiding_objects)
        objects_to_delete = [obj for i, obj in enumerate(self.mapping_server_rosnode._queries_labels['text']) if obj not in self.mapping_server_rosnode._target_objects and obj not in self.mapping_server_rosnode._background_objects and obj not in self.guiding_objects]
        self.mapping_server_rosnode.delete_queries(objects_to_delete)

        if self.mapping_server_rosnode._queries_labels is None or self.mapping_server_rosnode._queries_labels['text'] is None:
            return False
        if self.mapping_server_rosnode._target_objects is None:
            return False
        
        now = self.get_clock().now().nanoseconds/1e9
        self.lvlm_trigger_pub.publish(Bool(data=True))
        if self.last_request_time is None or (now - self.last_request_time) > self.request_interval:
            self.lvlm_trigger_pub.publish(Bool(data=True))
            self.last_request_time = now
            print("[LvlmBehavior] Requested LVLM guidance")
        
        if not self.guiding_objects:
            return False
        
        if len(self.guiding_objects) >= 1:
            print("queries_labels['text']", self.mapping_server_rosnode._queries_labels['text'])
            print("self.guiding_objects", self.guiding_objects)
            label_indices = [self.mapping_server_rosnode._queries_labels['text'].index(guiding_object) for guiding_object in self.guiding_objects]
            print("indexes", label_indices)
            ray_feat = mapper.global_rays_feat
            ray_orig_angles = mapper.global_rays_orig_angles
            if ray_feat is not None and ray_orig_angles is not None and ray_feat.shape[0] > 0:
                ray_lang_aligned = mapper.encoder.align_spatial_features_with_language(ray_feat.unsqueeze(-1).unsqueeze(-1))
                if ray_lang_aligned.ndim == 4:
                    ray_lang_aligned = ray_lang_aligned.squeeze(-1).squeeze(-1)
                if ray_lang_aligned.ndim == 2:
                    ray_lang_aligned = ray_lang_aligned
                if ray_lang_aligned.ndim == 1:
                    ray_lang_aligned = ray_lang_aligned.unsqueeze(0)
                
                if queries_feats is not None:
                    ray_scores = compute_cos_sim(queries_feats['text'], ray_lang_aligned, softmax=True)
                    threshold = 0.9
                    mask_any = (ray_scores[:, label_indices] > threshold).any(dim=1)
                    indices = mask_any.nonzero(as_tuple=True)[0]
                    if indices.numel() > 0:
                        self.indices = indices
                        self.ray_orig_angles = ray_orig_angles
                        return True
        
        return False


    def execute(self, mapper, point3d_dict, waypoint_locked, publisher_dict, subscriber_dict):
        path_publisher = publisher_dict['path']
        ray_orig = self.ray_orig_angles[:,:3]
        ray_angles = torch.deg2rad(self.ray_orig_angles[:,3:])
        ray_dir = torch.stack(g3d.spherical_to_cartesian(1,ray_angles[:,0],ray_angles[:,1]),dim=-1)
        fo = ray_orig[self.indices]
        fd = ray_dir[self.indices]
        
        # orig_world = torch.stack([fo[:,2],-fo[:,0],-fo[:,1]],dim=1)
        # dir_world = torch.stack([fd[:,2],-fd[:,0],-fd[:,1]],dim=1)
        # xy_dirs = dir_world[:,:2]
        # xy_dirs_np = xy_dirs.cpu().numpy()
        # xy_dirs_np_normed = xy_dirs_np / np.linalg.norm(xy_dirs_np, axis=1, keepdims=True)

        mean_origin = torch.mean(fo, dim=0)
        mean_direction = torch.mean(fd, dim=0)

        origin_np = mean_origin.cpu().numpy()
        direction_np = mean_direction.cpu().numpy()
        origin = np.array([origin_np[2],-origin_np[0],-origin_np[1]])
        direction = np.array([direction_np[2], -direction_np[0], -direction_np[1]])

        magnitude = 5.0
        path = Path()
        path.header.stamp = self.get_clock().now().to_msg()
        path.header.frame_id = "map"
        unit_dir = direction / np.linalg.norm(direction)
        target = origin + unit_dir * magnitude

        for factor in [0.0, 1.0]:  # 0% and 100% progress along direction
            pose = PoseStamped()
            pose.header.stamp = self.get_clock().now().to_msg()
            pose.header.frame_id = "map"
            pose.pose.position.x = origin[0] * (1 - factor) + target[0] * factor
            pose.pose.position.y = origin[1] * (1 - factor) + target[1] * factor
            pose.pose.position.z = origin[2] * (1 - factor) + target[2] * factor
            pose.pose.orientation.w = 1.0  # No rotation
            path.poses.append(pose)

        path_publisher.publish(path)

        return waypoint_locked, None, None
