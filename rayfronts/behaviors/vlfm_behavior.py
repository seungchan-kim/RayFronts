from rayfronts.utils import compute_cos_sim
from nav_msgs.msg import Path
import torch
from rayfronts import geometry3d as g3d
import numpy as np
from geometry_msgs.msg import PoseStamped
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point

class VlfmBehavior:
    def __init__(self, get_clock):
        self.get_clock = get_clock
        self.name = 'VLFM-based'

    def condition_check(self, queries_labels, target_objects, queries_feats, mapper, publisher_dict, subscriber_dict):
        return True
    
    def execute(self, node, mapper, point3d_dict, waypoint_locked, publisher_dict, subscriber_dict):
        cur_pose_np = point3d_dict['cur_pose']
        target_waypoint1 = point3d_dict['target1']
        target_waypoint2 = point3d_dict['target2']
        if node._queries_labels is not None and node._queries_labels['text'] is not None and len(node._target_objects) > 0:
            indices = [node._queries_labels['text'].index(target_object) for target_object in node._target_objects]
            ray_feat = node.mapper.global_rays_feat
            if ray_feat is not None and ray_feat.shape[0] > 0:
                ray_lang_aligned = node.mapper.encoder.align_spatial_features_with_language(ray_feat.unsqueeze(-1).unsqueeze(-1))
                if ray_lang_aligned.ndim == 4:
                    ray_lang_aligned = ray_lang_aligned.squeeze(-1).squeeze(-1)
                if ray_lang_aligned.ndim == 2:
                    ray_lang_aligned = ray_lang_aligned
                if ray_lang_aligned.ndim == 1:
                    ray_lang_aligned = ray_lang_aligned.unsqueeze(0)
                
                if node._queries_feats is not None:
                    ray_scores = compute_cos_sim(node._queries_feats['text'], ray_lang_aligned, softmax=True)
                    relevant_scores = ray_scores[:,indices]
                    _, flat_idx = torch.max(relevant_scores.view(-1), dim=0)
                    ray_idx = flat_idx // relevant_scores.shape[1]

                    ray_orig = node.mapper.global_rays_orig_angles[:,:3]
                    selected_orig = ray_orig[ray_idx].unsqueeze(0)

                    orig_world = torch.stack([selected_orig[:,2],-selected_orig[:,0],-selected_orig[:,1]],dim=1)

                    path = Path()
                    path.header.stamp = self.get_clock().now().to_msg()
                    path.header.frame_id = 'map'

                    origin = orig_world[0].cpu().numpy()
                    direction = origin - cur_pose_np
                    direction_norm = direction / np.linalg.norm(direction)

                    alpha=0.8
                    mid_pose_np = cur_pose_np * (1-alpha) + origin * alpha
                    mid_pose = PoseStamped()
                    mid_pose.header.stamp = self.get_clock().now().to_msg()
                    mid_pose.header.frame_id = 'map'
                    mid_pose.pose.position.x = float(mid_pose_np[0])
                    mid_pose.pose.position.y = float(mid_pose_np[1])
                    mid_pose.pose.position.z = float(mid_pose_np[2])
                    mid_pose.pose.orientation.w = 1.0
                    path.poses.append(mid_pose)

                    target_waypoint1 = origin

                    t1_pose = PoseStamped()
                    t1_pose.header.stamp = self.get_clock().now().to_msg()
                    t1_pose.header.frame_id = 'map'
                    t1_pose.pose.position.x = float(target_waypoint1[0])
                    t1_pose.pose.position.y = float(target_waypoint1[1])
                    t1_pose.pose.position.z = float(target_waypoint1[2])
                    t1_pose.pose.orientation.w = 1.0
                    path.poses.append(t1_pose)

                    node.path_publisher.publish(path)
        return waypoint_locked, target_waypoint1, target_waypoint2