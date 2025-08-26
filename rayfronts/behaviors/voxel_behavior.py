from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA
import numpy as np
import torch
import time
import scipy.ndimage
from rayfronts.utils import compute_cos_sim
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
import random

class VoxelBehavior:
    def __init__(self, get_clock):
        self.get_clock = get_clock
        self.name = 'Voxel-based'
        self.target_voxel_clusters = {}
        self.target_objects = None
        self.visited_clusters = []
        self.unvisited_clusters = []
        self.prev_voxel_cluster_ids = 0

    def condition_check(self, queries_labels, target_objects, queries_feats, mapper, publisher_dict, subscriber_dict):
        if queries_labels is None:
            return False
        
        if queries_labels['text'] is None:
            return False
        
        if len(target_objects)==0:
            return False
        
        if queries_labels is not None and queries_labels['text'] is not None and len(target_objects) > 0:
            label_indices = [queries_labels['text'].index(target_object) for target_object in target_objects]
            #label_index = queries_labels['text'].index(target_object)
            vox_xyz = mapper.global_vox_xyz
            vox_feat = mapper.global_vox_feat

            self.target_objects = target_objects

            if vox_feat is not None and vox_feat.shape[0] > 0:
                vox_lang_aligned = mapper.encoder.align_spatial_features_with_language(vox_feat.unsqueeze(-1).unsqueeze(-1))
                if vox_lang_aligned.ndim == 4:
                    vox_lang_aligned = vox_lang_aligned.squeeze(-1).squeeze(-1)
                if vox_lang_aligned.ndim == 2:
                    vox_lang_aligned = vox_lang_aligned
                if vox_lang_aligned.ndim == 1:
                    vox_lang_aligned = vox_lang_aligned.unsqueeze(0)
                
                if queries_feats is not None:
                    vox_scores = compute_cos_sim(queries_feats['text'], vox_lang_aligned, softmax=True)
                    print("vox_scores", torch.round(vox_scores*1000)/1000)
                    threshold = 0.98

                    relevant_scores = vox_scores[:, label_indices]
                    mask = (relevant_scores > threshold).any(dim=1)
                    indices = mask.nonzero(as_tuple=True)[0]
                    #indices = (vox_scores[:,label_index] > threshold).nonzero(as_tuple=True)[0]

                    filtered_vox = vox_xyz[indices]
                    filtered_vox = filtered_vox.round(decimals=3)
                    vox_size=0.5
                    if filtered_vox.numel() > 0:
                        ccl_st = time.time()
                        min_coords = filtered_vox.min(dim=0)[0]
                        norm_coords = ((filtered_vox - min_coords)/vox_size).long()
                        max_coords = norm_coords.max(dim=0)[0] + 1
                        occupancy = np.zeros(tuple(max_coords.tolist()), dtype=np.uint8)
                        for x,y,z in norm_coords:
                            occupancy[x,y,z] = 1
                        structure = np.ones((3,3,3),dtype=np.uint8)
                        labeled, num_components = scipy.ndimage.label(occupancy,structure=structure)
                        ccl_ed = time.time()
                        print("time for ccl: ", ccl_ed - ccl_st, " sec")
                        label_ids = torch.tensor([labeled[x,y,z] for x,y,z in norm_coords])
                        norm_np = norm_coords.cpu().numpy()
                        vox_cluster_count = 0
                        for label_val in range(1, num_components+1):
                            idx = (label_ids == label_val).nonzero(as_tuple=True)[0]
                            if len(idx) < 30:
                                continue
                            coords = norm_np[idx]
                            min_voxel = coords.min(axis=0)
                            max_voxel = coords.max(axis=0)
                            min_world = min_voxel * vox_size + min_coords.cpu().numpy()
                            max_world = (max_voxel+1) * vox_size + min_coords.cpu().numpy()
                            center = (min_world + max_world) / 2
                            size = max_world - min_world
                            cx = center[2]
                            cy = -center[0]
                            cz = -center[1]
                            sx = size[2]
                            sy = size[0]
                            sz = size[1]
                            self.target_voxel_clusters[vox_cluster_count] = [cx,cy,cz,sx,sy,sz]
                            vox_cluster_count += 1
                        
                        all_clusters = self.target_voxel_clusters.items()
                        print("all_clusters: ", all_clusters)

                        print("self.visited_clusters: ", self.visited_clusters)

                        self.unvisited_clusters = [(idx, cluster) for idx, cluster in all_clusters if not self.is_near_visited(np.array(cluster[:3]), np.array(cluster[3:6]), self.visited_clusters)]
                        print("self.unvisited_clusters: ", self.unvisited_clusters)


                        if len(self.unvisited_clusters) > 0:
                            return True                       
                        #if vox_cluster_count > 0:
                        #    return True                
        return False

    def execute(self, mapper, point3d_dict, waypoint_locked, publisher_dict, subscriber_dict):
        voxel_bbox_publisher = publisher_dict['voxel_bbox']
        self.visualize_voxel_cluster_bbox(voxel_bbox_publisher)

        cur_pose_np = point3d_dict['cur_pose']
        target_waypoint1 = point3d_dict['target1']
        target_waypoint2 = point3d_dict['target2']
        path_publisher = publisher_dict['path']

        all_clusters = self.target_voxel_clusters.items()
        print("execute:")
        print("all_clusters: ", all_clusters)

        self.unvisited_clusters = [(idx, cluster) for idx, cluster in all_clusters 
                              if not self.is_near_visited(np.array(cluster[:3]), np.array(cluster[3:6]), self.visited_clusters)]
        print("self.unvisited_clusters: ", self.unvisited_clusters)
        sorted_voxel_clusters_by_dist = sorted(self.unvisited_clusters,
                                               key=lambda item: np.linalg.norm(cur_pose_np - np.array(item[1][:3])))
        print("sorted_voxel_clusteres_by_dist: ", sorted_voxel_clusters_by_dist)

        path = Path()
        path.header.stamp = self.get_clock().now().to_msg()
        path.header.frame_id = 'map'

        for i, (idx, cluster) in enumerate(sorted_voxel_clusters_by_dist):
            center = np.array(cluster[:3])
            sizes = np.array(cluster[3:])
            half_sizes = sizes / 2.0
            dir = center - cur_pose_np
            dir_norm = dir / np.linalg.norm(dir)
            
            ray_origin_local = cur_pose_np - center

            tmin, tmax = -np.inf, np.inf
            for axis in range(3):
                if dir_norm[axis] != 0:
                    t1 = (-half_sizes[axis] - ray_origin_local[axis]) / dir_norm[axis]
                    t2 = ( half_sizes[axis] - ray_origin_local[axis]) / dir_norm[axis]
                    tmin = max(tmin, min(t1,t2))
                    tmax = min(tmax, max(t1,t2))
                else:
                    if abs(ray_origin_local[axis]) > half_sizes[axis]:
                        continue
            if tmax < max(tmin, 0):
                continue
            t_hit = tmin if tmin > 0 else tmax
            surface_point = cur_pose_np + dir_norm * t_hit
            offset = 2.0
            adjacent_np = surface_point - dir_norm * offset

            if i == 0:
                if not waypoint_locked:
                    target_waypoint2 = adjacent_np
                    waypoint_locked = True
                final_adj_pose_np = target_waypoint2
                
                alpha = 0.8
                final_mid_pose_np = cur_pose_np * (1-alpha) + final_adj_pose_np * alpha
                
                target_waypoint1 = final_mid_pose_np
                    
                mid_pose = PoseStamped()
                mid_pose.header.stamp = self.get_clock().now().to_msg()
                mid_pose.header.frame_id = 'map'
                mid_pose.pose.position.x = float(final_mid_pose_np[0])
                mid_pose.pose.position.y = float(final_mid_pose_np[1])
                mid_pose.pose.position.z = float(final_mid_pose_np[2])
                mid_pose.pose.orientation.w = 1.0
                path.poses.append(mid_pose)

                adj_pose = PoseStamped()
                adj_pose.header.stamp = self.get_clock().now().to_msg()
                adj_pose.header.frame_id = 'map'
                adj_pose.pose.position.x = float(final_adj_pose_np[0])
                adj_pose.pose.position.y = float(final_adj_pose_np[1])
                adj_pose.pose.position.z = float(final_adj_pose_np[2])
                adj_pose.pose.orientation.w = 1.0
                path.poses.append(adj_pose)

        path_publisher.publish(path)

        print("cur_pose_np - target_waypoint2 distance: ", np.linalg.norm(cur_pose_np - target_waypoint2))
        # print("min distance to cuboid", self.min_distance_to_cuboid(self.current_target_cluster, cur_pose_np))
        # if self.min_distance_to_cuboid(self.current_target_cluster, cur_pose_np) < 3.0:
        #     self.visited_clusters.append(self.current_target_cluster)
        #     waypoint_locked = False
        
        #if random.random() < 0.2:
        #    waypoint_locked = False
        
        if np.linalg.norm(cur_pose_np - target_waypoint2) < 5.0:
            if sorted_voxel_clusters_by_dist:
                current_close_cluster = np.array(sorted_voxel_clusters_by_dist[0][1])
                self.visited_clusters.append(current_close_cluster)
            waypoint_locked = False
        
        return waypoint_locked, target_waypoint1, target_waypoint2

    def visualize_voxel_cluster_bbox(self, voxel_bbox_publisher):
        self.clear_voxel_clusters(voxel_bbox_publisher)
        marker_array = MarkerArray()
        now = self.get_clock().now().to_msg()
        #for cluster_id, center_size_list in self.target_voxel_clusters[target_object].items():
        j = 0
        for cluster_id, center_size_list in self.unvisited_clusters:
            cx = center_size_list[0]
            cy = center_size_list[1]
            cz = center_size_list[2]
            sx = center_size_list[3]
            sy = center_size_list[4]
            sz = center_size_list[5]
            print("cx,cy,cz,sx,sy,sz", cx,cy,cz,sx,sy,sz)
            marker = Marker()
            marker.header.frame_id = 'map'
            marker.header.stamp = now
            marker.ns = 'ccl_boxes'
            marker.id = j
            marker.type = Marker.CUBE
            marker.action = Marker.ADD
            marker.pose.position.x = cx
            marker.pose.position.y = cy
            marker.pose.position.z = cz
            marker.scale.x = sx
            marker.scale.y = sy
            marker.scale.z = sz
            marker.color = ColorRGBA(r=0.0, g=1.0, b=0.0, a=0.2)
            marker.lifetime.sec = 1
            marker_array.markers.append(marker)
            j += 1
        self.prev_voxel_cluster_ids = j
        voxel_bbox_publisher.publish(marker_array)
    
    def is_near_visited(self, center, size, visited_clusters, threshold=10.0):
        return any(self.cuboid_distance(center, size, np.array(visited[:3]), np.array(visited[3:6])) < threshold for visited in visited_clusters)

    def cuboid_distance(self, center_a, size_a, center_b, size_b):
        half_a = size_a / 2.0
        half_b = size_b / 2.0
        dx = max(abs(center_a[0] - center_b[0]) - (half_a[0] + half_b[0]), 0)
        dy = max(abs(center_a[1] - center_b[1]) - (half_a[1] + half_b[1]), 0)
        dz = max(abs(center_a[2] - center_b[2]) - (half_a[2] + half_b[2]), 0)
        return np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
    
    def clear_voxel_clusters(self, voxel_bbox_publisher):
        if self.prev_voxel_cluster_ids > 0:
            clear_marker_array = MarkerArray()
            for i in range(self.prev_voxel_cluster_ids):
                clear_marker = Marker()
                clear_marker.header.frame_id = 'map'
                clear_marker.header.stamp = self.get_clock().now().to_msg()
                clear_marker.ns = 'ccl_boxes'
                clear_marker.id = i
                clear_marker.action = Marker.DELETE
                clear_marker_array.markers.append(clear_marker)
            voxel_bbox_publisher.publish(clear_marker_array)

    def min_distance_to_cuboid(self, cluster, cur_pose_np):
        cx,cy,cz, sx,sy,sz = cluster
        px, py, pz = cur_pose_np

        min_x, max_x = cx - sx / 2.0, cx + sx / 2.0
        min_y, max_y = cy - sy / 2.0, cy + sy / 2.0
        min_z, max_z = cz - sz / 2.0, cz + sz / 2.0

        closest_x = np.clip(px, min_x, max_x)
        closest_y = np.clip(py, min_y, max_y)
        closest_z = np.clip(pz, min_z, max_z)

        return np.linalg.norm([px - closest_x, py - closest_y, pz - closest_z])

