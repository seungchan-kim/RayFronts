from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA
import numpy as np
import torch
import time
import scipy.ndimage
from rayfronts.utils import compute_cos_sim
from nav_msgs.msg import Path

class VoxelBehavior:
    def __init__(self, get_clock):
        self.get_clock = get_clock
        self.name = 'Voxel-based'
        self.target_voxel_clusters = {}
        self.target_object = None

    def condition_check(self, queries_labels, target_object, queries_feats, mapper):
        if queries_labels is None:
            return False
        
        if queries_labels['text'] is None:
            return False
        
        if target_object is None:
            return False
        
        if queries_labels is not None and queries_labels['text'] is not None and target_object is not None:
            label_index = queries_labels['text'].index(target_object)
            vox_xyz = mapper.global_vox_xyz
            vox_feat = mapper.global_vox_feat

            self.target_object = target_object
            self.target_voxel_clusters[target_object] = {}

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
                    threshold = 0.95
                    indices = (vox_scores[:,label_index] > threshold).nonzero(as_tuple=True)[0]

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
                            sy = -size[0]
                            sz = -size[1]

                            self.target_voxel_clusters[target_object][vox_cluster_count] = [cx,cy,cz,sx,sy,sz]
                            vox_cluster_count += 1
                                                
                        if vox_cluster_count > 0:
                            return True
                        
        return False

    def execute(self, mapper, point3d_dict, waypoint_locked, publisher_dict):
        voxel_bbox_publisher = publisher_dict['voxel_bbox']
        self.visualize_voxel_cluster_bbox(self.target_object, voxel_bbox_publisher)
        return waypoint_locked, point3d_dict['target1'], point3d_dict['target2']

    def visualize_voxel_cluster_bbox(self, target_object, voxel_bbox_publisher):
        marker_array = MarkerArray()
        now = self.get_clock().now().to_msg()
        for cluster_id, center_size_list in self.target_voxel_clusters[target_object].items():
            cx = center_size_list[0]
            cy = center_size_list[1]
            cz = center_size_list[2]
            sx = center_size_list[3]
            sy = center_size_list[4]
            sz = center_size_list[5]
            marker = Marker()
            marker.header.frame_id = 'map'
            marker.header.stamp = now
            marker.ns = 'ccl_boxes'
            marker.id = cluster_id
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
        voxel_bbox_publisher.publish(marker_array)
