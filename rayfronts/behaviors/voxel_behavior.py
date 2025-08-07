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
                            vox_cluster_count += 1
                        
                        if vox_cluster_count > 0:
                            return True
                        
        return False

    def execute(self):
        pass

      #IF query is none: frontier-based
      # if self._queries_labels is None:
      #   self.behavior_mode = 'Frontier-based'
      # elif self._queries_labels['text'] is None:
      #   self.behavior_mode = 'Frontier-based'
      # elif self._target_object is None:
      #   self.behavior_mode = 'Frontier-based'
      # elif self._queries_labels['text'] is not None and self._target_object is not None:

      #   label_index = self._queries_labels['text'].index(self._target_object)

      #   #check voxels and if there are voxel-groups that match
      #   vox_xyz = self.mapper.global_vox_xyz
      #   vox_feat = self.mapper.global_vox_feat

      #   if vox_feat is not None and vox_feat.shape[0] > 0:
      #     vox_lang_aligned = self.mapper.encoder.align_spatial_features_with_language(vox_feat.unsqueeze(-1).unsqueeze(-1))
      #     if vox_lang_aligned.ndim == 4:
      #       vox_lang_aligned = vox_lang_aligned.squeeze(-1).squeeze(-1)
      #     if vox_lang_aligned.ndim == 2:
      #       vox_lang_aligned = vox_lang_aligned
      #     if vox_lang_aligned.ndim == 1:
      #       vox_lang_aligned = vox_lang_aligned.unsqueeze(0)

      #     if self._queries_feats is not None:
      #       vox_scores = compute_cos_sim(self._queries_feats['text'], vox_lang_aligned, softmax=True)
      #       print("vox_scores", torch.round(vox_scores*1000)/1000)
      #       threshold = 0.95
      #       indices = (vox_scores[:,label_index] > threshold).nonzero(as_tuple=True)[0]

      #       filtered_vox = vox_xyz[indices]
      #       filtered_vox = filtered_vox.round(decimals=3)
      #       vox_size = 0.5
      #       if filtered_vox.numel() > 0:
      #         ccl_st = time.time()
      #         min_coords = filtered_vox.min(dim=0)[0]
      #         norm_coords = ((filtered_vox - min_coords)/vox_size).long()
      #         max_coords = norm_coords.max(dim=0)[0] + 1
      #         occupancy = np.zeros(tuple(max_coords.tolist()), dtype=np.uint8)
      #         for x,y,z in norm_coords:
      #           occupancy[x,y,z] = 1
      #         structure = np.ones((3,3,3),dtype=np.uint8)
      #         labeled, num_components = scipy.ndimage.label(occupancy,structure=structure)
      #         ccl_ed = time.time()
      #         print("time for ccl: ", ccl_ed - ccl_st, " sec")

      #         label_ids = torch.tensor([labeled[x,y,z] for x,y,z in norm_coords])
      #         norm_np = norm_coords.cpu().numpy()
      #         marker_array = MarkerArray()
      #         now = self.get_clock().now().to_msg()

      #         marker_id = 0
      #         for label_val in range(1, num_components+1):
      #           idx = (label_ids == label_val).nonzero(as_tuple=True)[0]
      #           if len(idx) < 30:
      #             continue

      #           coords = norm_np[idx]
      #           min_voxel = coords.min(axis=0)
      #           max_voxel = coords.max(axis=0)

      #           min_world = min_voxel * vox_size + min_coords.cpu().numpy()
      #           max_world = (max_voxel+1)*vox_size + min_coords.cpu().numpy()
      #           center = (min_world + max_world) / 2
      #           size = max_world - min_world

      #           marker = Marker()
      #           marker.header.frame_id = 'map'
      #           marker.header.stamp = now
      #           marker.ns = 'ccl_boxes'
      #           marker.id = marker_id
      #           marker.type = Marker.CUBE
      #           marker.action = Marker.ADD
      #           marker.pose.position.x = center[2]
      #           marker.pose.position.y = -center[0]
      #           marker.pose.position.z = -center[1]
      #           marker.scale.x = size[2]
      #           marker.scale.y = -size[0]
      #           marker.scale.z = -size[1]
      #           marker.color = ColorRGBA(r=0.0, g=1.0, b=0.0, a=0.2)
      #           marker.lifetime.sec = 1
      #           marker_array.markers.append(marker)
      #           marker_id += 1

      #         if marker_id > 0:
      #           self.behavior_mode = 'Voxel-based'
      #         self.voxel_bbox_publisher.publish(marker_array)
      