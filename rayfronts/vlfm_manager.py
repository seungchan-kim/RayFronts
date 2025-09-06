import torch
import time
import numpy as np

from rayfronts.behaviors.voxel_behavior import VoxelBehavior
from rayfronts.behaviors.vlfm_behavior import VlfmBehavior

from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA
import scipy.ndimage

class VlfmManager:
    def __init__(self, get_clock, publisher_dict, node):
        self.behavior_mode = 'Frontier-based'
        self.get_clock = get_clock
        self.voxel_behavior = VoxelBehavior(self.get_clock)
        self.vlfm_behavior = VlfmBehavior(self.get_clock)
        self.behaviors = [self.voxel_behavior, self.vlfm_behavior]

    def mode_select(self, queries_labels, target_objects, queries_feats, mapper, publisher_dict, subscriber_dict):
        for behavior in self.behaviors:
            if behavior.condition_check(queries_labels, target_objects, queries_feats, mapper, publisher_dict, subscriber_dict):
                self.behavior_mode = behavior.name
                return
    
    def behavior_execute(self, node, behavior_mode, mapper, point3d_dict, waypoint_locked, publisher_dict, subscriber_dict):
        if behavior_mode == 'Voxel-based':
            wp_locked, tw1, tw2 = self.voxel_behavior.execute(mapper, point3d_dict, waypoint_locked, publisher_dict, subscriber_dict)
            return wp_locked, tw1, tw2
        elif behavior_mode == 'VLFM-based':
            wp_locked, tw1, tw2 = self.vlfm_behavior.execute(node, mapper, point3d_dict, waypoint_locked, publisher_dict, subscriber_dict)
            return wp_locked, tw1, tw2