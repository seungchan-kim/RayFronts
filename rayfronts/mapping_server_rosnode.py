"""Server to run and visualize semantic mapping from a posed RGBD data source

The map server can be queried with images or text with a messaging_service
online or with a predefined query file.

Loads the configs/default.yaml as the root config but all options can be
overwridden by other configs or through the command line. Check hydra-configs
for more details.

"""

import random
import os
import logging
import time
import atexit
import inspect
import threading
import signal
from enum import Enum
from functools import partial
from typing_extensions import List
import json

import torch
import torchvision
import numpy as np
import hydra
import struct

from rayfronts import datasets, visualizers, image_encoders, mapping, utils

import rclpy
from rclpy.node import Node
import std_msgs.msg
from std_msgs.msg import String
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
import scipy.ndimage
from sensor_msgs.msg import PointCloud2, PointField, Image
from std_msgs.msg import Header, ColorRGBA
from sensor_msgs_py import point_cloud2
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
from rayfronts import geometry3d as g3d
from rayfronts.utils import compute_cos_sim
from sklearn.cluster import DBSCAN

logger = logging.getLogger(__name__)

class MappingServer(Node):
  """Server performing mapping on a stream of posed RGBD data. Can be queried.
  
  Attributes:
    status: Status enum signaling the current state of the server.
    cfg: Stores the mapping system configuration as is.
    dataset: Stores the dataset/datasource object.
    vis: Stores the visualizer used. May be None.
    encoder: Stores the encoder model used by the mapper.
    mapper: Stores the mapper used.
    messaging_service: Stores the messaging service used for online querying.
  """

  class Status(Enum):
    INIT = 0 # Server is initializing
    MAPPING = 1 # Server is actively mapping
    IDLE = 2 # Server has stopped mapping and awaits any new queries.
    CLOSING = 3 # Server is in the process of shutting down.
    CLOSED = 4 # Server has shutdown.

  @torch.no_grad()
  def __init__(self, cfg):
    super().__init__('mapping_server')
    self.status = MappingServer.Status.INIT
    self._status_lock = threading.RLock()

    self.cfg = cfg
    self.dataset: datasets.PosedRgbdDataset = \
      hydra.utils.instantiate(cfg.dataset)

    self.path_publisher = self.create_publisher(Path, '/robot_1/global_plan', 10)
    #self.pc2_publisher = self.create_publisher(PointCloud2, '/colored_pointcloud', 10)
    #self.rays_publisher = self.create_publisher(MarkerArray, '/rays', 10)
    self.filtered_rays_publisher = self.create_publisher(MarkerArray, 'filtered_rays', 10)

    self.mode_text_publisher = self.create_publisher(Marker, '/mode_text', 10)

    self.viewpoint_pub = self.create_publisher(PointCloud2, "/frontier_viewpoints", 10)

    self.waypoint_locked = False

    self.target_waypoint = None
    self.target_waypoint2 = None

    self.behavior_mode = 'Frontier-based' #Frontier-based, Ray-based

    self.prev_filtered_marker_ids = 0

    self._target_object = None
    self.create_subscription(String, '/input_text', self.target_object_callback, 10)

    intrinsics_3x3 = self.dataset.intrinsics_3x3
    if "vox_size" in cfg.mapping:
      base_point_size = cfg.mapping.vox_size / 2
    else:
      base_point_size = None

    self.vis: visualizers.Mapping3DVisualizer = None
    if "vis" in cfg and cfg.vis is not None:
      self.vis = hydra.utils.instantiate(cfg.vis, intrinsics_3x3=intrinsics_3x3,
                                         base_point_size=base_point_size)

    # Ugly way to check if the chosen mapper constructor needs an encoder.
    c = getattr(mapping, cfg.mapping._target_.split(".")[-1])
    init_encoder = "encoder" in inspect.signature(c.__init__).parameters.keys()
    init_encoder = init_encoder and "encoder" in cfg
    mapper_kwargs = dict()

    self.encoder: image_encoders.ImageEncoder = None
    self.feat_compressor = None
    if self.cfg.mapping.feat_compressor is not None:
      self.feat_compressor = hydra.utils.instantiate(
        self.cfg.mapping.feat_compressor)

    if init_encoder:
      encoder_kwargs = dict()
      if (cfg.querying.text_query_mode is not None and
          "RadioEncoder" in cfg.encoder and cfg.encoder.lang_model is None):
        raise ValueError("Radio encoder must have a language model if text "
                        "querying is enabled.")
      if "NARadioEncoder" in cfg.encoder._target_:
        encoder_kwargs["input_resolution"] = [self.dataset.rgb_h,
                                              self.dataset.rgb_w]

      self.encoder = hydra.utils.instantiate(cfg.encoder, **encoder_kwargs)
      mapper_kwargs["encoder"] = self.encoder
      mapper_kwargs["feat_compressor"] = self.feat_compressor

    self.mapper: mapping.RGBDMapping = hydra.utils.instantiate(
      cfg.mapping, intrinsics_3x3=intrinsics_3x3, visualizer=self.vis,
      **mapper_kwargs)

    # Dictionary mapping a label group name to a list of string labels.
    # In the case of a text query, the label is the query. In case of image
    # querying, the label is the image file name.
    self._queries_labels = None

    # Dictionary mapping a label group name to an NxD torch tensor where
    # N corresponds to the number of queries in that group and D is the
    # feature dimension. N must equal len(self.queries_labels[k]) for a group.
    self._queries_feats = None

    # History is used when mapper is idling such that only new queries are
    # visualized as opposed to visualizing the full query set everytime.
    # This also depends on compute_prob value.
    self._queries_labels_history = set()

    # Flag to track if the set of queries has been updated or not.
    self._queries_updated = False

    # Color map mapping query label to color.
    self._query_cmap = dict()

    self._query_lock = threading.RLock()

    if cfg.querying.query_file is not None:
      with open(cfg.querying.query_file, "r", encoding="UTF-8") as f:
        if cfg.querying.query_file.endswith(".json"):
          cmap_queries = json.load(f)
          queries = list(cmap_queries.keys())
          self._query_cmap = {k: utils.hex_to_rgb(v) for
                              k, v in cmap_queries.items()}
        else:
          queries = [l.strip() for l in f.readlines()]
        self.add_queries(queries)

    self.messaging_service = None
    if "messaging_service" in cfg and cfg.messaging_service is not None:
      self.messaging_service = hydra.utils.instantiate(
        cfg.messaging_service,
        text_query_callback = self.add_queries if init_encoder else None)

  @torch.no_grad()
  def add_queries(self, queries: List[str]):
    """Adds a list of queries to query the map with at fixed intervals.
    
    Args:
      queries: List of string where each string is either a text query or a 
        path to a local image file for image querying.
    """
    if self.encoder is None or not hasattr(self.encoder, "encode_labels"):
      raise Exception("Trying to query without a capable text encoder.")

    if isinstance(queries, str):
      queries = [queries]

    with self._query_lock:
      queries = set(queries).difference(self._queries_labels_history)

      queries = list(queries)
      if len(queries) == 0:
        return

      self._queries_labels_history.update(queries)

    logger.info("Received queries: %s", str(queries))
    img_queries = [x for x in queries if os.path.exists(x)]
    text_queries = [x for x in queries if not os.path.exists(x)]
    text_queries_feats = None
    if len(text_queries) > 0:
      if self.cfg.querying.text_query_mode == "labels":
        text_queries_feats = self.encoder.encode_labels(text_queries)
      elif self.cfg.querying.text_query_mode == "prompts":
        text_queries_feats = self.encoder.encode_prompts(text_queries)
      else:
        raise ValueError("Invalid query type")

    img_queries_feats = None
    if len(img_queries) > 0:
      imgs = list()
      for q in img_queries:
        imgs.append(torch.nn.functional.interpolate(
          torchvision.io.read_image(q).unsqueeze(0).float().cuda()/255,
          size=(self.dataset.rgb_h, self.dataset.rgb_w),
          mode="bilinear", antialias=True))
      imgs = torch.cat(imgs, dim=0)
      img_queries_feats = self.encoder.align_global_features_with_language(
        self.encoder.encode_image_to_vector(imgs))

    queries_labels = dict(text=text_queries, img=img_queries)
    queries_feats = dict(text=text_queries_feats, img=img_queries_feats)
    if (self.feat_compressor is not None and
        self.cfg.querying.compressed):
      if not self.feat_compressor.is_fitted():
        logger.warning("The feature compressor was not fitted. "
                       "Will try to fit to query features which may fail.")
        l = [x for x in queries_feats.values() if x is not None]
        self.feat_compressor.fit(torch.cat(l, dim=0))
      for k,v in queries_feats.items():
        if v is None:
          continue
        queries_feats[k] = self.feat_compressor.compress(v)

    with self._query_lock:
      if self._queries_feats is None:
        self._queries_labels = queries_labels
        self._queries_feats = queries_feats
      else:
        for k, v in queries_feats.items():
          if v is None:
            continue
          if k not in self._queries_feats:
            self._queries_feats[k] = v
            self._queries_labels[k] = queries_labels
          else:
            self._queries_feats[k] = torch.concat(
              (self._queries_feats[k], queries_feats[k]), dim=0)
            self._queries_labels[k].extend(queries_labels[k])

      self._queries_updated = True

  def clear_queries(self):
    with self._query_lock:
      self._queries_labels = None
      self._queries_feats = None
      self._queries_updated = False

  def run_queries(self):
    with self._query_lock:
      if (self._queries_feats is not None and len(self._queries_feats) > 0):
        kwargs = dict()
        if self._query_cmap is not None and len(self._query_cmap) > 0:
          kwargs["vis_colors"] = self._query_cmap

        for k,v in self._queries_labels.items():
          if v is None or len(v) < 1:
            continue
          r = self.mapper.feature_query(
            self._queries_feats[k], softmax=self.cfg.querying.compute_prob,
            compressed=self.cfg.querying.compressed)
          if self.vis is not None and r is not None:
            self.mapper.vis_query_result(r, vis_labels=v, **kwargs)

        self._queries_updated = False
      with self._status_lock:
        if (self.status == MappingServer.Status.IDLE and
           not self.cfg.querying.compute_prob):
          # No need to relog old queries so we clear them.
          self._queries_feats = None
          self._queries_labels.clear()

  @torch.no_grad()
  def run(self):
    total_wall_t0 = time.time()
    total_map = 0
    total_frames_processed = 0
    wall_t0 = time.time()

    dataloader = list()
    with self._status_lock:
      if self.status == MappingServer.Status.INIT:
        self.status = MappingServer.Status.MAPPING
        dataloader = torch.utils.data.DataLoader(
          self.dataset, batch_size = self.cfg.batch_size)
        logger.info("Datastream opened. Starting mapping.")

    for i, batch in enumerate(dataloader):
      if batch is None:
        break
      rgb_img = batch["rgb_img"].cuda()
      depth_img = batch["depth_img"].cuda()
      pose_4x4 = batch["pose_4x4"].cuda()

      pose_4x4_np = pose_4x4.cpu().numpy()
      cur_pose_np = np.array([float(pose_4x4_np[0][2, 3]),
                              float(-pose_4x4_np[0][0, 3]),
                              float(-pose_4x4_np[0][1, 3])])


      kwargs = dict()
      if "confidence_map" in batch.keys():
        kwargs["conf_map"] = batch["confidence_map"].cuda()

      if self.cfg.depth_limit >= 0:
        depth_img[torch.logical_and(
          torch.isfinite(depth_img),
          depth_img > self.cfg.depth_limit)] = torch.inf

      # Visualize inputs
      if self.vis is not None:
        if i % self.cfg.vis.pose_period == 0:
          self.vis.log_pose(batch["pose_4x4"][-1])
        if i % self.cfg.vis.input_period == 0:
          self.vis.log_img(batch["rgb_img"][-1].permute(1,2,0))
          self.vis.log_depth_img(depth_img.cpu()[-1].squeeze())

      map_t0 = time.time()
      r = self.mapper.process_posed_rgbd(rgb_img, depth_img, pose_4x4, **kwargs)
      map_t1 = time.time()

      #IF query is none: frontier-based
      if self._queries_labels is None:
        self.behavior_mode = 'Frontier-based'
      elif self._queries_labels['text'] is None:
        self.behavior_mode = 'Frontier-based'
      elif self._target_object is None:
        self.behavior_mode = 'Frontier-based'
      elif self._queries_labels['text'] is not None and self._target_object is not None:

        ray_feat = self.mapper.global_rays_feat
        ray_orig_angles = self.mapper.global_rays_orig_angles
        
        label_index = self._queries_labels['text'].index(self._target_object)

        if ray_feat is not None and ray_orig_angles is not None and ray_feat.shape[0] > 0:
          ray_lang_aligned = self.mapper.encoder.align_spatial_features_with_language(self.mapper.global_rays_feat.unsqueeze(-1).unsqueeze(-1))
          if ray_lang_aligned.ndim == 4:
            ray_lang_aligned = ray_lang_aligned.squeeze(-1).squeeze(-1)
          if ray_lang_aligned.ndim == 2:
            ray_lang_aligned = ray_lang_aligned
          if ray_lang_aligned.ndim == 1:
            ray_lang_aligned = ray_lang_aligned.unsqueeze(0)

          if self._queries_feats is not None:
            ray_scores = compute_cos_sim(self._queries_feats['text'], ray_lang_aligned, softmax=True)
            print("ray_scores", torch.round(ray_scores*1000)/1000)
            threshold = 0.95
            indices = (ray_scores[:,label_index] > threshold).nonzero(as_tuple=True)[0]
            print("indices", indices)

            if indices.numel() > 0:
              self.behavior_mode = 'Ray-based'
      
      mode_text_marker = Marker()
      mode_text_marker.header.frame_id = "map"
      mode_text_marker.header.stamp = self.get_clock().now().to_msg()
      mode_text_marker.ns = "mode_text"
      mode_text_marker.id = 0
      mode_text_marker.type = Marker.TEXT_VIEW_FACING
      mode_text_marker.action = Marker.ADD
      mode_text_marker.pose.position.x = cur_pose_np[0]
      mode_text_marker.pose.position.y = cur_pose_np[1]
      mode_text_marker.pose.position.z = cur_pose_np[2] + 10
      mode_text_marker.pose.orientation.w = 1.0
      mode_text_marker.scale.z = 2.0 # Text height in meters
      mode_text_marker.color = ColorRGBA(r=1.0, g=1.0, b=1.0, a=1.0)

      if self._target_object is None:
        mode_text_marker.text = "No Target Object" + "\nExploration Mode: Frontier-based"
        #clear the filtered rays (visualization)
        self.clear_filtered_rays()
      else:
        mode_text_marker.text = "Target Object: " + self._target_object
        if self.behavior_mode == 'Frontier-based':
          mode_text_marker.text += "\nDidn't find any rays"
          #clear the filtered rays (visualization)
          self.clear_filtered_rays()
          mode_text_marker.text += "\nExploration Mode: Frontier-based"
        elif self.behavior_mode == 'Ray-based':
          mode_text_marker.text += "\nDetected Rays"
          mode_text_marker.text += "\nExploration Mode: Ray-based"
      mode_text_marker.lifetime.sec = 0

      self.mode_text_publisher.publish(mode_text_marker)

      if self.behavior_mode == 'Frontier-based':
        if self.mapper.frontiers is not None:
          transformed_frontiers = torch.stack([
            self.mapper.frontiers[:,2],
            -self.mapper.frontiers[:,0],
            -self.mapper.frontiers[:,1]
          ],dim=1)

          #filter out frontier points that are <= 1.5m
          transformed_frontiers = transformed_frontiers[transformed_frontiers[:, 2] > 1.5]

          ######fixed-grid-based chunking#######
          #chunk_size = 10
          #chunk_indices = (transformed_frontiers / chunk_size).floor().to(torch.int32)
          #unique_chunks, inverse_indices = torch.unique(chunk_indices, dim=0, return_inverse=True)
          #viewpoints = []
          #for i in range(len(unique_chunks)):
          #  mask = (inverse_indices == i)
          #  cluster_points = transformed_frontiers[mask]
          #  centroid = cluster_points.mean(dim=0)
          #  if centroid[2] > 2.0:
          #    viewpoints.append(centroid)
          #viewpoints = torch.stack(viewpoints)
          ###########################

          #####DBSCAN clustering of frontiers#####
          frontiers_cpu = transformed_frontiers.detach().cpu().numpy()
          clustering = DBSCAN(eps=2.7, min_samples=3).fit(frontiers_cpu)
          labels = clustering.labels_
          unique_labels = [l for l in set(labels) if l != -1]
          viewpoints = []
          for l in unique_labels:
            cluster_pts = frontiers_cpu[labels==l]
            centroid = cluster_pts.mean(axis=0)
            centroid_torch = torch.from_numpy(centroid)
            centroid_torch = centroid_torch.to(transformed_frontiers.device, dtype=transformed_frontiers.dtype)
            if centroid_torch[2] > 2.0:
              viewpoints.append(centroid_torch)
          viewpoints = torch.stack(viewpoints)
          ###########################

          cent_msg = self.create_pointcloud2_msg(viewpoints)
          self.viewpoint_pub.publish(cent_msg)


          robot_pos_torch = torch.tensor(cur_pose_np, dtype=viewpoints.dtype, device=viewpoints.device)
          distances = torch.norm(viewpoints - robot_pos_torch, dim=1) #distances
          #fill here
          if self.target_waypoint is not None:
            target_waypoint_tensor = torch.tensor(self.target_waypoint, device=viewpoints.device, dtype=viewpoints.dtype)
            cur_motion_vec = target_waypoint_tensor - robot_pos_torch
            cur_motion_vec = cur_motion_vec / (torch.norm(cur_motion_vec) + 1e-6)
            candidate_vecs = viewpoints - robot_pos_torch
            candidate_vecs = candidate_vecs / (torch.norm(candidate_vecs, dim=1, keepdim=True) + 1e-6)
            cos_sim = torch.matmul(candidate_vecs, cur_motion_vec)
            momentum_weight = 5
            scores = distances + momentum_weight * (1.0 - cos_sim)
          else:
            scores = distances
          best_idx = torch.argsort(scores)[0]
          best_cent = viewpoints[best_idx]


          path = Path()
          path.header.stamp = self.get_clock().now().to_msg()
          path.header.frame_id = 'map'

          if not self.waypoint_locked:          
            best_cent_np = best_cent.cpu().numpy()
            self.target_waypoint = best_cent_np
            dir = self.target_waypoint - cur_pose_np
            dir = dir / np.linalg.norm(self.target_waypoint - cur_pose_np)
            self.target_waypoint2 = self.target_waypoint + 2.0*dir
            print("Frontier Target Waypoint is Updated.")
            self.waypoint_locked = True
          
          # mid_pose_np = (cur_pose_np + self.target_waypoint) / 2.0
          
          # mid_pose = PoseStamped()
          # mid_pose.header.stamp = self.get_clock().now().to_msg()
          # mid_pose.header.frame_id = 'map'
          # mid_pose.pose.position.x = mid_pose_np[0]
          # mid_pose.pose.position.y = mid_pose_np[1]
          # mid_pose.pose.position.z = mid_pose_np[2]
          # mid_pose.pose.orientation.w = 1.0
          # path.poses.append(mid_pose)
          
          target_pose = PoseStamped()
          target_pose.header.stamp = self.get_clock().now().to_msg()
          target_pose.header.frame_id = 'map'        
          target_pose.pose.position.x = float(self.target_waypoint[0])
          target_pose.pose.position.y = float(self.target_waypoint[1])
          target_pose.pose.position.z = float(self.target_waypoint[2])
          target_pose.pose.orientation.w = 1.0
          path.poses.append(target_pose)

          target_pose2 = PoseStamped()
          target_pose2.header.stamp = self.get_clock().now().to_msg()
          target_pose2.header.frame_id = 'map'        
          target_pose2.pose.position.x = float(self.target_waypoint2[0])
          target_pose2.pose.position.y = float(self.target_waypoint2[1])
          target_pose2.pose.position.z = float(self.target_waypoint2[2])
          target_pose2.pose.orientation.w = 1.0
          path.poses.append(target_pose2)

          self.path_publisher.publish(path)

          if np.linalg.norm(cur_pose_np - self.target_waypoint) < 5.0:
            print("Robot went to the waypoint close to 5m, and free the lock.")
            self.waypoint_locked = False
            

      elif self.behavior_mode == 'Ray-based':
        ray_feat = self.mapper.global_rays_feat
        ray_orig_angles = self.mapper.global_rays_orig_angles
        
        label_index = self._queries_labels['text'].index(self._target_object)

        if ray_feat is not None and ray_orig_angles is not None and ray_feat.shape[0] > 0:
          #print("ray_feat", ray_feat.shape)
          ray_lang_aligned = self.mapper.encoder.align_spatial_features_with_language(self.mapper.global_rays_feat.unsqueeze(-1).unsqueeze(-1))
          if ray_lang_aligned.ndim == 4:
            ray_lang_aligned = ray_lang_aligned.squeeze(-1).squeeze(-1)
          if ray_lang_aligned.ndim == 2:
            ray_lang_aligned = ray_lang_aligned
          if ray_lang_aligned.ndim == 1:
            ray_lang_aligned = ray_lang_aligned.unsqueeze(0)
          #print("ray_lang_aligned", ray_lang_aligned.shape)
          
          ray_orig = ray_orig_angles[:,:3]
          #print("ray_orig", ray_orig)
          ray_angles = torch.deg2rad(ray_orig_angles[:,3:])
          #print("ray_angles", ray_angles)
          ray_dir = torch.stack(g3d.spherical_to_cartesian(1,ray_angles[:,0],ray_angles[:,1]),dim=-1)
          #print("ray_dir", ray_dir)


          if self._queries_feats is not None:
            #print("self._queries_labels", self._queries_labels)
            #print("self._queries_feats", self._queries_feats)
            ray_scores = compute_cos_sim(self._queries_feats['text'], ray_lang_aligned, softmax=True)
            #print("ray_lang_aligned", ray_lang_aligned.shape)
            #print("ray_scores", ray_scores.shape)
            print("ray_scores", torch.round(ray_scores*1000)/1000)
            #_, indices = torch.topk(ray_scores, k=1, dim=0)
            #indices = torch.nonzero(ray_scores>0.05, as_tuple=True)[0]
            threshold = 0.95
            indices = (ray_scores[:,label_index] > threshold).nonzero(as_tuple=True)[0]
            print("indices", indices)

            if indices.numel() > 0:
              filtered_origins = ray_orig[indices]
              filtered_directions = ray_dir[indices]
              fo = filtered_origins
              fd = filtered_directions
              orig_world = torch.stack([fo[:,2],-fo[:,0],-fo[:,1]],dim=1)
              dir_world = torch.stack([fd[:,2],-fd[:,0],-fd[:,1]],dim=1)
              xy_dirs = dir_world[:,:2]

              xy_dirs_np = xy_dirs.cpu().numpy()

              xy_dirs_np_normed = xy_dirs_np / np.linalg.norm(xy_dirs_np, axis=1, keepdims=True)
              angle_groups = []

              angle_threshold_cos = np.cos(np.deg2rad(45))

              for i, xy_dir in enumerate(xy_dirs_np_normed):
                assigned = False
                for group in angle_groups:
                  dot = np.dot(xy_dir, group['centroid'])
                  if dot >= angle_threshold_cos:
                    group['indices'].append(i)
                    group['rays'].append(xy_dir)
                    group['centroid'] = np.mean(group['rays'],axis=0)
                    group['centroid'] /= np.linalg.norm(group['centroid'])
                    assigned = True
                    break
                if not assigned:
                  angle_groups.append({
                    'centroid': xy_dir, 
                    'rays':[xy_dir],
                    'indices':[i]
                    })
              
              MIN_RAYS_PER_GROUP = 2
              angle_groups = [g for g in angle_groups if len(g['rays']) >= MIN_RAYS_PER_GROUP]

              #mean_origin = torch.mean(filtered_origins, dim=0)
              #mean_direction = torch.mean(filtered_directions, dim=0)
              
              group_averages = []
              for group in angle_groups:
                group_idx = group['indices']
                group_origins = orig_world[group_idx]
                group_directions = dir_world[group_idx]

                avg_origin = group_origins.mean(dim=0)
                avg_direction = group_directions.mean(dim=0)
                avg_direction = avg_direction / avg_direction.norm()

                #group_size = len(group_idx)

                group_averages.append((avg_origin, avg_direction))
              
              #sort the group_averages by the distance from the current_pose to the origin
              group_averages = sorted(group_averages, key=lambda pair: np.linalg.norm(pair[0].cpu().numpy() - cur_pose_np))
              #group_averages = sorted(group_averages, key=lambda x: x[2], reverse=True)


              #origin_np = mean_origin.cpu().numpy()
              #direction_np = mean_direction.cpu().numpy()

              #origin = np.array([origin_np[2],-origin_np[0],-origin_np[1]])
              #direction = np.array([direction_np[2], -direction_np[0], -direction_np[1]])

              magnitude = 2.0

              path = Path()
              path.header.stamp = self.get_clock().now().to_msg()
              path.header.frame_id = "map"
              
              #best ray-groups only
              # best_avg_origin, best_avg_direction, _ = group_averages[0]
              # origin_np = best_avg_origin.cpu().numpy()
              # direction_np = best_avg_direction.cpu().numpy()
              # direction_np = direction_np / np.linalg.norm(direction_np)
              # mid_pose_np = (cur_pose_np + origin_np) / 2.0
              # mid_pose = PoseStamped()
              # mid_pose.header.stamp = self.get_clock().now().to_msg()
              # mid_pose.header.frame_id = 'map'
              # mid_pose.pose.position.x = float(mid_pose_np[0])
              # mid_pose.pose.position.y = float(mid_pose_np[1])
              # mid_pose.pose.position.z = float(mid_pose_np[2])
              # mid_pose.pose.orientation.w = 1.0
              # path.poses.append(mid_pose)

              # target = origin_np + direction_np * magnitude
              # for factor in [0.0, 1.0]:
              #     pose = PoseStamped()
              #     pose.header.stamp = self.get_clock().now().to_msg()
              #     pose.header.frame_id = "map"
              #     pose.pose.position.x = float(origin_np[0]) * (1 - factor) + float(target[0]) * factor
              #     pose.pose.position.y = float(origin_np[1]) * (1 - factor) + float(target[1]) * factor
              #     pose.pose.position.z = float(origin_np[2]) * (1 - factor) + float(target[2]) * factor
              #     pose.pose.orientation.w = 1.0
              #     path.poses.append(pose)


              # TODO: this is using all rays 
              prev_target = cur_pose_np
              for ii, (avg_origin, avg_direction) in enumerate(group_averages):
                origin_np = avg_origin.cpu().numpy()
                direction_np = avg_direction.cpu().numpy()

                origin = origin_np
                direction = direction_np / np.linalg.norm(direction_np)

                #mid-point for smoothing entry
                mid_pose_np = (prev_target + origin) / 2.0
                mid_pose = PoseStamped()
                mid_pose.header.stamp = self.get_clock().now().to_msg()
                mid_pose.header.frame_id = 'map'
                mid_pose.pose.position.x = float(mid_pose_np[0])
                mid_pose.pose.position.y = float(mid_pose_np[1])
                mid_pose.pose.position.z = float(mid_pose_np[2])
                mid_pose.pose.orientation.w = 1.0
                path.poses.append(mid_pose)

                target = origin + direction * magnitude
                for factor in [0.0, 1.0]:
                  pose = PoseStamped()
                  pose.header.stamp = self.get_clock().now().to_msg()
                  pose.header.frame_id = "map"
                  pose.pose.position.x = float(origin[0]) * (1 - factor) + float(target[0]) * factor
                  pose.pose.position.y = float(origin[1]) * (1 - factor) + float(target[1]) * factor
                  pose.pose.position.z = float(origin[2]) * (1 - factor) + float(target[2]) * factor
                  pose.pose.orientation.w = 1.0
                  path.poses.append(pose)
                
                prev_target = target

              # unit_dir = direction / np.linalg.norm(direction)
              # target = origin + unit_dir * magnitude
              self.path_publisher.publish(path)

              rviz_filtered_dir = True ##visualize filtered rays on RVIZ
              if rviz_filtered_dir:
                self.clear_filtered_rays()
                arrow_length = 2
                filtered_marker_array = MarkerArray()
                #assert filtered_origins.shape[0] == filtered_directions.shape[0]
                colors = [(1.0, 0.0, 0.0),  # red
                          (0.0, 1.0, 0.0),  # green
                          (0.0, 0.0, 1.0),  # blue
                          (1.0, 1.0, 0.0),  # yellow
                          (0.0, 1.0, 1.0),  # cyan
                          (1.0, 0.0, 1.0),  # magenta
                          (0.5, 0.5, 0.5),  # gray
                          (1.0, 0.5, 0.0),  # orange
                          (0.5, 0.0, 1.0),  # purple
                          (0.0, 0.5, 0.5)   # teal
                          ]
                num_groups = len(angle_groups)
                
                j=0
                for i, group in enumerate(angle_groups):
                  idxes = group['indices']
                  rr,gg,bb = colors[i % len(colors)]
                  for idx in idxes:
                    dir0 = dir_world[idx].cpu().numpy()
                    p0 = orig_world[idx].cpu().numpy()
                    p1 = p0 + arrow_length*dir0
                    arrow = Marker()
                    arrow.header.frame_id = "map"
                    arrow.header.frame_id = "map"
                    arrow.header.stamp = self.get_clock().now().to_msg()
                    arrow.ns = "arrows"
                    arrow.id = j
                    arrow.type = Marker.ARROW
                    arrow.action = Marker.ADD
                    arrow.points = [Point(x=float(p0[0]), y=float(p0[1]), z=float(p0[2])), Point(x=float(p1[0]), y=float(p1[1]), z=float(p1[2]))]
                    arrow.scale.x = 0.6 #shaft diameter
                    arrow.scale.y = 1.2 #head diameter
                    arrow.scale.z = 0.75 #head length
                    arrow.color.r = rr
                    arrow.color.g = gg
                    arrow.color.b = bb
                    arrow.color.a = 0.5
                    filtered_marker_array.markers.append(arrow)
                    j += 1
                self.prev_filtered_marker_ids = j
                self.filtered_rays_publisher.publish(filtered_marker_array)


                # for i in range(filtered_directions.shape[0]):
                #   p0_ = filtered_origins[i].cpu().numpy()
                #   p0 = np.array([p0_[2],-p0_[0],-p0_[1]])
                #   dir0_ = filtered_directions[i].cpu().numpy()
                #   dir0 = np.array([dir0_[2], -dir0_[0], -dir0_[1]])
                #   p1 = p0 + arrow_length*dir0
                #   arrow = Marker()
                #   arrow.header.frame_id = "map"
                #   arrow.header.stamp = self.get_clock().now().to_msg()
                #   arrow.ns = "arrows"
                #   arrow.id = i
                #   arrow.type = Marker.ARROW
                #   arrow.action = Marker.ADD
                #   arrow.points = [Point(x=float(p0[0]), y=float(p0[1]), z=float(p0[2])), Point(x=float(p1[0]), y=float(p1[1]), z=float(p1[2]))]
                #   arrow.scale.x = 0.6 #shaft diameter
                #   arrow.scale.y = 1.2 #head diameter
                #   arrow.scale.z = 0.75 #head length
                #   arrow.color.r = 1.0
                #   arrow.color.g = 0.2
                #   arrow.color.b = 0.6
                #   arrow.color.a = 0.5
                #   filtered_marker_array.markers.append(arrow)
                # self.prev_filtered_marker_ids = filtered_directions.shape[0]
                # self.filtered_rays_publisher.publish(filtered_marker_array)
              


      if self.vis is not None:
        if i % self.cfg.vis.input_period == 0:
          self.mapper.vis_update(**r)
        if i % self.cfg.vis.map_period == 0:
          self.mapper.vis_map()

      if i % self.cfg.querying.period == 0:
        self.run_queries()

      if self.vis is not None:
        self.vis.step()

      # Stat calculation
      total_frames_processed += rgb_img.shape[0]
      map_p = map_t1-map_t0
      total_map += map_p
      map_thr = rgb_img.shape[0] / map_p
      wall_t1 = time.time()
      wall_p = wall_t1 - wall_t0
      wall_thr = rgb_img.shape[0] / wall_p
      wall_t0 = wall_t1
      logger.info("[#%4d#] Wall (#%6.4f# ms/batch - #%6.2f# frame/s), "
                  "Mapping (#%6.4f# ms/batch - #%6.2f# frame/s), "
                  "Mapping/Wall (#%6.4f%%)", 
                  i, wall_p*1e3, wall_thr, map_p*1e3, map_thr,
                  map_p/wall_p*100)

      with self._status_lock:
        if self.status != MappingServer.Status.MAPPING:
          logger.info("Mapping stopped.")
          break

    # Final stat calculation
    total_wall_t1 = time.time()
    total_wall = total_wall_t1 - total_wall_t0
    if total_map > 0 and total_wall > 0:
      logger.info("Total Wall (#%6.4f# ms/batch - #%6.2f# frame/s), "
                  "Mapping (#%6.4f# ms/batch - #%6.2f# frame/s), "
                  "Mapping/Wall (#%6.4f%%)", 
                  total_wall*1e3, total_frames_processed/total_wall,
                  total_map*1e3, total_frames_processed/total_map,
                  total_map/total_wall*100)

    # Shutting down or transitioning to idling
    self._status_lock.acquire()
    if self.status == MappingServer.Status.MAPPING:
      if self.messaging_service is not None:
        self.status = MappingServer.Status.IDLE
        try:
          self.dataset.shutdown()
        except AttributeError:
          pass # Its fine dataset doesn't have shutdown function
      else:
        self.shutdown()
        return

    # No new data is coming so we only need to add new queries and not
    # update old ones. Unless compute_prob is set to true b.c new queries
    # will not affect old results.
    if not self.cfg.querying.compute_prob:
      self._queries_feats = None
      self._queries_labels.clear()
    # Idling loop
    while self.status == MappingServer.Status.IDLE:
      self._status_lock.release()
      time.sleep(1)
      with self._query_lock:
        if self._queries_updated:
          self.run_queries()
      self._status_lock.acquire()

    self.status = MappingServer.Status.CLOSED
    self._status_lock.release()
    self.shutdown()

  def shutdown(self):
    with self._status_lock:
      self.status = MappingServer.Status.CLOSING
    if self.messaging_service is not None:
      self.messaging_service.shutdown()
    if self.dataset is not None:
      try:
        self.dataset.shutdown()
      except AttributeError:
        pass
    if self.vis is not None:
      try:
        self.vis.shutdown()
      except AttributeError:
        pass
    with self._status_lock:
      self.status = MappingServer.Status.CLOSED
  
  def target_object_callback(self, msg):
    data_cleaned = msg.data.strip().lower()
    if data_cleaned == "":
      self._target_object = None
    else:
      self._target_object = data_cleaned
    print("self._target_object", self._target_object)

  def clear_filtered_rays(self):
    if self.prev_filtered_marker_ids > 0:
      clear_marker_array = MarkerArray()
      for i in range(self.prev_filtered_marker_ids):
        clear_marker = Marker()
        clear_marker.header.frame_id = "map"
        clear_marker.header.stamp = self.get_clock().now().to_msg()
        clear_marker.ns = "arrows"
        clear_marker.id = i  # IDs must match those previously used
        clear_marker.action = Marker.DELETE
        clear_marker_array.markers.append(clear_marker)
      self.filtered_rays_publisher.publish(clear_marker_array)

  def create_pointcloud2_msg(self, xyz):
    if isinstance(xyz, torch.Tensor):
      xyz = xyz.detach().cpu().numpy()
    elif isinstance(xyz, np.ndarray):
      xyz = xyz
    else:
      raise TypeError(f"Expected torch.Tensor or numpy.ndarray, got {type(xyz)}")
    header = Header()
    header.stamp = self.get_clock().now().to_msg()
    header.frame_id = 'map'
    fields =  [PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
               PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
               PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1)]
    points = []
    for i in range(xyz.shape[0]):
      x,y,z = xyz[i]
      points.append([x,y,z])
    return point_cloud2.create_cloud(header, fields, points)


  def create_colored_pointcloud_msg(self, xyz_tensor, rgb_tensor):
    xyz = xyz_tensor.cpu().numpy()
    rgb = (rgb_tensor*255).cpu().numpy()
    assert xyz.shape[0] == rgb.shape[0]

    def pack_rgb(r,g,b):
      rgb_int = (int(r) << 16)| (int(g) << 8) | int (b)
      return struct.unpack('f', struct.pack('I', rgb_int))[0]
    
    points = []
    for i in range(xyz.shape[0]):
      xo,yo,zo = xyz[i]
      x,y,z = zo,-xo,-yo
      r,g,b = rgb[i]
      rgb_packed = pack_rgb(r,g,b)
      points.append([x,y,z,rgb_packed])
    
    fields = [PointField(name='x',offset=0,datatype=PointField.FLOAT32, count=1), 
              PointField(name='y',offset=4,datatype=PointField.FLOAT32, count=1), 
              PointField(name='z',offset=8,datatype=PointField.FLOAT32, count=1), 
              PointField(name='rgb',offset=12,datatype=PointField.FLOAT32, count=1)]
    header = Header()
    header.stamp = self.get_clock().now().to_msg()
    header.frame_id = 'map'
    return point_cloud2.create_cloud(header,fields, points)


def signal_handler(mapping_server: MappingServer, sig, frame):
  with mapping_server._status_lock:
    if mapping_server.status == MappingServer.Status.MAPPING:
      if mapping_server.messaging_service is not None:
        logger.info(
          "Received interrupt signal. Stopping mapping. Messaging service is "
          "still online. Interrupt again to shutdown.")

        mapping_server.status = MappingServer.Status.IDLE
      else:
        logger.info("Received interrupt signal. Shutting down.")
        mapping_server.status = MappingServer.Status.CLOSING

    elif mapping_server.status == MappingServer.Status.IDLE:
      logger.info("Received interrupt signal. Shutting down.")
      mapping_server.status = MappingServer.Status.CLOSING
  try:
    mapping_server.dataset.shutdown()
  except AttributeError:
    pass # Its fine dataset doesn't have shutdown function

@hydra.main(version_base=None, config_path="configs", config_name="default")
@torch.no_grad()
def main(cfg = None):
  if cfg.seed >= 0:
    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)

  rclpy.init()
  try:
    server = MappingServer(cfg)
  except KeyboardInterrupt:
    logger.info("Shutdown before initializing completed.")
    return
  

  spin_thread = threading.Thread(target=rclpy.spin, args=(server,), daemon=True)
  spin_thread.start()

  signal.signal(signal.SIGINT, partial(signal_handler, server))
  try:
    server.run()
  except Exception as e:
    server.shutdown()
    raise e

if __name__ == "__main__":
  # Cleanup for nanobind. See https://github.com/wjakob/nanobind/issues/19
  def cleanup():
    import typing
    for cleanup in typing._cleanups:
      cleanup()
  atexit.register(cleanup)

  main()
