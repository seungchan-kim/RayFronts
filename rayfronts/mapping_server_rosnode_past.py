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
from std_msgs.msg import String
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
import scipy.ndimage
from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs.msg import Image
from std_msgs.msg import Header, ColorRGBA
from sensor_msgs_py import point_cloud2
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
from rayfronts import geometry3d as g3d
from rayfronts.utils import compute_cos_sim

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
    self.pc2_publisher = self.create_publisher(PointCloud2, '/colored_pointcloud', 10)
    self.rays_publisher = self.create_publisher(MarkerArray, '/rays', 10)
    self.filtered_rays_publisher = self.create_publisher(MarkerArray, '/filtered_rays', 10)

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

    self.sky_feat = self.encoder.encode_labels(['sky'])

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

      #visualize 3d voxels on rviz
      r, pc_xyz, pc_rgb, pc_feat, ray_angles, ray_feat = self.mapper.process_posed_rgbd(rgb_img, depth_img, pose_4x4, **kwargs)
      if pc_xyz is not None and pc_rgb is not None and pc_feat is not None:
        #filter out sky voxels
        #pc_lang_aligned = self.mapper.encoder.align_spatial_features_with_language(pc_feat.unsqueeze(-1).unsqueeze(-1)).squeeze()
        #pc_sim_score = compute_cos_sim(self.sky_feat, pc_lang_aligned, softmax=False)
        #sky_indices = torch.nonzero(pc_sim_score > 0.02, as_tuple=True)[0]
        #print('sky', sky_indices.shape)
        #mask = torch.ones(pc_xyz.size(0),dtype=torch.bool)
        #mask[sky_indices] = False
        #pc_xyz_ = pc_xyz[mask]
        #pc_rgb_ = pc_rgb[mask] 

        colored_pc_msg = self.create_colored_pointcloud_msg(pc_xyz, pc_rgb)
        self.pc2_publisher.publish(colored_pc_msg)
        
        #print("pc_xyz", pc_xyz.shape)
        #print("pc_rgb", pc_rgb.shape)
        #print("pc_feat", pc_feat.shape)
      
      #visualize rays from rayfronts
      #print("ray_angles", ray_angles)
      #print("ray_feat", ray_feat)
      if ray_angles is not None and ray_feat is not None and ray_feat.shape[0] > 0 and ray_angles.shape[0] > 0:
        rorig = ray_angles[:,:3]
        ang = torch.deg2rad(ray_angles[:,3:])
        rdir = torch.stack(g3d.spherical_to_cartesian(1,ang[:,0],ang[:,1]),dim=-1)
        rviz_rays = True
        if rviz_rays:
          arrow_length=1
          marker_array = MarkerArray()
          assert rorig.shape[0] == rdir.shape[0]
          for i in range(rorig.shape[0]):
            p0_ = rorig[i].cpu().numpy()
            p0 = np.array([p0_[2],-p0_[0],-p0_[1]])
            dir0_ = rdir[i].cpu().numpy()
            dir0 = np.array([dir0_[2],-dir0_[0],-dir0_[1]])
            p1 = p0 + arrow_length * dir0
            arrow = Marker()
            arrow.header.frame_id = 'map'
            arrow.header.stamp = self.get_clock().now().to_msg()
            arrow.ns = "arrows"
            arrow.id = i
            arrow.type = Marker.ARROW
            arrow.action = Marker.ADD
            arrow.points=[Point(x=float(p0[0]),y=float(p0[1]),z=float(p0[2])), Point(x=float(p1[0]),y=float(p1[1]),z=float(p1[2]))]
            arrow.scale.x = 0.2
            arrow.scale.y = 0.4
            arrow.scale.z = 0.25
            arrow.color.r = 0.5
            arrow.color.g = 0.8
            arrow.color.b = 0.9
            arrow.color.a = 0.8
            marker_array.markers.append(arrow)
          self.rays_publisher.publish(marker_array)
      
        #visualize filtered_rays
        print("ray_feat shape", ray_feat.shape)
        ray_lang_aligned = self.mapper.encoder.align_spatial_features_with_language(ray_feat.unsqueeze(-1).unsqueeze(-1))
        
        if ray_lang_aligned.ndim == 4:
            ray_lang_aligned = ray_lang_aligned.squeeze(-1).squeeze(-1)
        if ray_lang_aligned.ndim == 2:
            ray_lang_aligned = ray_lang_aligned
        elif ray_lang_aligned.ndim == 1:
            ray_lang_aligned = ray_lang_aligned.unsqueeze(0)
        
        if self._queries_feats is not None:
          #from pdb import set_trace as bp
          #bp()
          print("self._queries_feats['text'] shape", self._queries_feats['text'])
          print("ray_lang_aligned shape", ray_lang_aligned.shape)
          ray_scores = compute_cos_sim(self._queries_feats['text'], ray_lang_aligned, softmax=False)
          indices = torch.nonzero(ray_scores>0.07, as_tuple=True)[0]
          filtered_origins = rorig[indices]
          filtered_directions = rdir[indices]
          print("indices", indices.shape)
          rviz_filtered_rays = True
          if rviz_filtered_rays:
            arrow_length = 4
            marker_array = MarkerArray()
            assert filtered_origins.shape[0] == filtered_directions.shape[0]
            for i in range(filtered_directions.shape[0]):
              p0_ = filtered_origins[i].cpu().numpy()
              p0 = np.array([p0_[2],-p0_[0],-p0_[1]])
              dir0_ = filtered_directions[i].cpu().numpy()
              dir0 = np.array([dir0_[2],-dir0_[0],-dir0_[1]])
              p1 = p0 + arrow_length*dir0
              arrow = Marker()
              arrow.header.frame_id = 'map'
              arrow.header.stamp = self.get_clock().now().to_msg()
              arrow.ns = "arrows"
              arrow.id = i
              arrow.type = Marker.ARROW
              arrow.action = Marker.ADD
              arrow.points = [Point(x=float(p0[0]),y=float(p0[1]),z=float(p0[2])), Point(x=float(p1[0]),y=float(p1[1]),z=float(p1[2]))]
              arrow.scale.x=  0.6 #shaft diameter
              arrow.scale.y = 1.2 #head diameter
              arrow.scale.z = 0.75 #head length
              arrow.color.r = 1.0
              arrow.color.g = 0.2
              arrow.color.b = 0.6
              arrow.color.a = 1.0
              marker_array.markers.append(arrow)
            self.filtered_rays_publisher.publish(marker_array)
        norms_ = torch.norm(filtered_directions, dim=1)
        mean_origin = torch.mean(filtered_origins, dim=0)
        mean_direction = torch.mean(filtered_directions, dim=0)

        origin_np = mean_origin.cpu().numpy()
        direction_np = mean_direction.cpu().numpy()
        origin = np.array([origin_np[2],-origin_np[0],-origin_np[1]])
        direction = np.array([direction_np[2],-direction_np[0],-direction_np[1]])
        magnitude = 5

        path = Path()
        path.header.stamp = self.get_clock().now().to_msg()
        path.header.frame_id = "map"
        unit_dir = direction / np.linalg.norm(direction)
        target = origin + unit_dir * magnitude

        for factor in [0.0, 1.0]:
            pose = PoseStamped()
            pose.header.stamp = self.get_clock().now().to_msg()
            pose.header.frame_id = "map"
            pose.pose.position.x = origin[0] * (1-factor) + target[0] * factor
            pose.pose.position.y = origin[1] * (1-factor) + target[1] * factor
            pose.pose.position.z = 15.0#origin[2] * (1-factor) + target[2] * factor
            pose.pose.orientation.w = 1.0
            path.poses.append(pose)

        self.path_publisher.publish(path)
        


      map_t1 = time.time()

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

  def create_colored_pointcloud_msg(self, xyz_tensor, rgb_tensor):
    xyz = xyz_tensor.cpu().numpy()
    rgb = (rgb_tensor*255).cpu().numpy()
    assert xyz.shape[0] == rgb.shape[0]

    def pack_rgb(r,g,b):
      rgb_int = (int(r) << 16) | (int(g) << 8) | int(b)
      return struct.unpack('f', struct.pack('I', rgb_int))[0]
    
    points = []
    for i in range(xyz.shape[0]):
      xo,yo,zo = xyz[i]
      x,y,z = zo,-xo,-yo
      r,g,b = rgb[i]
      rgb_packed= pack_rgb(r,g,b)
      points.append([x,y,z,rgb_packed])
    
    fields = [PointField(name='x',offset=0,datatype=PointField.FLOAT32, count=1), 
              PointField(name='y',offset=4,datatype=PointField.FLOAT32, count=1), 
              PointField(name='z',offset=8,datatype=PointField.FLOAT32, count=1), 
              PointField(name='rgb',offset=12,datatype=PointField.FLOAT32, count=1)]
    header = Header()
    header.stamp = self.get_clock().now().to_msg()
    header.frame_id = 'map'
    return point_cloud2.create_cloud(header, fields, points)

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
