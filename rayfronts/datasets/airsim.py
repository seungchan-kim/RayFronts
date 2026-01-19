"""Defines a data loader that connects to a running Airsim instance

Requires airsim

Much faster message fetching fix:
https://github.com/microsoft/AirSim/issues/3333#issuecomment-827894198
"""
import threading
import queue
import logging
import time

import airsim
import cv2
import numpy as np
import torch
import scipy.spatial.transform as st

from rayfronts.datasets.base import PosedRgbdDataset
from rayfronts import geometry3d as g3d

logger = logging.getLogger(__name__)

class AirSimDataset(PosedRgbdDataset):
  """A dataset that connects to a running AirSim instance and fetches RGBD frames.
  
  Attributes:
    intrinsics_3x3:  See base.
    rgb_h: See base.
    rgb_w: See base.
    depth_h: See base.
    depth_w: See base.
    frame_skip: See base.
    interp_mode: See base.
    vehicle_name: See __init__.
    camera_name: See __init__.
    depth_cutoff: See __init__.
    ip: See __init__.
    port: See __init__.
  """

  def __init__(self,
               rgb_resolution = None,
               depth_resolution = None,
               frame_skip: int = 0,
               interp_mode: str ="bilinear",
               camera_name: str = "Front",
               vehicle_name: str = "Drone1",
               depth_cutoff: float = 50,
               ip: str = "",
               port: int = 41451):
    """
    Args:
      rgb_resolution: See base.
      depth_resolution: See base.
      frame_skip: See base.
      interp_mode: See base.
      camera_name: Name of the AirSim camera to fetch images from.
      vehicle_name: Name of the AirSim vehicle to fetch images from.
      depth_cutoff: Maximum depth value to keep. Values beyond this are set to
        inf. Set to -1 to disable.
      ip: IP address of the AirSim instance to connect to.
      port: Port of the AirSim instance to connect to.
    """
    super().__init__(rgb_resolution=rgb_resolution,
                     depth_resolution=depth_resolution,
                     frame_skip=frame_skip,
                     interp_mode=interp_mode)

    self._client = airsim.MultirotorClient(ip=ip, port=port)
    self._client.enableApiControl(True)
    self._client.confirmConnection()
    self.camera_name = camera_name
    self.vehicle_name = vehicle_name
    self.depth_cutoff = depth_cutoff

    # Cache Intrinsics
    assert self.depth_h > 0 and self.depth_w > 0, (
      "AirsimDataset requires depth_resolution to be set.")
    self.intrinsics_3x3 = self._get_intrinsics()

    self._src2rdf_transform = g3d.mat_3x3_to_4x4(
      g3d.get_coord_system_transform("frd", "rdf"))

    # Buffer for background fetching
    self._shutdown_event = threading.Event()
    self._frame_queue = queue.Queue(maxsize=3)
    self._prefetch_thread = threading.Thread(
      target=self._prefetcher, daemon=True)
    self._prefetch_thread.start()

  def _prefetcher(self):
    requests = [
      airsim.ImageRequest(self.camera_name, airsim.ImageType.Scene,
                          pixels_as_float=False, compress=True),
      airsim.ImageRequest(self.camera_name, airsim.ImageType.DepthPlanar,
                          pixels_as_float=True, compress=False)
    ]
    while not self._shutdown_event.is_set():
      if not self._frame_queue.full():
        responses = self._client.simGetImages(
          requests, vehicle_name=self.vehicle_name)
        # 1. Process RGB (using the 'compressed' byte stream)
        rgb_resp = responses[0]
        # When compressed=True, image_data_uint8 contains the PNG/JPG binary blob
        img_1d = np.frombuffer(rgb_resp.image_data_uint8, dtype=np.uint8)
        img_bgr = cv2.imdecode(img_1d, cv2.IMREAD_COLOR)
        rgb_img = torch.from_numpy(img_bgr[..., ::-1].copy()).permute(2, 0, 1).float() / 255.0

        # 2. Process Depth (using the 'pixels_as_float' list)
        depth_resp = responses[1]
        # When pixels_as_float=True, we use image_data_float
        depth_img = np.array(depth_resp.image_data_float, dtype=np.float32)
        depth_img = depth_img.reshape(depth_resp.height, depth_resp.width)
        depth_img = torch.from_numpy(depth_img).unsqueeze(0)

        if self.depth_cutoff > 0:
            depth_img[depth_img > self.depth_cutoff] = torch.inf
            depth_img[depth_img == 0] = torch.nan

        # 4. Pack for the main loop
        self._frame_queue.put({
            "rgb": rgb_img,
            "depth": depth_img,
            "pos": depth_resp.camera_position,
            "ori": depth_resp.camera_orientation
        })
      else:
        time.sleep(0.1)

  def _get_intrinsics(self):
    """Extracts intrinsics from AirSim camera info."""
    cam_info = self._client.simGetCameraInfo(self.camera_name, self.vehicle_name)
    # AirSim FOV is horizontal in degrees
    fov_rad = np.radians(cam_info.fov)
    f_x = self.depth_w / (2 * np.tan(fov_rad / 2))
    f_y = f_x # Assuming square pixels
    c_x = self.depth_w / 2
    c_y = self.depth_h / 2

    K = np.array([
        [f_x, 0,   c_x],
        [0,   f_y, c_y],
        [0,   0,   1]
    ], dtype=np.float32)
    return torch.from_numpy(K)

  def __iter__(self):
    while not self._shutdown_event.is_set():
      frame_data = self._frame_queue.get()

      rgb_img = frame_data["rgb"]
      depth_img = frame_data["depth"]

      pos = frame_data["pos"]
      ori = frame_data["ori"]
      r = st.Rotation.from_quat(
        [ori.x_val, ori.y_val, ori.z_val, ori.w_val]).as_matrix()
      src_pose_4x4 = np.eye(4, dtype="float")
      src_pose_4x4[:3, :3] = r
      src_pose_4x4[:3, -1] = [pos.x_val, pos.y_val, pos.z_val]
      src_pose_4x4 = torch.from_numpy(src_pose_4x4).float()
      rdf_pose_4x4 = g3d.transform_pose_4x4(
        src_pose_4x4, self._src2rdf_transform)

      if (self.rgb_h != rgb_img.shape[-2] or
          self.rgb_w != rgb_img.shape[-1]):
        rgb_img = torch.nn.functional.interpolate(rgb_img.unsqueeze(0),
          size=(self.rgb_h, self.rgb_w), mode=self.interp_mode,
          antialias=self.interp_mode in ["bilinear", "bicubic"]).squeeze(0)

      if (self.depth_h != depth_img.shape[-2] or
          self.depth_w != depth_img.shape[-1]):
        depth_img = torch.nn.functional.interpolate(depth_img.unsqueeze(0),
          size=(self.depth_h, self.depth_w),
          mode="nearest-exact").squeeze(0)

      yield {
        "rgb_img": rgb_img,
        "depth_img": depth_img,
        "pose_4x4": rdf_pose_4x4,
      }

  def shutdown(self):
    self._shutdown_event.set()
    del self._client
    if self._prefetch_thread is not None:
      self._prefetch_thread.join()
    logger.info("Airsim loader shutdown.")
