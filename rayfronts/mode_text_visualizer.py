from visualization_msgs.msg import Marker
from std_msgs.msg import ColorRGBA, String

class ModeTextVisualizer:
    def __init__(self, get_clock, mode_text_publisher, node):
       self.get_clock = get_clock
       self.mode_text_publisher = mode_text_publisher
       self.latest_trajectory_length = '0.0'
       self.latest_success_count = '0'
       self.latest_progress = '0.0'
       self.latest_ppl = '0.0'
       self.traj_length_sub = node.create_subscription(String, "trajectory_length", self.traj_length_callback, 10)
       self.success_count_sub = node.create_subscription(String, "success_count", self.success_count_callback, 10)
       self.progress_sub = node.create_subscription(String, "progress", self.progress_callback, 10)
       self.ppl_sub = node.create_subscription(String, "ppl", self.ppl_callback, 10)
       self.node = node


    def traj_length_callback(self, msg):
        self.latest_trajectory_length = msg.data

    def success_count_callback(self, msg):
        self.latest_success_count = msg.data

    def progress_callback(self, msg):
        self.latest_progress = msg.data
    
    def ppl_callback(self, msg):
        self.latest_ppl = msg.data

    def modeTextVisualize(self, cur_pose_np, target_objects, behavior_mode):
        mode_text_marker = Marker()
        mode_text_marker.header.frame_id = "map"
        mode_text_marker.header.stamp = self.get_clock().now().to_msg()
        mode_text_marker.ns = "mode_text"
        mode_text_marker.id = 0
        mode_text_marker.type = Marker.TEXT_VIEW_FACING
        mode_text_marker.action = Marker.ADD
        mode_text_marker.pose.position.x = cur_pose_np[0]
        mode_text_marker.pose.position.y = cur_pose_np[1]
        mode_text_marker.pose.position.z = cur_pose_np[2] + 20
        mode_text_marker.pose.orientation.w = 1.0
        mode_text_marker.scale.z = 2.0
        mode_text_marker.color = ColorRGBA(r=1.0,g=1.0,b=1.0,a=1.0)

        mode_text_marker.text = ""
        mode_text_marker.text += f"Trajectory length: {self.latest_trajectory_length} m\n"
        mode_text_marker.text += f"Number of reached objects: {self.latest_success_count}\n"
        mode_text_marker.text += f"Progress: {self.latest_progress}\n"
        mode_text_marker.text += f"PPL: {self.latest_ppl}\n\n"

        if len(target_objects) == 0:
            mode_text_marker.text += "No Target Object" + "\nExploration Mode: Frontier-based"
        else:
            target_object_string = ", ".join(target_objects)
            mode_text_marker.text += "Target Object: " + target_object_string
            if behavior_mode == "Frontier-based":
                mode_text_marker.text += "\nDidn't find any semantic cues"
                mode_text_marker.text += "\nExploration Mode: Frontier-based"
            elif behavior_mode == "Voxel-based":
                mode_text_marker.text += "\nDetected Voxel clusters"
                mode_text_marker.text += "\nExploration Mode: Voxel-based"
            elif behavior_mode == "Ray-based":
                mode_text_marker.text += "\nDetected Rays"
                mode_text_marker.text += "\nExploration Mode: Ray-based"
            elif behavior_mode == "LVLM-guided":
                mode_text_marker.text += "\nExploration Mode: LVLM-guided"
                mode_text_marker.text += "\nGuiding Objects: " + ', '.join(self.node.behavior_manager.lvlm_guided_behavior.guiding_objects)
        mode_text_marker.lifetime.sec = 0
        self.mode_text_publisher.publish(mode_text_marker)

    