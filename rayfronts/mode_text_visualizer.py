from visualization_msgs.msg import Marker
from std_msgs.msg import ColorRGBA

class ModeTextVisualizer:
    def __init__(self, get_clock, mode_text_publisher):
       self.get_clock = get_clock
       self.mode_text_publisher = mode_text_publisher
    
    def modeTextVisualize(self, cur_pose_np, target_object, behavior_mode):
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
        mode_text_marker.scale.z = 2.0
        mode_text_marker.color = ColorRGBA(r=1.0,g=1.0,b=1.0,a=1.0)

        if target_object is None:
            mode_text_marker.text = "No Target Object" + "\nExploration Mode: Frontier-based"
        else:
            mode_text_marker.text = "Target Object: " + target_object
            if behavior_mode == "Frontier-based":
                mode_text_marker.text += "\nDidn't find any semantic cues"
                mode_text_marker.text += "\nExploration Mode: Frontier-based"
            elif behavior_mode == "Voxel-based":
                mode_text_marker.text += "\nDetected Voxel clusters"
                mode_text_marker.text += "\nExploration Mode: Voxel-based"
            elif behavior_mode == "Ray-based":
                mode_text_marker.text += "\nDetected Rays"
                mode_text_marker.text += "\nExploration Mode: Ray-based"
        mode_text_marker.lifetime.sec = 0
        self.mode_text_publisher.publish(mode_text_marker)

    