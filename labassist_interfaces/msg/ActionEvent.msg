std_msgs/Header header
string action_name        # e.g., "pipette_A_to_B", "vortex", "incubate_start"
string actor              # e.g., "hand", "pipette"
string[] objects          # e.g., ["tube_A","tube_B"]
float32 confidence        # 0.0–1.0
