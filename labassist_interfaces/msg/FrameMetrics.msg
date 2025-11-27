# Latency metrics for a single frame through the pipeline
std_msgs/Header header

# Timestamps (nanoseconds since epoch)
uint64 t_frame_pub      # When camera_player published the frame
uint64 t_feature_start  # When feature node received the frame
uint64 t_feature_end    # When feature node finished extraction
uint64 t_mstcn_start    # When mstcn_infer received features
uint64 t_mstcn_end      # When mstcn_infer finished inference

# Derived latencies (computed by metrics_collector, in milliseconds)
float32 feature_latency_ms    # t_feature_end - t_feature_start
float32 mstcn_latency_ms      # t_mstcn_end - t_mstcn_start
float32 e2e_latency_ms        # t_mstcn_end - t_frame_pub

# Frame metadata
uint32 frame_seq              # Sequence number
string predicted_action       # Action predicted by MS-TCN
float32 confidence            # Prediction confidence

