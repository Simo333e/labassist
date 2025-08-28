std_msgs/Header header
string level              # "soft" | "hard"
string code               # "OUT_OF_ORDER" | "TIMEOUT" | "UNKNOWN" | "OK"
string message            # human-readable
string expected_next      # name of expected action
string observed_action    # last observed action (if any)
