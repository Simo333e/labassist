#!/usr/bin/env python3
"""
FSM-based Pipette Tip Hygiene Validator

This module implements a finite state machine to validate proper pipette tip hygiene
in laboratory protocols. The key safety rule is:

    Before manipulating reagents, the user must:
    1. Eject the current pipette tip
    2. Attach a new (clean) pipette tip

This prevents cross-contamination between PCR tubes and reagent sources.

Author: LabAssist Project
Date: 2026-01-15
"""

from enum import Enum
from typing import List, Dict, Tuple, Optional
import numpy as np


class TipState(Enum):
    """States in the pipette tip hygiene FSM"""
    NO_TIP = "no_tip"              # No tip attached
    CLEAN_TIP = "clean_tip"        # Fresh tip attached, not used yet
    USED_TIP = "used_tip"          # Tip has been used for manipulation
    CONTAMINATED = "contaminated"  # Tip used on PCR tubes, unsafe for reagents


class AlertLevel(Enum):
    """Alert severity levels"""
    OK = "ok"
    WARNING = "warning"
    CRITICAL = "critical"


class Alert:
    """Represents a hygiene violation alert"""
    def __init__(
        self,
        frame: int,
        level: AlertLevel,
        message: str,
        predicted_action: str,
        fsm_state: TipState,
        context_frames: List[Tuple[int, int]]  # [(frame, action_class), ...]
    ):
        self.frame = frame
        self.level = level
        self.message = message
        self.predicted_action = predicted_action
        self.fsm_state = fsm_state
        self.context_frames = context_frames

    def to_dict(self) -> Dict:
        """Convert alert to dictionary for JSON serialization"""
        return {
            "frame": int(self.frame),
            "level": self.level.value,
            "message": self.message,
            "predicted_action": self.predicted_action,
            "fsm_state": self.fsm_state.value,
            "context": [
                {"frame": int(f), "action_class": int(a)}
                for f, a in self.context_frames
            ]
        }


class TipHygieneFSM:
    """
    Finite State Machine for validating pipette tip hygiene.
    
    Supports both 5-class and 7-class taxonomies.
    
    5-class taxonomy:
        0: none
        1: Manipulate PCR tube
        2: Manipulate reagent
        3: Attach pipette tip
        4: Eject pipette tip
    
    7-class taxonomy:
        0: none
        1: Adjust pipette volume
        2: Attach pipette tip
        3: Eject pipette tip
        4: Manipulate pcr tubes
        5: Manipulate reagents
        6: Switch pipette
    
    FSM Rules:
        - Attach tip → CLEAN_TIP
        - Eject tip → NO_TIP
        - Manipulate PCR tubes:
            - If CLEAN_TIP → CONTAMINATED
            - If CONTAMINATED → remain CONTAMINATED
            - If NO_TIP → WARNING alert
        - Manipulate reagents:
            - If CLEAN_TIP → USED_TIP (OK)
            - If CONTAMINATED → CRITICAL alert (cross-contamination!)
            - If USED_TIP → WARNING alert (reusing tip)
            - If NO_TIP → WARNING alert
        - Switch pipette (7-class only) → NO_TIP (assume tip is ejected)
    """
    
    def __init__(self, context_window: int = 10, num_classes: int = 5):
        """
        Initialize FSM
        
        Args:
            context_window: Number of frames to keep in context for alerts
            num_classes: Number of action classes (5 or 7)
        """
        self.context_window = context_window
        self.num_classes = num_classes
        
        # Detect taxonomy and set action IDs
        if num_classes == 5:
            # 5-class taxonomy
            self.ACTION_NONE = 0
            self.ACTION_MANIPULATE_PCR = 1
            self.ACTION_MANIPULATE_REAGENT = 2
            self.ACTION_ATTACH_TIP = 3
            self.ACTION_EJECT_TIP = 4
            self.ACTION_ADJUST_VOLUME = None
            self.ACTION_SWITCH_PIPETTE = None
        elif num_classes == 7:
            # 7-class taxonomy
            self.ACTION_NONE = 0
            self.ACTION_ADJUST_VOLUME = 1
            self.ACTION_ATTACH_TIP = 2
            self.ACTION_EJECT_TIP = 3
            self.ACTION_MANIPULATE_PCR = 4
            self.ACTION_MANIPULATE_REAGENT = 5
            self.ACTION_SWITCH_PIPETTE = 6
        else:
            raise ValueError(f"Unsupported number of classes: {num_classes}. Must be 5 or 7.")
        
        self.reset()
    
    def reset(self):
        """Reset FSM to initial state"""
        self.state = TipState.NO_TIP
        self.alerts: List[Alert] = []
        self.frame_history: List[Tuple[int, int]] = []  # [(frame, action), ...]
        self.state_history: List[Tuple[int, TipState]] = []  # [(frame, state), ...]
    
    def _add_to_history(self, frame: int, action: int):
        """Add frame to history and maintain window size"""
        self.frame_history.append((frame, action))
        if len(self.frame_history) > self.context_window:
            self.frame_history.pop(0)
    
    def _get_context(self) -> List[Tuple[int, int]]:
        """Get recent context frames"""
        return list(self.frame_history)
    
    def _trigger_alert(
        self,
        frame: int,
        level: AlertLevel,
        message: str,
        action_name: str
    ):
        """Trigger a hygiene violation alert"""
        alert = Alert(
            frame=frame,
            level=level,
            message=message,
            predicted_action=action_name,
            fsm_state=self.state,
            context_frames=self._get_context()
        )
        self.alerts.append(alert)
    
    def process_frame(self, frame: int, action: int, action_names: List[str]):
        """
        Process a single frame and update FSM state
        
        Args:
            frame: Frame number
            action: Predicted action class (0-6)
            action_names: List of action names for logging
        """
        action_name = action_names[action] if 0 <= action < len(action_names) else "unknown"
        
        # Add to history
        self._add_to_history(frame, action)
        
        # Record state before transition
        prev_state = self.state
        
        # FSM transitions based on action
        if action == self.ACTION_ATTACH_TIP:
            # Attaching a new tip → always clean
            self.state = TipState.CLEAN_TIP
        
        elif action == self.ACTION_EJECT_TIP:
            # Ejecting tip → no tip
            self.state = TipState.NO_TIP
        
        elif self.ACTION_SWITCH_PIPETTE is not None and action == self.ACTION_SWITCH_PIPETTE:
            # Switching pipette → assume tip is ejected (7-class only)
            self.state = TipState.NO_TIP
        
        elif action == self.ACTION_MANIPULATE_PCR:
            # Manipulating PCR tubes
            if self.state == TipState.NO_TIP:
                self._trigger_alert(
                    frame,
                    AlertLevel.WARNING,
                    "Manipulating PCR tubes without a tip attached",
                    action_name
                )
            elif self.state == TipState.CLEAN_TIP:
                # Clean tip used on PCR → now contaminated
                self.state = TipState.CONTAMINATED
            # If already contaminated or used, remain in that state
        
        elif action == self.ACTION_MANIPULATE_REAGENT:
            # Manipulating reagents - THE CRITICAL CHECK
            if self.state == TipState.NO_TIP:
                self._trigger_alert(
                    frame,
                    AlertLevel.WARNING,
                    "Manipulating reagent without a tip attached",
                    action_name
                )
            elif self.state == TipState.CONTAMINATED:
                self._trigger_alert(
                    frame,
                    AlertLevel.CRITICAL,
                    "CROSS-CONTAMINATION RISK: Using PCR-contaminated tip on reagent source",
                    action_name
                )
                # Tip is now even more contaminated
            elif self.state == TipState.USED_TIP:
                self._trigger_alert(
                    frame,
                    AlertLevel.WARNING,
                    "Reusing tip for reagent manipulation (should use fresh tip)",
                    action_name
                )
            elif self.state == TipState.CLEAN_TIP:
                # This is OK! Clean tip used on reagent
                self.state = TipState.USED_TIP
        
        # Record state after transition
        self.state_history.append((frame, self.state))
    
    def process_sequence(
        self,
        predictions: np.ndarray,
        action_names: List[str],
        start_frame: int = 0
    ) -> List[Alert]:
        """
        Process a full sequence of predictions
        
        Args:
            predictions: Array of predicted action classes [T]
            action_names: List of action names
            start_frame: Starting frame number (for videos with offset)
        
        Returns:
            List of alerts triggered during processing
        """
        self.reset()
        
        for i, action in enumerate(predictions):
            frame = start_frame + i
            self.process_frame(frame, int(action), action_names)
        
        return self.alerts
    
    def get_stats(self) -> Dict:
        """Get statistics about the processed sequence"""
        total_alerts = len(self.alerts)
        critical_alerts = sum(1 for a in self.alerts if a.level == AlertLevel.CRITICAL)
        warning_alerts = sum(1 for a in self.alerts if a.level == AlertLevel.WARNING)
        
        return {
            "total_alerts": total_alerts,
            "critical_alerts": critical_alerts,
            "warning_alerts": warning_alerts,
            "ok_alerts": total_alerts - critical_alerts - warning_alerts
        }


def validate_ground_truth_sequence(
    ground_truth: np.ndarray,
    action_names: List[str],
    context_window: int = 10,
    num_classes: int = None
) -> Tuple[List[Alert], int]:
    """
    Validate ground truth sequence to find TRUE violations
    
    This tells us how many violations actually exist in the ground truth,
    so we can compare against what the model detects.
    
    Args:
        ground_truth: Ground truth action classes [T]
        action_names: List of action names
        context_window: Context window size
        num_classes: Number of action classes (auto-detect if None)
    
    Returns:
        (alerts, num_true_violations)
    """
    if num_classes is None:
        num_classes = len(action_names)
    fsm = TipHygieneFSM(context_window=context_window, num_classes=num_classes)
    alerts = fsm.process_sequence(ground_truth, action_names)
    
    # Count true violations (critical alerts only)
    num_violations = sum(1 for a in alerts if a.level == AlertLevel.CRITICAL)
    
    return alerts, num_violations


def compare_predictions_vs_ground_truth(
    predictions: np.ndarray,
    ground_truth: np.ndarray,
    action_names: List[str],
    context_window: int = 10,
    num_classes: int = None
) -> Dict:
    """
    Compare FSM alerts from predictions vs ground truth
    
    This helps us understand:
    - True positives: Model detects violations that exist in GT
    - False positives: Model detects violations that don't exist in GT
    - False negatives: Model misses violations that exist in GT
    
    Args:
        predictions: Predicted action classes [T]
        ground_truth: Ground truth action classes [T]
        action_names: List of action names
        context_window: Context window size
        num_classes: Number of action classes (auto-detect if None)
    
    Returns:
        Dictionary with comparison metrics
    """
    if num_classes is None:
        num_classes = len(action_names)
    
    # Run FSM on predictions
    fsm_pred = TipHygieneFSM(context_window=context_window, num_classes=num_classes)
    pred_alerts = fsm_pred.process_sequence(predictions, action_names)
    
    # Run FSM on ground truth
    fsm_gt = TipHygieneFSM(context_window=context_window, num_classes=num_classes)
    gt_alerts = fsm_gt.process_sequence(ground_truth, action_names)
    
    # Extract critical alert frames
    pred_critical_frames = set(a.frame for a in pred_alerts if a.level == AlertLevel.CRITICAL)
    gt_critical_frames = set(a.frame for a in gt_alerts if a.level == AlertLevel.CRITICAL)
    
    # Calculate metrics (frame-level, with tolerance)
    tolerance = 5  # frames (0.5s at 10 FPS)
    
    def frames_match(f1_set, f2_set, tol):
        """Check if frames match within tolerance"""
        matched = set()
        for f1 in f1_set:
            for f2 in f2_set:
                if abs(f1 - f2) <= tol:
                    matched.add(f1)
                    break
        return matched
    
    true_positives = frames_match(pred_critical_frames, gt_critical_frames, tolerance)
    false_positives = pred_critical_frames - true_positives
    false_negatives = gt_critical_frames - frames_match(gt_critical_frames, pred_critical_frames, tolerance)
    
    # Calculate metrics
    precision = len(true_positives) / len(pred_critical_frames) if pred_critical_frames else 0.0
    recall = len(true_positives) / len(gt_critical_frames) if gt_critical_frames else 1.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        "predictions": {
            "total_alerts": len(pred_alerts),
            "critical_alerts": len(pred_critical_frames),
            "warning_alerts": sum(1 for a in pred_alerts if a.level == AlertLevel.WARNING),
            "alerts": [a.to_dict() for a in pred_alerts]
        },
        "ground_truth": {
            "total_alerts": len(gt_alerts),
            "critical_alerts": len(gt_critical_frames),
            "warning_alerts": sum(1 for a in gt_alerts if a.level == AlertLevel.WARNING),
            "alerts": [a.to_dict() for a in gt_alerts]
        },
        "metrics": {
            "true_positives": len(true_positives),
            "false_positives": len(false_positives),
            "false_negatives": len(false_negatives),
            "precision": precision,
            "recall": recall,
            "f1": f1
        }
    }


if __name__ == "__main__":
    # Simple test for both taxonomies
    print("Testing TipHygieneFSM...")
    
    # Test 7-class taxonomy
    print("\n" + "="*60)
    print("TEST 1: 7-class taxonomy")
    print("="*60)
    
    action_names_7 = [
        "none",
        "Adjust pipette volume",
        "Attach pipette tip",
        "Eject pipette tip",
        "Manipulate pcr tubes",
        "Manipulate reagents",
        "Switch pipette"
    ]
    
    # Test sequence with a violation:
    # 1. Attach tip (2)
    # 2. Manipulate PCR (4) - tip becomes contaminated
    # 3. Manipulate reagent (5) - VIOLATION!
    test_sequence_7 = np.array([
        0, 0,  # background
        2,     # attach tip
        0,     # background
        4, 4,  # manipulate PCR tubes
        0,     # background
        5, 5,  # manipulate reagents - VIOLATION!
        0, 0   # background
    ])
    
    fsm_7 = TipHygieneFSM(num_classes=7)
    alerts_7 = fsm_7.process_sequence(test_sequence_7, action_names_7)
    
    print(f"Test sequence processed: {len(test_sequence_7)} frames")
    print(f"Alerts triggered: {len(alerts_7)}")
    
    for alert in alerts_7:
        print(f"\n[{alert.level.value.upper()}] Frame {alert.frame}: {alert.message}")
        print(f"  Action: {alert.predicted_action}")
        print(f"  State: {alert.fsm_state.value}")
    
    stats_7 = fsm_7.get_stats()
    print(f"\nStatistics:")
    print(f"  Total alerts: {stats_7['total_alerts']}")
    print(f"  Critical: {stats_7['critical_alerts']}")
    print(f"  Warnings: {stats_7['warning_alerts']}")
    
    # Test 5-class taxonomy
    print("\n" + "="*60)
    print("TEST 2: 5-class taxonomy")
    print("="*60)
    
    action_names_5 = [
        "none",
        "Manipulate PCR tube",
        "Manipulate reagent",
        "Attach pipette tip",
        "Eject pipette tip"
    ]
    
    # Test sequence with a violation (5-class):
    # 1. Attach tip (3)
    # 2. Manipulate PCR (1) - tip becomes contaminated
    # 3. Manipulate reagent (2) - VIOLATION!
    test_sequence_5 = np.array([
        0, 0,  # background
        3,     # attach tip
        0,     # background
        1, 1,  # manipulate PCR tubes
        0,     # background
        2, 2,  # manipulate reagents - VIOLATION!
        0, 0   # background
    ])
    
    fsm_5 = TipHygieneFSM(num_classes=5)
    alerts_5 = fsm_5.process_sequence(test_sequence_5, action_names_5)
    
    print(f"Test sequence processed: {len(test_sequence_5)} frames")
    print(f"Alerts triggered: {len(alerts_5)}")
    
    for alert in alerts_5:
        print(f"\n[{alert.level.value.upper()}] Frame {alert.frame}: {alert.message}")
        print(f"  Action: {alert.predicted_action}")
        print(f"  State: {alert.fsm_state.value}")
    
    stats_5 = fsm_5.get_stats()
    print(f"\nStatistics:")
    print(f"  Total alerts: {stats_5['total_alerts']}")
    print(f"  Critical: {stats_5['critical_alerts']}")
    print(f"  Warnings: {stats_5['warning_alerts']}")
    
    print("\n" + "="*60)
    print("✓ FSM tests complete for both taxonomies!")
    print("="*60)




