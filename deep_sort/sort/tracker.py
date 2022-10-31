# vim: expandtab:ts=4:sw=4
from __future__ import absolute_import
import numpy as np
from . import kalman_filter
from . import linear_assignment
from . import iou_matching
from .track import Track


class Tracker:
    """
    This is the multi-target tracker.

    Parameters
    ----------
    metric : nn_matching.NearestNeighborDistanceMetric
        A distance metric for measurement-to-track association.
    max_age : int
        Maximum number of missed misses before a track is deleted.
    n_init : int
        Number of consecutive detections before the track is confirmed. The
        track state is set to `Deleted` if a miss occurs within the first
        `n_init` frames.

    Attributes
    ----------
    metric : nn_matching.NearestNeighborDistanceMetric
        The distance metric used for measurement to track association.
    max_age : int
        Maximum number of missed misses before a track is deleted.
    n_init : int
        Number of frames that a track remains in initialization phase.
    kf : kalman_filter.KalmanFilter
        A Kalman filter to filter target trajectories in image space.
    tracks : List[Track]
        The list of active tracks at the current time step.

    """

    def __init__(self, metric, max_iou_distance=0.7, max_age=70, n_init=5):
        self.metric = metric
        self.max_iou_distance = max_iou_distance
        self.max_age = max_age
        self.n_init = n_init

        self.kf = kalman_filter.KalmanFilter()
        self.tracks = []
        self._next_id = 1
        # list used to store the id which has been changed or duplicated once
        # self.changed_id = []

    def predict(self):
        """Propagate track state distributions one time step forward.

        This function should be called once every time step, before `update`.
        """
        for track in self.tracks:
            track.predict(self.kf)

    def update(
        self,
        detections,
        classes,
        scores,
        width_of_image,
        prevent_different_classes_match=False,
        match_across_boundary=False,
    ):
        """Perform measurement update and track management.

        Parameters
        ----------
        detections : List[deep_sort.detection.Detection]
            A list of detections at the current time step.

        """
        # matches_tracks = []
        matches_detections = []
        matches_tracks = []
        # confirmed id
        current_ids = []
        # all the ids
        current_ids_all = []
        # Run matching cascade.
        if prevent_different_classes_match == False:
            matches, unmatched_tracks, unmatched_detections = self._match(
                detections, width_of_image, match_across_boundary
            )
        else:
            matches, unmatched_tracks, unmatched_detections = self._match(
                detections, width_of_image, match_across_boundary, classes.tolist()
            )
        # print(matches)
        # Update the kf mean of the matched track.
        for track_idx, detection_idx in matches:
            # print(self.tracks[track_idx]._class, classes[detection_idx])
            self.tracks[track_idx].update(
                self.kf,
                detections[detection_idx],
                width_of_image,
                match_across_boundary,
            )
            self.tracks[track_idx]._score = scores[detection_idx]
            matches_tracks.append(track_idx)
            matches_detections.append(detection_idx)

        for track_idx in unmatched_tracks:
            self.tracks[track_idx].mark_missed()

        for detection_idx in unmatched_detections:
            self._initiate_track(
                detections[detection_idx],
                classes[detection_idx],
                scores[detection_idx],
            )

        for track in self.tracks:
            current_ids_all.append(track.track_id)
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            current_ids.append(track.track_id)

        self.tracks = [t for t in self.tracks if not t.is_deleted()]

        # Update distance metric.
        active_targets = [t.track_id for t in self.tracks if t.is_confirmed()]
        features, targets = [], []
        for track in self.tracks:
            if not track.is_confirmed():
                continue
            features += track.features
            targets += [track.track_id for _ in track.features]
            track.features = []
        self.metric.partial_fit(
            np.asarray(features), np.asarray(targets), active_targets
        )

    def _match(self, detections, width_of_image, match_across_boundary, classes=None):
        def gated_metric(tracks, dets, track_indices, detection_indices, classes):
            features = np.array([dets[i].feature for i in detection_indices])
            targets = np.array([tracks[i].track_id for i in track_indices])
            cost_matrix = self.metric.distance(features, targets)

            # on the basis of appearance distance, filter matches whose Mahalanobis distance is greater than a threshold
            cost_matrix = linear_assignment.gate_cost_matrix(
                self.kf,
                cost_matrix,
                tracks,
                dets,
                track_indices,
                detection_indices,
                width_of_image,
                match_across_boundary,
            )

            # limit the match between track and detection of different classes
            if classes != None:
                # in order not to match the detection and the track with different classes (person and other classes), set the corresponding figure large in the cost matrix
                # row means tracks
                for row in range(cost_matrix.shape[0]):
                    # col means detections
                    for col in range(cost_matrix.shape[1]):
                        # car, truck and bus are acceptable to be matched
                        if tracks[track_indices[row]]._class in [2, 5, 7] and classes[
                            detection_indices[col]
                        ] in [
                            2,
                            5,
                            7,
                        ]:
                            cost_matrix[row, col] = cost_matrix[row, col]
                        # bicycle and motorcycle are acceptable to be matched
                        elif tracks[track_indices[row]]._class in [0, 1, 3] and classes[
                            detection_indices[col]
                        ] in [
                            0,
                            1,
                            3,
                        ]:
                            cost_matrix[row, col] = cost_matrix[row, col]
                        elif (
                            tracks[track_indices[row]]._class
                            != classes[detection_indices[col]]
                        ):
                            cost_matrix[row, col] = 1e5
            return cost_matrix

        # Split track set into confirmed and unconfirmed tracks.
        confirmed_tracks = [i for i, t in enumerate(self.tracks) if t.is_confirmed()]
        unconfirmed_tracks = [
            i for i, t in enumerate(self.tracks) if not t.is_confirmed()
        ]

        # Associate confirmed tracks using appearance features.
        (
            matches_a,
            unmatched_tracks_a,
            unmatched_detections,
        ) = linear_assignment.matching_cascade(
            gated_metric,
            self.metric.matching_threshold,
            self.max_age,
            self.tracks,
            detections,
            confirmed_tracks,
            None,
            classes,
        )

        # Associate remaining tracks together with unconfirmed tracks using IOU.
        iou_track_candidates = unconfirmed_tracks + [
            k for k in unmatched_tracks_a if self.tracks[k].time_since_update == 1
        ]
        unmatched_tracks_a = [
            k for k in unmatched_tracks_a if self.tracks[k].time_since_update != 1
        ]

        (
            matches_b,
            unmatched_tracks_b,
            unmatched_detections,
        ) = linear_assignment.min_cost_matching(
            iou_matching.iou_cost,
            self.max_iou_distance,
            self.tracks,
            detections,
            iou_track_candidates,
            unmatched_detections,
            classes,
        )

        matches = matches_a + matches_b
        unmatched_tracks = list(set(unmatched_tracks_a + unmatched_tracks_b))
        return matches, unmatched_tracks, unmatched_detections

    def _initiate_track(self, detection, _class, score):
        mean, covariance = self.kf.initiate(detection.to_xyah())
        self.tracks.append(
            Track(
                mean,
                covariance,
                self._next_id,
                self.n_init,
                self.max_age,
                _class,
                score,
                detection.feature,
            )
        )
        # print("new_id:", self._next_id)
        self._next_id += 1
