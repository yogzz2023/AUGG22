import numpy as np
import math
import csv
import matplotlib.pyplot as plt
import pandas as pd
import mplcursors
from scipy.stats import chi2, multivariate_normal

# Define lists to store results
r = []
el = []
az = []

class CVFilter:
    def __init__(self):
        self.Sf = np.zeros((6, 1))  # Filter state vector
        self.Pf = np.eye(6)  # Filter state covariance matrix
        self.Sp = np.zeros((6, 1))  # Predicted state vector
        self.Pp = np.eye(6)  # Predicted state covariance matrix
        self.plant_noise = 20  # Plant noise covariance
        self.H = np.eye(3, 6)  # Measurement matrix
        self.R = np.eye(3)  # Measurement noise covariance
        self.Meas_Time = 0  # Measured time
        self.prev_Time = 0
        self.Q = np.eye(6)
        self.Phi = np.eye(6)
        self.Z = np.zeros((3, 1)) 
        self.Z1 = np.zeros((3, 1)) # Measurement vector
        self.Z2 = np.zeros((3, 1)) 
        self.first_rep_flag = False
        self.second_rep_flag = False
        self.gate_threshold = 9000.21  # 95% confidence interval for Chi-square distribution with 3 degrees of freedom

    def initialize_filter_state(self, x, y, z, vx, vy, vz, time):
        if not self.first_rep_flag:
            self.Z1 = np.array([[x], [y], [z]])
            self.Sf[0] = x
            self.Sf[1] = y
            self.Sf[2] = z
            self.Meas_Time = time
            self.prev_Time = self.Meas_Time
            self.first_rep_flag = True
        elif self.first_rep_flag and not self.second_rep_flag:
            self.Z2 = np.array([[x], [y], [z]])
            self.prev_Time = self.Meas_Time
            self.Meas_Time = time
            dt = self.Meas_Time - self.prev_Time
            self.vx = (self.Z1[0] - self.Z2[0]) / dt
            self.vy = (self.Z1[1] - self.Z2[1]) / dt
            self.vz = (self.Z1[2] - self.Z2[2]) / dt

            self.Meas_Time = time
            self.second_rep_flag = True
        else:
            self.Z = np.array([[x], [y], [z]])
            self.prev_Time = self.Meas_Time
            self.Meas_Time = time

    def predict_step(self, current_time):
        dt = current_time - self.prev_Time
        T_2 = (dt*dt)/2.0
        T_3 = (dt*dt*dt)/3.0
        self.Phi[0, 3] = dt
        self.Phi[1, 4] = dt
        self.Phi[2, 5] = dt
              
        self.Q[0, 0] = T_3
        self.Q[1, 1] = T_3
        self.Q[2, 2] = T_3
        self.Q[0, 3] = T_2
        self.Q[1, 4] = T_2
        self.Q[2, 5] = T_2
        self.Q[3, 0] = T_2
        self.Q[4, 1] = T_2
        self.Q[5, 2] = T_2
        self.Q[3, 3] = dt
        self.Q[4, 4] = dt
        self.Q[5, 5] = dt
        self.Q = self.Q * self.plant_noise
        self.Sp = np.dot(self.Phi, self.Sf)
        self.Pp = np.dot(np.dot(self.Phi, self.Pf), self.Phi.T) + self.Q
        self.Meas_Time = current_time

    def update_step(self, Z):
        Inn = Z - np.dot(self.H, self.Sp)
        S = np.dot(self.H, np.dot(self.Pp, self.H.T)) + self.R
        K = np.dot(np.dot(self.Pp, self.H.T), np.linalg.inv(S))
        self.Sf = self.Sp + np.dot(K, Inn)
        self.Pf = np.dot(np.eye(6) - np.dot(K, self.H), self.Pp)

    def gating(self, Z):
        Inn = Z - np.dot(self.H, self.Sp)
        S = np.dot(self.H, np.dot(self.Pp, self.H.T)) + self.R
        d2 = np.dot(np.dot(np.transpose(Inn), np.linalg.inv(S)), Inn)
        return d2 < self.gate_threshold

def chi_square_test(track, report, kalman_filter):
    Z = np.array([[report[0]], [report[1]], [report[2]]])
    S = np.dot(kalman_filter.H, np.dot(kalman_filter.Pp, kalman_filter.H.T)) + kalman_filter.R
    Inn = Z - np.dot(kalman_filter.H, kalman_filter.Sp)
    d2 = np.dot(np.dot(Inn.T, np.linalg.inv(S)), Inn)
    return d2 < kalman_filter.gate_threshold

def mahalanobis_distance(track, report, cov_inv):
    diff = np.array(track) - np.array(report)
    distance = np.sqrt(np.dot(np.dot(diff.T, cov_inv), diff))
    return distance

def form_clusters_via_association(tracks, reports, kalman_filter, chi2_threshold):
    association_list = []
    cov_inv = np.linalg.inv(kalman_filter.Pp[:3, :3])  # 3x3 covariance matrix for position only

    # Step 1: Create associations based on Mahalanobis distance or chi-square test
    for i, track in enumerate(tracks):
        for j, report in enumerate(reports):
            distance = mahalanobis_distance(track, report, cov_inv)
            if distance < chi2_threshold:
                association_list.append((i, j))
                print(f"Track {i} associated with Report {j}, Mahalanobis distance: {distance:.4f}")

    # Step 2: Form clusters of tracks and reports
    clusters = []
    while association_list:
        cluster_tracks = set()
        cluster_reports = set()
        stack = [association_list.pop(0)]
        
        while stack:
            track_idx, report_idx = stack.pop()
            cluster_tracks.add(track_idx)
            cluster_reports.add(report_idx)
            new_assoc = [(t, r) for t, r in association_list if t == track_idx or r == report_idx]
            for assoc in new_assoc:
                if assoc not in stack:
                    stack.append(assoc)
            association_list = [assoc for assoc in association_list if assoc not in new_assoc]
        
        # Append clusters as a tuple of two lists (tracks, reports)
        clusters.append((list(cluster_tracks), list(cluster_reports)))

    print("Clusters formed:", clusters)
    return clusters

def form_measurement_groups(measurements, max_time_diff=0.050):
    measurement_groups = []
    current_group = []
    base_time = measurements[0][3]

    for measurement in measurements:
        if measurement[3] - base_time <= max_time_diff:
            current_group.append(measurement)
        else:
            measurement_groups.append(current_group)
            current_group = [measurement]
            base_time = measurement[3]

    if current_group:
        measurement_groups.append(current_group)

    return measurement_groups

def read_measurements_from_csv(file_path):
    measurements = []
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header if exists
        for row in reader:
            # Adjust column indices based on CSV file structure
            mr = float(row[7])  # MR column
            ma = float(row[8])  # MA column
            me = float(row[9])  # ME column
            mt = float(row[10])  # MT column
            x, y, z = sph2cart(ma, me, mr)  # Convert spherical to Cartesian coordinates
            r, az, el = cart2sph(x, y, z)  # Convert Cartesian to spherical coordinates
            measurements.append((r, az, el, mt))
    return measurements

def chi_square_clustering(Z, kalman_filter):
    Inn = Z - np.dot(kalman_filter.H, kalman_filter.Sp)
    S = np.dot(kalman_filter.H, np.dot(kalman_filter.Pp, self.H.T)) + kalman_filter.R
    d2 = np.dot(np.dot(np.transpose(Inn), np.linalg.inv(S)), Inn)
    gate_threshold = kalman_filter.gate_threshold
    print("gate thres:", gate_threshold)
    print("d2", d2)
    return d2 < gate_threshold

def is_valid_hypothesis(hypothesis):
    track_to_report = {}
    report_used = set()
    
    for track_idx, report_idx in hypothesis:
        if report_idx >= 0:  # Skip the case where the report_idx is -1
            if report_idx in report_used:
                return False  # Report is used more than once
            report_used.add(report_idx)
            if track_idx in track_to_report:
                return False  # Track is associated with more than one report
            track_to_report[track_idx] = report_idx
    
    return True

def generate_hypotheses(tracks, reports):
    num_tracks = len(tracks)
    num_reports = len(reports)
    hypotheses = []

    def recursive_hypothesis(track_idx, current_hypothesis):
        if track_idx == num_tracks:
            if is_valid_hypothesis(current_hypothesis):
                hypotheses.append(current_hypothesis.copy())
            return

        for report_idx in range(-1, num_reports):
            current_hypothesis.append((track_idx, report_idx))
            recursive_hypothesis(track_idx + 1, current_hypothesis)
            current_hypothesis.pop()

    recursive_hypothesis(0, [])
    return hypotheses


def compute_hypothesis_likelihood(hypothesis, kalman_filter):
    likelihood = 1.0
    for track_idx, report_idx in hypothesis:
        if report_idx >= 0:  # Valid report index
            Z = np.array([[kalman_filter.Sp[0][0]], [kalman_filter.Sp[1][0]], [kalman_filter.Sp[2][0]]])
            Inn = Z - np.dot(kalman_filter.H, kalman_filter.Sp)
            S = np.dot(kalman_filter.H, np.dot(kalman_filter.Pp, kalman_filter.H.T)) + kalman_filter.R
            likelihood *= np.exp(-0.5 * np.dot(np.dot(Inn.T, np.linalg.inv(S)), Inn))
        else:
            # No report associated with this track, apply a default likelihood or skip
            likelihood *= 1e-6  # Very low likelihood for unmatched tracks

    return likelihood



def jpda(clusters, kalman_filter):
    hypotheses = []
    for cluster_tracks, cluster_reports in clusters:
        hypotheses.extend(generate_hypotheses(cluster_tracks, cluster_reports))

    if not hypotheses:
        return None

    hypothesis_likelihoods = [compute_hypothesis_likelihood(h, kalman_filter) for h in hypotheses]
    total_likelihood = sum(hypothesis_likelihoods)

    if total_likelihood == 0:
        marginal_probabilities = [1.0 / len(hypotheses)] * len(hypotheses)
    else:
        marginal_probabilities = [likelihood / total_likelihood for likelihood in hypothesis_likelihoods]

    best_hypothesis_index = np.argmax(marginal_probabilities)
    best_hypothesis = hypotheses[best_hypothesis_index]

    return best_hypothesis



def sph2cart(az, el, r):
    x = r * np.cos(el * np.pi / 180) * np.sin(az * np.pi / 180)
    y = r * np.cos(el * np.pi / 180) * np.cos(az * np.pi / 180)
    z = r * np.sin(el * np.pi / 180)
    return x, y, z

def cart2sph(x, y, z):
    r = np.sqrt(x**2 + y**2 + z**2)
    el = math.atan2(z, np.sqrt(x**2 + y**2)) * 180 / np.pi
    az = math.atan2(y, x)

    if x > 0.0:
        az = np.pi / 2 - az
    else:
        az = 3 * np.pi / 2 - az

    az = az * 180 / np.pi

    if az < 0.0:
        az = 360 + az

    if az > 360:
        az = az - 360

    return r, az, el

def cart2sph2(x: float, y: float, z: float, filtered_values_csv):
    for i in range(len(filtered_values_csv)):
        r.append(np.sqrt(x[i]**2 + y[i]**2 + z[i]**2))
        el.append(math.atan(z[i] / np.sqrt(x[i]**2 + y[i]**2)) * 180 / 3.14)
        az.append(math.atan(y[i] / x[i]))

        if x[i] > 0.0:                
            az[i] = 3.14 / 2 - az[i]
        else:
            az[i] = 3 * 3.14 / 2 - az[i]       

        az[i] = az[i] * 180 / 3.14 

        if az[i] < 0.0:
            az[i] = 360 + az[i]

        if az[i] > 360:
            az[i] = az[i] - 360   
      
    return r, az, el

def main():
    # File path for measurements CSV
    file_path = 'ttk_84_test.csv'

    # Read measurements from CSV
    measurements = read_measurements_from_csv(file_path)
    
    kalman_filter = CVFilter()
    
    csv_file_predicted = "ttk_84_test.csv"
    df_predicted = pd.read_csv(csv_file_predicted)
    filtered_values_csv = df_predicted[['F_TIM', 'F_X', 'F_Y', 'F_Z']].values
    measured_values_csv = df_predicted[['MT', 'MR', 'MA', 'ME']].values

    A = cart2sph2(filtered_values_csv[:, 1], filtered_values_csv[:, 2], filtered_values_csv[:, 3], filtered_values_csv)
    number = 1000

    result = np.divide(A[0], number)              

    # Form measurement groups based on time intervals less than 50 milliseconds
    measurement_groups = form_measurement_groups(measurements, max_time_diff=0.050)

    t = []
    rnge = []
    azme = []
    elem = []

    # Process each group of measurements
    for group_idx, group in enumerate(measurement_groups):
        print(f"Processing measurement group {group_idx + 1}...")

        # List of tracks and reports (in Cartesian coordinates)
        tracks = []
        reports = []

        for i, (rng, azm, ele, mt) in enumerate(group):
            print(f"Measurement {i + 1}: (az={rng}, el={azm}, r={ele}, t={mt})")
            x, y, z = sph2cart(azm, ele, rng)

            if not kalman_filter.first_rep_flag:
                kalman_filter.initialize_filter_state(x, y, z, 0, 0, 0, mt)
                print("Initialized Filter state:", kalman_filter.Sf.flatten())
                continue

            elif kalman_filter.first_rep_flag and not kalman_filter.second_rep_flag:
                kalman_filter.initialize_filter_state(x, y, z, 0, 0, 0, mt)
                print("Initialized Filter state 2nd M:", kalman_filter.Sf.flatten())
                continue

            kalman_filter.predict_step(mt)
            reports.append((x, y, z))  # Treat the measurements as reports

            # Use the Kalman filter state as the track
            tracks.append(kalman_filter.Sf[:3].flatten())

        # Step 1: Form clusters using the association logic
        clusters = form_clusters_via_association(tracks, reports, kalman_filter, chi2_threshold=kalman_filter.gate_threshold)

        # Step 2: For each cluster, perform JPDA and update the Kalman filter
        for cluster_tracks, cluster_reports in clusters:
            if cluster_tracks and cluster_reports:
                # Pass both cluster_tracks and cluster_reports to generate_hypotheses
                best_hypothesis = jpda([(cluster_tracks, cluster_reports)], kalman_filter)
                print("check best hypo ",best_hypothesis)
                if best_hypothesis is not None:
                    Z = np.array([[kalman_filter.Sp[0][0]], [kalman_filter.Sp[1][0]], [kalman_filter.Sp[2][0]]])
                    kalman_filter.update_step(Z)
                    print("Updated filter state:", kalman_filter.Sf.flatten())

                    # Save the updated state for plotting
                    r_val, az_val, el_val = cart2sph(kalman_filter.Sf[0], kalman_filter.Sf[1], kalman_filter.Sf[2])
                    rnge.append(r_val)
                    azme.append(az_val)
                    elem.append(el_val)
                    t.append(mt)

    # Plotting remains unchanged
    plt.figure(figsize=(12, 6))
    plt.subplot(facecolor="white")
    plt.scatter(t, rnge, label='filtered range (code)', color='green', marker='*')
    plt.scatter(filtered_values_csv[:, 0], result, label='filtered range (track id 31)', color='red', marker='*')
    plt.scatter(measured_values_csv[:, 0], measured_values_csv[:, 1], label='measured range (code)', color='blue', marker='o')
    plt.xlabel('Time', color='black')
    plt.ylabel('Range (r)', color='black')
    plt.title('Range vs. Time', color='black')
    plt.grid(color='gray', linestyle='--')
    plt.legend()
    plt.tight_layout()
    mplcursors.cursor(hover=True)
    plt.show()

    # Plot azimuth (az) vs. time
    plt.figure(figsize=(12, 6))
    plt.subplot(facecolor="white")
    plt.scatter(t, azme, label='filtered azimuth (code)', color='green', marker='*')
    plt.scatter(filtered_values_csv[:, 0], A[1], label='filtered azimuth (track id 31)', color='red', marker='*')
    plt.scatter(measured_values_csv[:, 0], measured_values_csv[:, 2], label='measured azimuth (code)', color='blue', marker='o')
    plt.xlabel('Time', color='black')
    plt.ylabel('Azimuth (az)', color='black')
    plt.title('Azimuth vs. Time', color='black')
    plt.grid(color='gray', linestyle='--')
    plt.legend()
    plt.tight_layout()
    mplcursors.cursor(hover=True)
    plt.show()

    # Plot elevation (el) vs. time
    plt.figure(figsize=(12, 6))
    plt.subplot(facecolor="white")
    plt.scatter(t, elem, label='filtered elevation (code)', color='green', marker='*')
    plt.scatter(filtered_values_csv[:, 0], A[2], label='filtered elevation (track id 31)', color='red', marker='*')
    plt.scatter(measured_values_csv[:, 0], measured_values_csv[:, 3], label='measured elevation (code)', color='blue', marker='o')
    plt.xlabel('Time', color='black')
    plt.ylabel('Elevation (el)', color='black')
    plt.title('Elevation vs. Time', color='black')
    plt.grid(color='gray', linestyle='--')
    plt.legend()
    plt.tight_layout()
    mplcursors.cursor(hover=True)
    plt.show()

if __name__ == "__main__":
    main()

