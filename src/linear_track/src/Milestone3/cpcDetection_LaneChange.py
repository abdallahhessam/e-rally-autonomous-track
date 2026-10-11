#!/usr/bin/env python3

import rospy, numpy as np, cv2, torch, copy
from std_msgs.msg import Float64
from nav_msgs.msg import Odometry
from tf.transformations import euler_from_quaternion
import queue
import math
from cv_bridge import CvBridge
from ultralytics import YOLO
from sensor_msgs.msg import Image

class ObjectDetection:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.clss = ["car", "cone", "pedestrian"]
        self.model = YOLO("/home/morad/Downloads/E-Rally/Milestone_1/f_ms/src/linear_track/src/CPC_model.pt")
        self.bridge = CvBridge()
        self.original_dict = {"car": [], "cone": [], "pedestrian": []}
        self.known_height_m = {"car": 1.55, "cone": 0.66, "pedestrian": 1.65}
        self.colors = [(238, 75, 43), (170, 255, 0), (0, 150, 255)]

        #track specific variables
        self.lane_change_ticket = False

    def detect(self, msg):
        self.detected = copy.deepcopy(self.original_dict)
        image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        results = self.model(image)
        boxes = results[0].boxes.cpu().numpy()  # Bounding boxes
        if len(boxes) > 0:
            for box in boxes:
                data = box.data[0]
                obj_name = self.clss[data[-1].astype(int)]
                confidence = data[-2].astype(float)
                x1, y1, x2, y2 = data[:4].astype(int)

                if confidence > 0.7:
                    object_in_shape = self.isInside(480, 240, 45, 480, 915, 480, x2 if x2 < 480 else x1, y2)
                    if object_in_shape:
                        distance = max(((( (y2-y1) / self.known_height_m[obj_name] ) - 157.5) / -10.5), 0)
                        self.detected[obj_name].append(distance)
                    else:
                        distance = "N/A"
                    self.plot((image, data[-1].astype(int), (x1, y1, x2, y2), distance))
                else:
                    self.plot((image, None))
        else:
            self.plot((image, None))

        if self.detected["car"] and min(self.detected["car"]) <= 12.0 and self.lane_change_ticket == False:
            self.lane_change_ticket = True
            vCont.raise_change_lane_flag()
            self.lane_change_ticket = False
                
        if self.detected["pedestrian"]:
            vCont.pedestrian_distance.put(min(self.detected["pedestrian"]))
        else:
            vCont.pedestrian_distance.put(500)

    def plot(self, data):
        image = data[0]
        id = data[1]
        if id is not None:
            pos = data[2]
            try:
                dis = round(data[3], 2)
            except:
                dis = "N/A"
            cv2.rectangle(image, (pos[0], pos[1]), (pos[2], pos[3]), self.colors[id], 2)
            cv2.putText(image, f"dist: {dis}", (pos[0], pos[1]-5), cv2.FONT_ITALIC, 0.5, self.colors[id], 2)
        cv2.imshow("Image", image)
        cv2.waitKey(1)

    def isInside(self, x1, y1, x2, y2, x3, y3, x, y):
        # Calculate area of the main triangle
        A = abs(x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)) / 2.0
        
        # Calculate area of the sub-triangles
        A1 = abs(x * (y2 - y3) + x2 * (y3 - y) + x3 * (y - y2)) / 2.0
        A2 = abs(x1 * (y - y3) + x * (y3 - y1) + x3 * (y1 - y)) / 2.0
        A3 = abs(x1 * (y2 - y) + x2 * (y - y1) + x * (y1 - y2)) / 2.0
        
        # Check if the point is inside the triangle
        return A == A1 + A2 + A3
    
class Kalman_Filter:
    def __init__(self, r1, r2, r3):
        # Define the Kalman filter parameters and variables
        self.dt = 0.1  # Time step
        self.A = np.array([[1, self.dt, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0, 0],
                    [0, 0, 1, self.dt, 0, 0],
                    [0, 0, 0, 1, 0, 0],
                    [0, 0, 0, 0, 1, self.dt],
                    [0, 0, 0, 0, 0, 1]])  # State transition matrix
        self.B = np.zeros((6, 3))  # Control input matrix
        self.C = np.array([[1, 0, 0, 0, 0, 0],
                    [0, 0, 1, 0, 0, 0],
                    [0, 0, 0, 0, 1, 0]])  # Measurement matrix
        self.Q = np.eye(6) * 0.01  # Process noise covariance matrix
        self.R = np.array([[r1, 0, 0],
                            [0, r2, 0],
                            [0, 0, r3]])  # Measurement noise covariance matrix
        self.U = np.array([[0],[0.2],[0]])
        self.P = np.eye(6) * 10000  # Initial covariance matrix
        self.X = np.zeros((6, 1))  # Initial state estimate
        self.filtered_odom = []

        # Kalman filter prediction step
    def kf_prediction(self):
        self.Xpredicted = np.matmul(self.A, self.X) + np.matmul(self.B, self.U)			##Predicted state vector			            (6x1)
        self.Ppredicted = np.matmul(self.A, np.matmul(self.P, np.transpose(self.A))) + self.Q	##Predicted error co-variance matrix    (6x6)	
        return (self.Xpredicted, self.Ppredicted)
    def kf_correction(self, Z):			
        self.CTrans = np.transpose(self.C)				
        self.num = np.matmul(self.Ppredicted, self.CTrans)		##Numerature of Kalman gain equation
        self.den1 = np.matmul(self.C, self.Ppredicted) 		##CMatrix * PMatrix
        self.den2 = np.matmul(self.den1, self.CTrans) + self.R  	##CMatrix * PMatrix * CMatrix^T _+ R
        self.den = np.matrix(self.den2)  			##Place denemrature in matrix form  
        self.deninv = self.den.getI()				##(CMatrix * PMatrix * CMatrix^T _+ R) Inverse 	
        self.KGain = np.matmul(self.num, self.deninv) 		##Calculate the Kalman gain
        self.Xfiltered = self.Xpredicted + np.matmul(self.KGain, (Z - np.matmul(self.C, self.Xpredicted)))	##Estimated updated state vector
        self.Pfiltered = self.Ppredicted - np.matmul(self.KGain, np.matmul(self.C, self.Ppredicted)) 	##Estimated updated error co-variance matrix
        return (self.Xfiltered, self.Pfiltered)

class PIDController:
    def __init__(self, kp, ki, kd, target_distance, inverse=False):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.target_distance = target_distance
        self.cumulative_error = 0
        self.last_error = 0
        self.last_time = rospy.Time.now()
        self.inverse = inverse

    def compute(self, current_distance):
        if not self.inverse:
            error = self.target_distance - current_distance
        else:
            error = current_distance - self.target_distance
        current_time = rospy.Time.now()
        delta_time = (current_time - self.last_time).to_sec() if (current_time - self.last_time).to_sec() != 0 else 0.1

        self.cumulative_error += error * delta_time
        derivative = (error - self.last_error) / delta_time

        output = self.kp * error + self.ki * self.cumulative_error + self.kd * derivative

        self.last_error = error
        self.last_time = current_time

        return max(min(output, 1.0), 0.0)  # Limit output to between 0 and max allowable velocity

class VehicleController:
    def __init__(self, req_distance):
        self.pub_vel    = rospy.Publisher('/cmd_vel', Float64, queue_size=10, latch=True)
        self.pub_steer  = rospy.Publisher('/SteeringAngle', Float64, queue_size=10, latch=True)
        self.brk        = rospy.Publisher('/brakes', Float64, queue_size=10, latch=True)
        self.rate = rospy.Rate(100)
        
        self.req_distance = req_distance
        self.current_distance = 0
        self.current_orientation = 0
        self.brk.publish(0.0)
        self.pub_vel.publish(0.0)

        self.controllerSpeed = 1.0
        self.pedestrian_PID = PIDController(kp=0.0823, ki=0.000015, kd=0.45, target_distance=6.5, inverse=True)
        self.pedestrian_distance = queue.Queue()
        self.pedestrian_distance.put(500)
        
        self.change_lane_flag = False

    def track_path(self):
        self.pid_straight = PIDController(kp=0.05, ki=0.000065, kd=0.045, target_distance=self.req_distance)
        while not rospy.is_shutdown():
            result = self.pid_straight.compute(self.current_distance)
            self.controllerSpeed = result
            self.publish_velocity()

            if self.change_lane_flag:
                self.Change_lane('left')
                self.Change_lane('right')
                self.change_lane_flag = False

            if self.current_distance >= self.req_distance-0.3:
                self.brk.publish(min(1-result, 0.5))
                rospy.loginfo("Stopped")
                if self.current_distance >= self.req_distance-0.2:
                    break

            self.rate.sleep()
    
    def raise_change_lane_flag(self):
        self.change_lane_flag = True

    def update_odom_values(self, current_distance, current_orientation):
        self.current_distance = current_distance
        self.current_orientation = current_orientation

    def Change_lane(self, lane, angle = 32):
        max_steering_angle = 30
        if lane == 'left':
            Target_angle = self.current_orientation + angle
            inverse = False
            const = 1
        elif lane == 'right':
            Target_angle = self.current_orientation - angle
            inverse = True
            const = -1
        else:
            raise ValueError("Invalid lane value")

        self.pid_lane_change = PIDController(kp=0.0375, ki=0.001, kd=0.0675, target_distance=Target_angle, inverse=inverse)
        rospy.loginfo((self.current_orientation < Target_angle) if not inverse else (self.current_orientation > Target_angle))
        while (self.current_orientation < Target_angle) if not inverse else (self.current_orientation > Target_angle):
            self.publish_velocity()
            steering_angle = self.pid_lane_change.compute(self.current_orientation)*max_steering_angle
            self.pub_steer.publish(steering_angle * const)
            rospy.loginfo(f"{lane} turn: {self.current_orientation}, {Target_angle} -- {steering_angle}")
            self.rate.sleep()
        self.pub_steer.publish(0)

    def publish_velocity(self):
        controller_speed = 0.22*self.controllerSpeed
        detector_speed = self.pedestrian_PID.compute(list(self.pedestrian_distance.queue)[-1])  * 0.22
        rospy.loginfo(f"Controller Speed: {controller_speed/0.22}, Detector Speed: {detector_speed/0.22}")
        
        if controller_speed < detector_speed:
            self.brk.publish(0.0)
            self.pub_vel.publish(controller_speed)
        else:
            if detector_speed > 0.01:
                self.brk.publish(0.0)
                self.pub_vel.publish(detector_speed)
            else:
                self.brk.publish(1.0)
                self.pub_vel.publish(0.0)

class Odom_Manager:
    def __init__(self):
        self.last_z_orien = None
        self.orientation_multiplier = 0

    def odom_callback(self, odom_msg):
        position = odom_msg.pose.pose.position
        orientation = odom_msg.pose.pose.orientation

        noisy_position = np.array([[position.x], [position.y], [position.z]])
        noisy_orientation = np.array([[orientation.x], [orientation.y], [orientation.z]])

        Position_KF = Kalman_Filter(1.1345, 0.8245, 0.01)
        Position_KF.kf_prediction()    # Kalman filter prediction step
        X_filteredPos , P_filteredPos = Position_KF.kf_correction(noisy_position)    # Kalman filter correction step

        Position_KF.X = X_filteredPos
        Position_KF.P = P_filteredPos
        
        orientation_KF = Kalman_Filter(1.1345, 0.8245, 0.01)
        orientation_KF.kf_prediction()    # Kalman filter prediction step
        X_filteredOri , P_filteredOri = orientation_KF.kf_correction(noisy_orientation)    # Kalman filter correction step
        
        orientation_KF.X = X_filteredOri
        orientation_KF.P = P_filteredOri
        
        current_position = np.array([position.x, position.y])
        filtered_ori     = np.array((orientation.x, orientation.y, orientation.z, orientation.w))
        z_orien = euler_from_quaternion(filtered_ori)[2] * (180/math.pi)

        if z_orien < 0:
            z_orien = 360 + z_orien     # z_orien is a range from (0, 360)
        if self.last_z_orien is not None:
            if self.last_z_orien > 355 and z_orien < 5:
                self.orientation_multiplier += 1    
            elif self.last_z_orien < 5 and z_orien > 355:
                self.orientation_multiplier -= 1

        total_z_orien = z_orien + (self.orientation_multiplier * 360)   # total_z_orien is a range from (-inf, inf)

        vCont.update_odom_values(current_position[1], total_z_orien)
        rospy.loginfo(f"z_ori: {z_orien}, total: {total_z_orien}")
        self.last_z_orien = z_orien     # last_z_orien is a range from (0, 360)

if __name__ == '__main__':
    DT = 0.1
    rospy.init_node('velocity_publisher', disable_signals=True)
    vCont = VehicleController(req_distance=75.0)
    
    detector = ObjectDetection()
    rospy.Subscriber("/image", Image, detector.detect)
    odom_class = Odom_Manager()
    rospy.Subscriber('/odom', Odometry, odom_class.odom_callback)
    
    vCont.track_path()  

