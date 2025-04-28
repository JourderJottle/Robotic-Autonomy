// Setup Arduino
#if defined(ARDUINO) && ARDUINO >= 100
  #include "Arduino.h"
#else
  #include <WProgram.h>
#endif

// Setup Dependencies
#include "VNH3SP30.h"
#include <SPI.h>
#include <ros.h>
#include <geometry_msgs/Twist.h>
#include <std_msgs/String.h>

// Define Motor Control Pins
// Left Side
#define M1SpeedPin 3
#define M1InAPin 2
#define M1InBPin 4

// Right Side
#define M2SpeedPin 6
#define M2InAPin 5
#define M2InBPin 7

#define LED1 9
#define LED2 10
#define LED3 11
#define LED4 12
#define LED5 13

// Setup Motors
VNH3SP30 MotorLeft(M1SpeedPin, M1InAPin, M1InBPin);
VNH3SP30 MotorRight(M2SpeedPin, M2InAPin, M2InBPin);

// Setup ROS
ros::NodeHandle nh;
std_msgs::String str_msg;

// Setup Variables
float dtheta = 0, dx = 0, omega_left = 0, omega_right = 0;
bool leftForward, rightForward;

// Calibration Variables
float right_lin_cal = 1.300;
float left_lin_cal = 1.000;
float right_ang_cal = 1.150;
float left_ang_cal = 1.000;

float theta_scale = 0.1;
float linear_scale = 0.2;
float throttle_threshold = 0.04;

float wheel_base = 0.45; // In meters
float wheel_radius = 0.15/2;
float max_speed = 1.4; 

float speed_scalar = 1;

void controlCallback( const geometry_msgs::Twist& twist_msg){
  // Scale dx & dtheta
  dx =  twist_msg.angular.z * theta_scale;
  dtheta = twist_msg.linear.x * linear_scale;
	
  // Calculate omega_left & omega_right and calibrating with scalers
  
  omega_right = min(max((-dx*right_ang_cal - right_lin_cal*dtheta * wheel_base / 2) / wheel_radius, -1), 1)/speed_scalar; // If commands are flipped, flip these variables. 
  omega_left = min(max((-dx*left_ang_cal + left_lin_cal*dtheta * wheel_base / 2) / wheel_radius, -1), 1)/speed_scalar;


  // Check Minimum Throttle
  // Left
  if(-throttle_threshold > omega_left || omega_left > throttle_threshold){
    MotorLeft.Throttle(omega_left);
  } else{
    MotorLeft.Stop(); 
  }
  // Right
  if(-throttle_threshold > omega_right || omega_right > throttle_threshold){
    MotorRight.Throttle(omega_right);
  } else {
    MotorRight.Stop();
  }

}

ros::Subscriber<geometry_msgs::Twist> control_sub("/cmd_vel", &controlCallback);

void setup() {
  Serial.begin(57600);

  // Setup Pin Modes
  pinMode(LED_BUILTIN, OUTPUT);
  pinMode(LED1, OUTPUT);
  pinMode(LED2, OUTPUT);
  pinMode(LED3, OUTPUT);
  pinMode(LED4, OUTPUT);
  pinMode(LED5, OUTPUT);
  pinMode(8, INPUT);

  // Setup Motors
  MotorLeft.Stop();
  MotorRight.Stop();
  nh.initNode();
  nh.subscribe(control_sub);
  
}

void loop() {
  
  nh.spinOnce();
  if (dtheta == 0 || dx == 0) {
    digitalWrite(LED_BUILTIN, HIGH);
  }
  delay(5);
  digitalWrite(LED_BUILTIN, LOW);
  //String str = String(omega_right);
  //int str_len = str.length() + 1;
  //char char_arr[str_len];
  //str.toCharArray(char_arr, str_len);
  //nh.loginfo(char_arr);
}