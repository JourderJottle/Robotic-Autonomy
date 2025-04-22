#include "VNH3SP30.h"
//-- public methods --//
//<<constructor>>
VNH3SP30::VNH3SP30(int _PWMPIN, int _INAPIN, int _INBPIN){
  PWMPIN = _PWMPIN;
  INAPIN = _INAPIN;
  INBPIN = _INBPIN;
  pinMode(PWMPIN, OUTPUT);
  pinMode(INAPIN, OUTPUT);
  pinMode(INBPIN, OUTPUT);
}
VNH3SP30::~VNH3SP30(){};
void VNH3SP30::Throttle(float Throttle) {
if (Throttle>0){
  digitalWrite(INAPIN, HIGH);
  digitalWrite(INBPIN, LOW);
  SetPWMA(abs(Throttle)*255);
}
if (Throttle<0){
  digitalWrite(INAPIN, LOW);
  digitalWrite(INBPIN, HIGH);
  SetPWMA(abs(Throttle)*255);
}
// else{
// SetPWMA(0);
// digitalWrite(INAPIN, LOW);
// digitalWrite(INBPIN, LOW);
// }
//
}
void VNH3SP30::Stop() {
  SetPWMA(0);
  digitalWrite(INAPIN, LOW);
  digitalWrite(INBPIN, LOW);
}
//-- private methods --//
void VNH3SP30::SetPWMA(byte Value) {
  analogWrite(PWMPIN, Value);
}