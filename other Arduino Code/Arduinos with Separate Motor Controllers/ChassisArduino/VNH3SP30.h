#ifndef VNH3SP30_H
#define VNH3SP30_H


#include <Arduino.h>

class VNH3SP30{
      
      public:
              VNH3SP30(int _PWMPIN, int _INAPIN, int _INBPIN);
              ~VNH3SP30();
      
              void Throttle(float Throttle); //Value between -1 and 1
              void Stop() ; 
               
                                     
      private:
			  int PWMPIN, INAPIN, INBPIN;
              void SetPWMA(byte Value); 
      
      
};

#endif

