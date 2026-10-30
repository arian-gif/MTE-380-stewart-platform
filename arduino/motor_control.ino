// Moves the three Stewart-platform servos from serial commands "M<motor> <angle>".
// Pairs with motor_control.py / platform_control.py (115200 baud).
//
// Hardware: Adafruit 16-channel PWM & Servo shield (PCA9685, I2C address 0x40) on the Arduino.
// Servos plug into the shield's numbered channels 0-15 (NOT Arduino pins).
// Library: Arduino IDE > Library Manager > "Adafruit PWM Servo Driver Library".
//
// TODO: COM port is unknown until the Arduino is plugged in.

#include <Wire.h>
#include <Adafruit_PWMServoDriver.h>

Adafruit_PWMServoDriver pwm = Adafruit_PWMServoDriver(0x40);

const int NUM_MOTORS = 3;
// Shield channel each motor is plugged into (motor 1 = first entry).
// TODO: confirm against the numbers printed on the shield once wired.
const int MOTOR_CHANNELS[NUM_MOTORS] = {1, 2, 3};

// Pulse width for 0 and 180 degrees. TODO: calibrate for the actual servos (datasheet / test),
// 500-2500 us is typical of small hobby servos but some only accept about 1000-2000 us.
const int MIN_US = 500;
const int MAX_US = 2500;

void setup() {
  Serial.begin(115200);
  pwm.begin();
  pwm.setOscillatorFrequency(27000000);  // the PCA9685's internal clock
  pwm.setPWMFreq(50);                    // 50 Hz is the standard servo frame rate
  Serial.println("READY");
}

void loop() {
  if (!Serial.available()) return;
  String line = Serial.readStringUntil('\n');
  line.trim();

  int space = line.indexOf(' ');
  if (line.length() < 3 || line[0] != 'M' || space < 0) {
    Serial.println("ERR bad command");
    return;
  }

  int motor = line.substring(1, space).toInt();
  float angle = line.substring(space + 1).toFloat();
  if (motor < 1 || motor > NUM_MOTORS || angle < 0 || angle > 180) {
    Serial.println("ERR out of range");
    return;
  }

  int pulse_us = MIN_US + (int)((MAX_US - MIN_US) * angle / 180.0);
  pwm.writeMicroseconds(MOTOR_CHANNELS[motor - 1], pulse_us);

  Serial.print("OK ");
  Serial.print(motor);
  Serial.print(" ");
  Serial.println(angle);
}
