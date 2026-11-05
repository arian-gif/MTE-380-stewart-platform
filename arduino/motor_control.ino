// Moves the three Stewart-platform servos from serial commands. Pairs with motor_control.py (115200 baud).
//
// Commands, one per line (newline-terminated). <angle> is 0-180 degrees, <motor> is 1-3:
//   M<motor> <angle>   move a motor; replies "OK <motor> <angle>"
//   m<motor> <angle>   same, but no reply (the control loop sends ~90 of these per second)
//   S<motor> <angle>   store the safe angle for a motor (does not move it); replies "OK S <motor> <angle>"
//   s<motor> <angle>   same, no reply
// A bad command always replies "ERR ...". On start-up it prints "READY v2".
//
// Safety timeout: once safe angles are stored for all motors, if no M/m command arrives for
// SAFE_TIMEOUT_MS the servos move to the stored safe angles (the controller sends the level pose as the
// safe pose at start-up). They move again on the next M/m command. The board resets every time the PC
// opens the serial port, which clears the stored angles, so manual tests with motor_control.py never
// trigger the timeout.
//
// Hardware: Adafruit 16-channel PWM & Servo shield (PCA9685, I2C address 0x40) on the Arduino.
// Servos plug into the shield's numbered channels 0-15 (NOT Arduino pins).
// Library: Arduino IDE > Library Manager > "Adafruit PWM Servo Driver Library".

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

const unsigned long SAFE_TIMEOUT_MS = 500;  // no command for this long -> go to the safe angles

// Incoming line, filled one character at a time (no String, so no memory fragmentation on long runs)
const int BUF_SIZE = 32;
char buf[BUF_SIZE];
int bufLen = 0;
bool overflow = false;  // the current line was too long and is being thrown away

float safeAngle[NUM_MOTORS];
bool safeSet[NUM_MOTORS] = {false, false, false};
unsigned long lastCommandMs = 0;
bool inSafe = false;  // the servos are at the safe angles because of the timeout

// Move a motor (1..NUM_MOTORS) to an angle in degrees
void setServo(int motor, float angle) {
  int pulse_us = MIN_US + (int)((MAX_US - MIN_US) * angle / 180.0);
  pwm.writeMicroseconds(MOTOR_CHANNELS[motor - 1], pulse_us);
}

// Handle one complete command line
void handleLine(const char* line) {
  char cmd = line[0];
  bool quiet = (cmd == 'm' || cmd == 's');
  bool isMove = (cmd == 'M' || cmd == 'm');
  bool isSafe = (cmd == 'S' || cmd == 's');
  if (!isMove && !isSafe) {
    Serial.println("ERR bad command");
    return;
  }

  // Parse "<motor> <angle>" after the command letter
  char* end;
  long motor = strtol(line + 1, &end, 10);
  if (end == line + 1 || *end != ' ') {
    Serial.println("ERR bad command");
    return;
  }
  char* angleStart = end + 1;
  float angle = strtod(angleStart, &end);
  if (end == angleStart) {
    Serial.println("ERR bad command");
    return;
  }
  if (motor < 1 || motor > NUM_MOTORS || !(angle >= 0 && angle <= 180)) {  // written to reject NaN too
    Serial.println("ERR out of range");
    return;
  }

  if (isMove) {
    setServo((int)motor, angle);
    inSafe = false;
  } else {
    safeAngle[motor - 1] = angle;
    safeSet[motor - 1] = true;
  }
  lastCommandMs = millis();

  if (!quiet) {
    Serial.print("OK ");
    if (isSafe) Serial.print("S ");
    Serial.print((int)motor);
    Serial.print(" ");
    Serial.println(angle);
  }
}

// If the PC has gone quiet, go to the safe angles (only once safe angles have been stored)
void checkTimeout() {
  if (inSafe) return;
  for (int i = 0; i < NUM_MOTORS; i++) {
    if (!safeSet[i]) return;
  }
  if (millis() - lastCommandMs > SAFE_TIMEOUT_MS) {
    for (int i = 0; i < NUM_MOTORS; i++) {
      setServo(i + 1, safeAngle[i]);
    }
    inSafe = true;
  }
}

void setup() {
  Serial.begin(115200);
  pwm.begin();
  Wire.setClock(400000);                 // faster I2C so three servo updates take about 1 ms
  pwm.setOscillatorFrequency(27000000);  // the PCA9685's internal clock
  pwm.setPWMFreq(50);                    // 50 Hz is the standard servo frame rate
  Serial.println("READY v2");
}

void loop() {
  while (Serial.available()) {
    char c = Serial.read();
    if (c == '\r') continue;
    if (c == '\n') {
      if (overflow) {
        Serial.println("ERR line too long");
      } else if (bufLen > 0) {
        buf[bufLen] = '\0';
        handleLine(buf);
      }
      bufLen = 0;
      overflow = false;
    } else if (bufLen < BUF_SIZE - 1) {
      buf[bufLen++] = c;
    } else {
      overflow = true;
    }
  }
  checkTimeout();
}
