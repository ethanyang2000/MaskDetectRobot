#include "Move.h"
#include <Arduino.h>
Move::Move (int In_PWML, int In_CRLL, int In_SLPL, int In_PWMR, int In_CRLR, int In_SLPR)
{
    PWML = In_PWML;
    PWMR = In_PWMR;
    CRLL = In_CRLL;
    CRLR = In_CRLR;
    SLPL = In_SLPL;
    SLPR = In_SLPR;
}

Move::Move(const Move& In_Move) {
    PWML = In_Move.PWML;
    PWMR = In_Move.PWMR;
    CRLL = In_Move.CRLL;
    CRLR = In_Move.CRLR;
    SLPL = In_Move.SLPL;
    SLPR = In_Move.SLPR;
}

Move::~Move(){

}

void Move::InitMove() {
    pinMode(PWML, OUTPUT);
    pinMode(CRLL, OUTPUT);
    pinMode(SLPL, OUTPUT);
    pinMode(PWMR, OUTPUT);
    pinMode(CRLR, OUTPUT);
    pinMode(SLPR, OUTPUT);

    digitalWrite(CRLL, LOW);
    digitalWrite(CRLR, LOW);
    digitalWrite(SLPL, HIGH);
    digitalWrite(SLPR, HIGH);
}

void Move::turnleft() {
    digitalWrite(SLPR, HIGH);
    digitalWrite(SLPL, HIGH);
    digitalWrite(CRLL, HIGH);
    digitalWrite(CRLR, LOW);
    analogWrite(PWML, 200);
    analogWrite(PWMR, 205);
    delay(5);
    digitalWrite(PWML, LOW);
    digitalWrite(PWMR, LOW);
    delay(10);
    analogWrite(PWML, 200);
    analogWrite(PWMR, 205);
}

void Move::turnright() {
    digitalWrite(SLPL, HIGH);
    digitalWrite(SLPR, HIGH);
    digitalWrite(CRLL, LOW);
    digitalWrite(CRLR, HIGH);
    analogWrite(PWML, 200);
    analogWrite(PWMR, 205);
    delay(5);
    digitalWrite(PWML, LOW);
    digitalWrite(PWMR, LOW);
    delay(10);
    analogWrite(PWML, 200);
    analogWrite(PWMR, 205);
}

void Move::sleep() {
    digitalWrite(SLPL, LOW);
    digitalWrite(SLPR, LOW);
}

void Move::fl(){
    digitalWrite(SLPL, HIGH);
    digitalWrite(SLPR, HIGH);
    digitalWrite(CRLL, LOW);
    digitalWrite(CRLR, LOW);
    analogWrite(PWML, 145);
    analogWrite(PWMR, 150);
    //Serial.println("l1");
    delay(7);
    digitalWrite(PWML, LOW);
    digitalWrite(PWMR, LOW);
    //Serial.println("l2");
    delay(8);
    analogWrite(PWML, 145);
    analogWrite(PWMR, 150);
    //Serial.println("l3");
}

void Move::fr(){
    digitalWrite(SLPL, HIGH);
    digitalWrite(SLPR, HIGH);
    digitalWrite(CRLL, LOW);
    digitalWrite(CRLR, LOW);
    analogWrite(PWML, 148);
    analogWrite(PWMR, 145);
    //Serial.println("r1");
    delay(8);
    digitalWrite(PWMR,LOW);
    //Serial.println("r2");
    //delay(3);
    digitalWrite(PWML,LOW);
    //Serial.println("r3");
    delay(8);
    analogWrite(PWML, 148);
    analogWrite(PWMR, 145);
}

void Move::bl(){
    digitalWrite(SLPL, HIGH);
    digitalWrite(SLPR, HIGH);
    digitalWrite(CRLL, HIGH);
    digitalWrite(CRLR, HIGH);
    analogWrite(PWML, 145);
    analogWrite(PWMR, 155);
    //Serial.println("l1");
    delay(8);
    digitalWrite(PWML, LOW);
    digitalWrite(PWMR, LOW);
    //Serial.println("l2");
    delay(8);
    analogWrite(PWML, 145);
    analogWrite(PWMR, 155);
}

void Move::br(){
    digitalWrite(SLPL, HIGH);
    digitalWrite(SLPR, HIGH);
    digitalWrite(CRLL, HIGH);
    digitalWrite(CRLR, HIGH);
    analogWrite(PWML, 145);
    analogWrite(PWMR, 145);
    //Serial.println("r1");
    delay(8);
    digitalWrite(PWMR,LOW);
    //Serial.println("r2");
    //delay(3);
    digitalWrite(PWML,LOW);
    //Serial.println("r3");
    delay(8);
    analogWrite(PWML, 145);
    analogWrite(PWMR, 145);
 
}
void Move::forwardleft() {
    digitalWrite(SLPL, HIGH);
    digitalWrite(SLPR, HIGH);
    digitalWrite(CRLL, LOW);
    digitalWrite(CRLR, LOW);
    analogWrite(PWML, 200);
    analogWrite(PWMR, 205);
    //Serial.println("l1");
    delay(7);
    digitalWrite(PWML, LOW);
    digitalWrite(PWMR, LOW);
    //Serial.println("l2");
    delay(8);
    analogWrite(PWML, 200);
    analogWrite(PWMR, 205);
    //Serial.println("l3");
}

void Move::forwardright() {
    digitalWrite(SLPL, HIGH);
    digitalWrite(SLPR, HIGH);
    digitalWrite(CRLL, LOW);
    digitalWrite(CRLR, LOW);
    analogWrite(PWML, 200);
    analogWrite(PWMR, 200);
    //Serial.println("r1");
    delay(8);
    digitalWrite(PWMR,LOW);
    //Serial.println("r2");
    //delay(3);
    digitalWrite(PWML,LOW);
    //Serial.println("r3");
    delay(8);
    analogWrite(PWML, 200);
    analogWrite(PWMR, 200);
    //Serial.println("r4");
}

void Move::backwardleft() {
    digitalWrite(SLPL, HIGH);
    digitalWrite(SLPR, HIGH);
    digitalWrite(CRLL, HIGH);
    digitalWrite(CRLR, HIGH);
    analogWrite(PWML, 200);
    analogWrite(PWMR, 210);
    //Serial.println("l1");
    delay(8);
    digitalWrite(PWML, LOW);
    digitalWrite(PWMR, LOW);
    //Serial.println("l2");
    delay(8);
    analogWrite(PWML, 200);
    analogWrite(PWMR, 210);
    //Serial.println("l3");
}

void Move::backwardright() {
    
    digitalWrite(SLPL, HIGH);
    digitalWrite(SLPR, HIGH);
    digitalWrite(CRLL, HIGH);
    digitalWrite(CRLR, HIGH);
    analogWrite(PWML, 200);
    analogWrite(PWMR, 205);
    //Serial.println("r1");
    delay(8);
    digitalWrite(PWMR,LOW);
    //Serial.println("r2");
    //delay(3);
    digitalWrite(PWML,LOW);
    //Serial.println("r3");
    delay(8);
    analogWrite(PWML, 200);
    analogWrite(PWMR, 205);
    //Serial.println("r4");
}