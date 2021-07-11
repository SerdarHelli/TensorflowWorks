# -*- coding: utf-8 -*-
"""
Created on Tue Jun 29 14:11:06 2021

@author: serdarhelli
"""

import tensorflow as tf 


class Adaptive_Wing_Loss():
    def __init__(self, alpha=float(2.1), omega=float(5), epsilon=float(1),theta=float(0.5)):   
        self.alpha=alpha
        self.omega=omega
        self.epsilon=epsilon
        self.theta=theta
    def Loss(self,y_true,y_pred):
        A = self.omega * (1/(1+(self.theta/self.epsilon)**(self.alpha-y_true)))*(self.alpha-y_true)*((self.theta/self.epsilon)**(self.alpha-y_true-1))/self.epsilon
        C = self.theta*A - self.omega*tf.math.log(1+(self.theta/self.epsilon)**(self.alpha-y_true))
        loss=tf.where(tf.math.greater_equal(tf.math.abs(y_true-y_pred), self.theta),A*tf.math.abs(y_true-y_pred) - C,self.omega*tf.math.log(1+tf.math.abs((y_true-y_pred)/self.epsilon)**(self.alpha-y_true)))
        return tf.reduce_mean(loss)