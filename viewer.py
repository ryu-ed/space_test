#!/usr/bin/env python
# license removed for brevity
import rospy
import numpy as np
#check world debug
import sys
from std_msgs.msg import String
from sensor_msgs.msg import CompressedImage
from geometry_msgs.msg import Twist
import pygame
from pygame.locals import *
import cv2
# new line code
#import model
# -----------------------------
video_size = 500, 500
velocity_publisher = rospy.Publisher('space_invader/move', String, queue_size=10)

def key_action():
    vel_msg = Twist()
    keys=pygame.key.get_pressed()
		
    if keys[K_LEFT]:
        return "5"
    if keys[K_UP]:
        return "1"
    if keys[K_RIGHT]:
        return "2"
    return "0"


def callback(ros_data):
    np_arr = np.fromstring(ros_data.data, np.uint8)
    image_np = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    screen = pygame.display.set_mode(video_size)
    surf = pygame.surfarray.make_surface(image_np)
    screen.blit(surf, (0, 0))
    pygame.display.update()
    vel_msg = key_action()
    velocity_publisher.publish(vel_msg)

def main(args):
    '''Initializes and cleanup ros node'''
    rospy.init_node('agent', anonymous=True)
    subscriber = rospy.Subscriber('space_invader/image_raw', CompressedImage, callback)
    try:
        screen = pygame.display.set_mode(video_size)
        vel_msg = key_action()
        velocity_publisher.publish(vel_msg)
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down ROS Gym Image Viewer module")
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main(sys.argv)
#----new line ----
'''    env = gym.make('SpaceInvaders-v0')
    brain = Agent(gamma=0.95, epsilon=1.0,
             alpha=0.003, maxMemorySize=5000,
             replace=None)
    while brain.memCntr < brain.memSize:
        observation = env.reset()
        done = False
        while not done:
            # 0 no action, 1 fire, 2 move right, 3 move left, 4 move right fire, 5 move left fire
            action = env.action_space.sample()
            observation_, reward, done, info = env.step(action)
            if done and info['ale.lives'] == 0:
                reward = -100
            brain.storeTransition(np.mean(observation[15:200,30:125], axis=2), action, reward,
                                np.mean(observation_[15:200,30:125], axis=2))
            observation = observation_
    print('done initializing memory')

    scores = []
    epsHistory = []
    numGames = 50
    batch_size=32
    # uncomment the line below to record every episode.
    env = wrappers.Monitor(env, "tmp/space-invaders-1", video_callable=lambda episode_id: True, force=True)
    for i in range(numGames):
        print('starting game ', i+1, 'epsilon: %.4f' % brain.EPSILON)
        epsHistory.append(brain.EPSILON)
        done = False
        observation = env.reset()
        frames = [np.sum(observation[15:200,30:125], axis=2)]
        score = 0
        lastAction = 0
        while not done:
            if len(frames) == 3:
                action = brain.chooseAction(frames)
                frames = []
            else:
                action = lastAction
            observation_, reward, done, info = env.step(action)
            score += reward
            frames.append(np.sum(observation_[15:200,30:125], axis=2))
            if done and info['ale.lives'] == 0:
                reward = -100
            brain.storeTransition(np.mean(observation[15:200,30:125], axis=2), action, reward,
                                  np.mean(observation_[15:200,30:125], axis=2))
            observation = observation_
            brain.learn(batch_size)
            lastAction = action
            #env.render(
        scores.append(score)
        print('score:',score)
    x = [i+1 for i in range(numGames)]
    fileName = str(numGames) + 'Games' + 'Gamma' + str(brain.GAMMA) + \
               'Alpha' + str(brain.ALPHA) + 'Memory' + str(brain.memSize)+ '.png'
    plotLearning(x, scores, epsHistory, fileName)

#----end new line----'''