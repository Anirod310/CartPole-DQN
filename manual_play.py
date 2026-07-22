import pygame
import gymnasium as gym

env = gym.make("CartPole-v1", render_mode="human")

env.reset()
env.render()

quit_game = False
action = 0
clock = pygame.time.Clock()

while not quit_game:
    keys = pygame.key.get_pressed()
    action_taken = False
    
    if keys[pygame.K_RIGHT]:
        action = 1
        action_taken = True
    elif keys[pygame.K_LEFT]:
        action = 0
        action_taken = True

    if action_taken:
        state, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
                env.reset()
                action = 0
   
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            quit_game = True

    clock.tick(30)

env.close()