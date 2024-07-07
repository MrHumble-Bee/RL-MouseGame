import pygame
import numpy as np
import random
import time
import os

# Initialize Pygame
pygame.init()
pygame.mixer.init() 

# Constants
GRID_SIZE = 20
CELL_SIZE = 30
GRID_WIDTH = GRID_SIZE * CELL_SIZE
SCREEN_WIDTH = GRID_WIDTH * 2 + CELL_SIZE
SLIDER_HEIGHT = 30
TOTAL_HEIGHT = GRID_WIDTH + 7 * SLIDER_HEIGHT  
FILE_PATH =  os.path.dirname(os.path.abspath(__file__))

# Image Paths
MOUSE_IMAGE_PATH = os.path.join(FILE_PATH, "media/mouse.jpeg")
CHEESE_IMAGE_PATH = os.path.join(FILE_PATH, "media/cheese.png")
SOUNDTRACK_PATH = os.path.join(FILE_PATH, "media/8-bit-music.mp3")

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
GRAY = (200, 200, 200)

# Create the screen
screen = pygame.display.set_mode((SCREEN_WIDTH, TOTAL_HEIGHT))
pygame.display.set_caption("Reinforcement Learning Grid World")

# Load Images
mouse_img = pygame.image.load(MOUSE_IMAGE_PATH)
cheese_img = pygame.image.load(CHEESE_IMAGE_PATH)
mouse_img = pygame.transform.scale(mouse_img, (CELL_SIZE, CELL_SIZE))
cheese_img = pygame.transform.scale(cheese_img, (CELL_SIZE, CELL_SIZE))

# Load and play soundtrack
pygame.mixer.music.load(SOUNDTRACK_PATH)
pygame.mixer.music.play(-1)

class GridWorld:
    def __init__(self):
        self.mouse_pos = (0, 0)
        self.cheese_pos = (GRID_SIZE-1, GRID_SIZE-1)
        self.walls = set()
        self.start_time = time.time()

    def reset(self):
        self.mouse_pos = (0, 0)
        self.start_time = time.time()
        return self.mouse_pos

    def step(self, action):
        x, y = self.mouse_pos
        if action == 0 and x > 0:  # Up
            x -= 1
        elif action == 1 and x < GRID_SIZE - 1:  # Down
            x += 1
        elif action == 2 and y > 0:  # Left
            y -= 1
        elif action == 3 and y < GRID_SIZE - 1:  # Right
            y += 1

        if (x, y) not in self.walls:
            self.mouse_pos = (x, y)

        done = self.mouse_pos == self.cheese_pos or (time.time() - self.start_time) > 15
        return self.mouse_pos, done

    def add_wall(self, pos):
        if pos != (0, 0) and pos != self.cheese_pos:
            self.walls.add(pos)

    def remove_wall(self, pos):
        self.walls.discard(pos)

class QLearningAgent:
    def __init__(self, state_size, action_size):
        self.q_table = np.zeros((state_size, state_size, action_size))
        self.current_reward = 0

    def get_action(self, state, epsilon):
        if random.random() < epsilon:
            return random.randint(0, 3)
        else:
            return np.argmax(self.q_table[state])

    def update_q_table(self, state, action, reward, next_state, done, alpha, gamma):
        current_q = self.q_table[state][action]
        if done:
            target_q = reward
        else:
            target_q = reward + gamma * np.max(self.q_table[next_state])
        self.q_table[state][action] += alpha * (target_q - current_q)
        self.current_reward = reward

class Slider:
    def __init__(self, x, y, width, height, min_val, max_val, initial_val, label):
        SHIFT_X = 300
        self.rect = pygame.Rect(x + SHIFT_X, y, width - SHIFT_X, height)
        self.min_val = min_val
        self.max_val = max_val
        self.val = initial_val
        self.clicked = False
        self.label = label
        self.label_x = x

    def draw(self, screen):
        pygame.draw.rect(screen, WHITE, self.rect)
        pos = self.rect.x + int((self.val - self.min_val) / (self.max_val - self.min_val) * self.rect.width)
        pygame.draw.rect(screen, BLUE, (pos - 5, self.rect.y, 10, self.rect.height))

        font = pygame.font.Font(None, 24)
        text = font.render(f"{self.label}: {self.val:.2f}", True, BLACK)
        screen.blit(text, (self.label_x, self.rect.y))

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            if self.rect.collidepoint(event.pos):
                self.clicked = True
        elif event.type == pygame.MOUSEBUTTONUP:
            self.clicked = False
        elif event.type == pygame.MOUSEMOTION and self.clicked:
            self.val = (event.pos[0] - self.rect.x) / self.rect.width * (self.max_val - self.min_val) + self.min_val
            self.val = max(self.min_val, min(self.max_val, self.val))

def draw_grid(surface, grid_world, offset_x=0):
    for x, y in grid_world.walls:
        rect = pygame.Rect(offset_x + y * CELL_SIZE, x * CELL_SIZE, CELL_SIZE, CELL_SIZE)
        pygame.draw.rect(surface, BLACK, rect)

    mouse_rect = pygame.Rect(offset_x + grid_world.mouse_pos[1] * CELL_SIZE, grid_world.mouse_pos[0] * CELL_SIZE, CELL_SIZE, CELL_SIZE)
    surface.blit(mouse_img, mouse_rect)

    cheese_rect = pygame.Rect(offset_x + grid_world.cheese_pos[1] * CELL_SIZE, grid_world.cheese_pos[0] * CELL_SIZE, CELL_SIZE, CELL_SIZE)
    surface.blit(cheese_img, cheese_rect)

def draw_q_values(surface, q_table, offset_x):
    max_q = np.max(q_table)
    min_q = np.min(q_table)
    q_range = max_q - min_q if max_q != min_q else 1

    for x in range(GRID_SIZE):
        for y in range(GRID_SIZE):
            rect = pygame.Rect(offset_x + y * CELL_SIZE, x * CELL_SIZE, CELL_SIZE, CELL_SIZE)
            max_q_value = np.max(q_table[x, y])
            
            if np.allclose(q_table[x, y], np.zeros_like(q_table[x, y])):
                color = BLACK
            else:
                normalized_q = (max_q_value - min_q) / q_range
                color = pygame.Color(int(255 * (1 - normalized_q)), 0, int(255 * normalized_q))
            
            pygame.draw.rect(surface, color, rect)

            font = pygame.font.Font(None, 20)
            text = font.render(f"{max_q_value:.2f}", True, WHITE)
            text_rect = text.get_rect(center=rect.center)
            surface.blit(text, text_rect)


def main():
    grid_world = GridWorld()
    agent = QLearningAgent(GRID_SIZE, 4)

    time_penalty_slider = Slider(10, GRID_WIDTH, SCREEN_WIDTH - 20, SLIDER_HEIGHT, -1, 0, -0.1, "Time Penalty")
    distance_penalty_slider = Slider(10, GRID_WIDTH + SLIDER_HEIGHT, SCREEN_WIDTH - 20, SLIDER_HEIGHT, -1, 0, -0.1, "Distance Penalty")
    cheese_reward_slider = Slider(10, GRID_WIDTH + 2 * SLIDER_HEIGHT, SCREEN_WIDTH - 20, SLIDER_HEIGHT, 80, 100, 10, "Cheese Reward")
    speed_slider = Slider(10, GRID_WIDTH + 3 * SLIDER_HEIGHT, SCREEN_WIDTH - 20, SLIDER_HEIGHT, 1, 1000, 10, "Training Speed")
    discount_rate_slider = Slider(10, GRID_WIDTH + 4 * SLIDER_HEIGHT, SCREEN_WIDTH - 20, SLIDER_HEIGHT, 0, 1, 0.99, "Discount Rate")
    timeout_penalty_slider = Slider(10, GRID_WIDTH + 5 * SLIDER_HEIGHT, SCREEN_WIDTH - 20, SLIDER_HEIGHT, -10, 0, -1, "Timeout Penalty")

    epsilon = 0.05
    alpha = 0.5

    clock = pygame.time.Clock()
    running = True
    training = False
    last_move_time = time.time()
    move_delay = 0.01  # Increased speed for smoother animation

    grid_surface = pygame.Surface((GRID_WIDTH, GRID_WIDTH))
    q_table_surface = pygame.Surface((GRID_WIDTH, GRID_WIDTH))

    generation_count = 0

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1 and event.pos[1] < GRID_WIDTH:  # Left click
                    grid_x = event.pos[1] // CELL_SIZE
                    grid_y = event.pos[0] // CELL_SIZE if event.pos[0] < GRID_WIDTH else (event.pos[0] - GRID_WIDTH - CELL_SIZE) // CELL_SIZE
                    grid_world.add_wall((grid_x, grid_y))
                elif event.button == 3 and event.pos[1] < GRID_WIDTH:  # Right click
                    grid_x = event.pos[1] // CELL_SIZE
                    grid_y = event.pos[0] // CELL_SIZE if event.pos[0] < GRID_WIDTH else (event.pos[0] - GRID_WIDTH - CELL_SIZE) // CELL_SIZE
                    grid_world.remove_wall((grid_x, grid_y))
                elif event.button == 1 and GRID_WIDTH <= event.pos[1] <= TOTAL_HEIGHT:
                    training = not training
            
            time_penalty_slider.handle_event(event)
            distance_penalty_slider.handle_event(event)
            cheese_reward_slider.handle_event(event)
            speed_slider.handle_event(event)
            discount_rate_slider.handle_event(event)
            timeout_penalty_slider.handle_event(event)

        screen.fill(GRAY)

        current_time = time.time()
        if training and current_time - last_move_time >= move_delay / speed_slider.val:
            episode_finished = False
            for _ in range(int(speed_slider.val)):
                state = grid_world.mouse_pos
                action = agent.get_action(state, epsilon)
                next_state, done = grid_world.step(action)
                
                time_penalty = time_penalty_slider.val
                distance_penalty = distance_penalty_slider.val * np.sqrt((next_state[0] - grid_world.cheese_pos[0])**2 + (next_state[1] - grid_world.cheese_pos[1])**2)
                cheese_reward = cheese_reward_slider.val if done and next_state == grid_world.cheese_pos else 0
                timeout_penalty = timeout_penalty_slider.val if done and next_state != grid_world.cheese_pos else 0
                
                reward = time_penalty + distance_penalty + cheese_reward + timeout_penalty
                
                agent.update_q_table(state, action, reward, next_state, done, alpha, discount_rate_slider.val)
                
                if done:
                    grid_world.reset()
                    generation_count += 1
                    episode_finished = True
                    break
            
            last_move_time = current_time
            if episode_finished:
                draw_q_values(q_table_surface, agent.q_table, 0)

        grid_surface.fill(WHITE)
        draw_grid(grid_surface, grid_world)
        screen.blit(grid_surface, (0, 0))
        screen.blit(q_table_surface, (GRID_WIDTH + CELL_SIZE, 0))
        
        time_penalty_slider.draw(screen)
        distance_penalty_slider.draw(screen)
        cheese_reward_slider.draw(screen)
        speed_slider.draw(screen)
        discount_rate_slider.draw(screen)
        timeout_penalty_slider.draw(screen)

        font = pygame.font.Font(None, 24)
        text = font.render(f"{'Stop' if training else 'Start'} Training", True, BLACK)
        text_rect = text.get_rect(center=(SCREEN_WIDTH // 2, TOTAL_HEIGHT - SLIDER_HEIGHT // 2))
        screen.blit(text, text_rect)

        reward_text = font.render(f"Current Reward: {agent.current_reward:.2f}", True, BLACK)
        screen.blit(reward_text, (10, GRID_WIDTH + 6 * SLIDER_HEIGHT))

        generation_text = font.render(f"Generation: {generation_count}", True, BLACK)
        screen.blit(generation_text, (SCREEN_WIDTH - 150, GRID_WIDTH + 6 * SLIDER_HEIGHT))

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()

if __name__ == "__main__":
    main()
