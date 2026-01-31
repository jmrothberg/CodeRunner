import pygame
import sys
import random
import math
from enum import Enum

# Initialize Pygame
pygame.init()

# Constants
TILE_SIZE = 24
MAZE_HEIGHT = 31  # Fixed: actual maze has 31 rows
MAZE_WIDTH = 28
SCREEN_WIDTH = MAZE_WIDTH * TILE_SIZE
SCREEN_HEIGHT = MAZE_HEIGHT * TILE_SIZE

# Colors
BLACK = (0, 0, 0)
BLUE = (33, 33, 255)
WHITE = (255, 255, 255)
YELLOW = (255, 255, 0)
RED = (255, 0, 0)
PINK = (255, 184, 255)
CYAN = (0, 255, 255)
ORANGE = (255, 184, 82)
DARK_BLUE = (0, 0, 139)
PELLET_BLUE = (51, 51, 255)

# Game States
class GameState(Enum):
    READY = 1
    PLAYING = 2
    DYING = 3
    GAME_OVER = 4
    LEVEL_COMPLETE = 5

# Direction
class Direction(Enum):
    UP = (0, -1)
    DOWN = (0, 1)
    LEFT = (-1, 0)
    RIGHT = (1, 0)
    NONE = (0, 0)

# Maze layout (0=empty, 1=wall, 2=dot, 3=power pellet, 4=ghost house, 5=tunnel)
MAZE = [
    [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
    [1,2,2,2,2,2,2,2,2,2,2,2,2,1,1,2,2,2,2,2,2,2,2,2,2,2,2,1],
    [1,2,1,1,1,1,2,1,1,1,1,1,2,1,1,2,1,1,1,1,1,2,1,1,1,1,2,1],
    [1,3,1,1,1,1,2,1,1,1,1,1,2,1,1,2,1,1,1,1,1,2,1,1,1,1,3,1],
    [1,2,1,1,1,1,2,1,1,1,1,1,2,1,1,2,1,1,1,1,1,2,1,1,1,1,2,1],
    [1,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,1],
    [1,2,1,1,1,1,2,1,1,2,1,1,1,1,1,1,1,1,2,1,1,2,1,1,1,1,2,1],
    [1,2,1,1,1,1,2,1,1,2,1,1,1,1,1,1,1,1,2,1,1,2,1,1,1,1,2,1],
    [1,2,2,2,2,2,2,1,1,2,2,2,2,1,1,2,2,2,2,1,1,2,2,2,2,2,2,1],
    [1,1,1,1,1,1,2,1,1,1,1,1,0,1,1,0,1,1,1,1,1,2,1,1,1,1,1,1],
    [1,1,1,1,1,1,2,1,1,1,1,1,0,1,1,0,1,1,1,1,1,2,1,1,1,1,1,1],
    [1,1,1,1,1,1,2,1,1,0,0,0,0,0,0,0,0,0,0,1,1,2,1,1,1,1,1,1],
    [1,1,1,1,1,1,2,1,1,0,1,1,1,4,4,1,1,1,0,1,1,2,1,1,1,1,1,1],
    [1,1,1,1,1,1,2,1,1,0,1,4,4,4,4,4,4,1,0,1,1,2,1,1,1,1,1,1],
    [5,0,0,0,0,0,2,0,0,0,1,4,4,4,4,4,4,1,0,0,0,2,0,0,0,0,0,5],
    [1,1,1,1,1,1,2,1,1,0,1,4,4,4,4,4,4,1,0,1,1,2,1,1,1,1,1,1],
    [1,1,1,1,1,1,2,1,1,0,1,1,1,1,1,1,1,1,0,1,1,2,1,1,1,1,1,1],
    [1,1,1,1,1,1,2,1,1,0,0,0,0,0,0,0,0,0,0,1,1,2,1,1,1,1,1,1],
    [1,1,1,1,1,1,2,1,1,0,1,1,1,1,1,1,1,1,0,1,1,2,1,1,1,1,1,1],
    [1,1,1,1,1,1,2,1,1,0,1,1,1,1,1,1,1,1,0,1,1,2,1,1,1,1,1,1],
    [1,2,2,2,2,2,2,2,2,2,2,2,2,1,1,2,2,2,2,2,2,2,2,2,2,2,2,1],
    [1,2,1,1,1,1,2,1,1,1,1,1,2,1,1,2,1,1,1,1,1,2,1,1,1,1,2,1],
    [1,2,1,1,1,1,2,1,1,1,1,1,2,1,1,2,1,1,1,1,1,2,1,1,1,1,2,1],
    [1,3,2,2,1,1,2,2,2,2,2,2,2,0,0,2,2,2,2,2,2,2,1,1,2,2,3,1],
    [1,1,1,2,1,1,2,1,1,2,1,1,1,1,1,1,1,1,2,1,1,2,1,1,2,1,1,1],
    [1,1,1,2,1,1,2,1,1,2,1,1,1,1,1,1,1,1,2,1,1,2,1,1,2,1,1,1],
    [1,2,2,2,2,2,2,1,1,2,2,2,2,1,1,2,2,2,2,1,1,2,2,2,2,2,2,1],
    [1,2,1,1,1,1,1,1,1,1,1,1,2,1,1,2,1,1,1,1,1,1,1,1,1,1,2,1],
    [1,2,1,1,1,1,1,1,1,1,1,1,2,1,1,2,1,1,1,1,1,1,1,1,1,1,2,1],
    [1,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,1],
    [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
]

class Pacman:
    def __init__(self):
        self.reset()
        self.mouth_angle = 0
        self.mouth_opening = True
        self.animation_speed = 15
        
    def reset(self):
        self.grid_x = 14
        self.grid_y = 23
        self.x = self.grid_x * TILE_SIZE
        self.y = self.grid_y * TILE_SIZE
        self.direction = Direction.NONE
        self.next_direction = Direction.NONE
        self.speed = 2
        self.alive = True
        self.death_animation_frame = 0
        
    def update(self, maze):
        if not self.alive:
            return
            
        # Update mouth animation
        if self.mouth_opening:
            self.mouth_angle += self.animation_speed
            if self.mouth_angle >= 45:
                self.mouth_opening = False
        else:
            self.mouth_angle -= self.animation_speed
            if self.mouth_angle <= 0:
                self.mouth_opening = True
        
        # Check if aligned with grid
        aligned_x = self.x % TILE_SIZE == 0
        aligned_y = self.y % TILE_SIZE == 0
        
        if aligned_x and aligned_y:
            self.grid_x = self.x // TILE_SIZE
            self.grid_y = self.y // TILE_SIZE
            
            # Try to change direction
            if self.next_direction != Direction.NONE:
                if self.can_move(self.next_direction, maze):
                    self.direction = self.next_direction
                    self.next_direction = Direction.NONE
            
            # Check if current direction is blocked
            if not self.can_move(self.direction, maze):
                self.direction = Direction.NONE
        
        # Move
        if self.direction != Direction.NONE:
            dx, dy = self.direction.value
            self.x += dx * self.speed
            self.y += dy * self.speed
            
            # Handle tunnel wrapping
            if self.x < -TILE_SIZE:
                self.x = SCREEN_WIDTH
            elif self.x > SCREEN_WIDTH:
                self.x = -TILE_SIZE
    
    def can_move(self, direction, maze):
        if direction == Direction.NONE:
            return False
            
        dx, dy = direction.value
        next_grid_x = self.grid_x + dx
        next_grid_y = self.grid_y + dy
        
        # Check bounds
        if next_grid_y < 0 or next_grid_y >= MAZE_HEIGHT:
            return False
        if next_grid_x < 0 or next_grid_x >= MAZE_WIDTH:
            return True  # Allow tunnel movement
            
        # Check if wall
        tile = maze[next_grid_y][next_grid_x]
        return tile != 1
    
    def set_next_direction(self, direction):
        self.next_direction = direction
    
    def draw(self, screen):
        if not self.alive:
            # Death animation
            if self.death_animation_frame < 12:
                angles = [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330]
                angle = angles[self.death_animation_frame]
                pygame.draw.arc(screen, YELLOW, 
                              (self.x, self.y, TILE_SIZE, TILE_SIZE),
                              0, 6.28, 3)
            return
        
        center_x = self.x + TILE_SIZE // 2
        center_y = self.y + TILE_SIZE // 2
        radius = TILE_SIZE // 2 - 2
        
        # Determine rotation based on direction
        rotation = 0
        if self.direction == Direction.RIGHT:
            rotation = 0
        elif self.direction == Direction.LEFT:
            rotation = 180
        elif self.direction == Direction.UP:
            rotation = 90
        elif self.direction == Direction.DOWN:
            rotation = 270
        
        # Draw Pac-Man with animated mouth
        mouth_rad = math.radians(self.mouth_angle)
        start_angle = math.radians(rotation) + mouth_rad
        end_angle = math.radians(rotation) - mouth_rad + math.pi * 2
        
        # Draw filled circle
        pygame.draw.circle(screen, YELLOW, (center_x, center_y), radius)
        
        # Draw mouth (black triangle)
        if self.mouth_angle > 0:
            points = [(center_x, center_y)]
            for angle in [start_angle, end_angle]:
                px = center_x + radius * math.cos(angle)
                py = center_y - radius * math.sin(angle)
                points.append((px, py))
            pygame.draw.polygon(screen, BLACK, points)

class Ghost:
    def __init__(self, color, start_x, start_y, personality):
        self.color = color
        self.start_x = start_x
        self.start_y = start_y
        self.personality = personality  # 'blinky', 'pinky', 'inky', 'clyde'
        self.reset()
        self.animation_frame = 0
        
    def reset(self):
        self.grid_x = self.start_x
        self.grid_y = self.start_y
        self.x = self.grid_x * TILE_SIZE
        self.y = self.grid_y * TILE_SIZE
        self.direction = Direction.LEFT
        self.speed = 2
        self.mode = 'chase'  # 'chase', 'scatter', 'frightened', 'eaten'
        self.frightened_timer = 0
        self.in_house = True
        self.exit_timer = 0
        
    def update(self, pacman, ghosts, maze):
        self.animation_frame = (self.animation_frame + 1) % 20
        
        # Handle frightened mode timer
        if self.mode == 'frightened':
            self.frightened_timer -= 1
            if self.frightened_timer <= 0:
                self.mode = 'chase'
        
        # Handle ghost house exit
        if self.in_house:
            self.exit_timer += 1
            if self.exit_timer > 60:
                self.in_house = False
                self.grid_x = 14
                self.grid_y = 11
                self.x = self.grid_x * TILE_SIZE
                self.y = self.grid_y * TILE_SIZE
            return
        
        # Handle eaten mode - return to house
        if self.mode == 'eaten':
            if self.grid_x == 14 and self.grid_y == 14:
                self.mode = 'chase'
                self.in_house = True
                self.exit_timer = 0
        
        # Check if aligned with grid
        aligned_x = self.x % TILE_SIZE == 0
        aligned_y = self.y % TILE_SIZE == 0
        
        if aligned_x and aligned_y:
            self.grid_x = self.x // TILE_SIZE
            self.grid_y = self.y // TILE_SIZE
            
            # Choose next direction
            self.choose_direction(pacman, ghosts, maze)
        
        # Move
        if self.direction != Direction.NONE:
            dx, dy = self.direction.value
            speed = self.speed if self.mode != 'frightened' else 1
            if self.mode == 'eaten':
                speed = 4
            self.x += dx * speed
            self.y += dy * speed
            
            # Handle tunnel wrapping
            if self.x < -TILE_SIZE:
                self.x = SCREEN_WIDTH
            elif self.x > SCREEN_WIDTH:
                self.x = -TILE_SIZE
    
    def choose_direction(self, pacman, ghosts, maze):
        if self.mode == 'frightened':
            # Random movement when frightened
            possible_dirs = self.get_possible_directions(maze)
            if possible_dirs:
                self.direction = random.choice(possible_dirs)
            return
        
        # Get target tile based on personality
        target_x, target_y = self.get_target_tile(pacman, ghosts)
        
        # Find best direction using distance calculation
        possible_dirs = self.get_possible_directions(maze)
        if not possible_dirs:
            return
        
        best_dir = possible_dirs[0]
        best_dist = float('inf')
        
        for direction in possible_dirs:
            dx, dy = direction.value
            next_x = self.grid_x + dx
            next_y = self.grid_y + dy
            
            # Calculate distance to target
            dist = ((next_x - target_x) ** 2 + (next_y - target_y) ** 2) ** 0.5
            
            if dist < best_dist:
                best_dist = dist
                best_dir = direction
        
        self.direction = best_dir
    
    def get_target_tile(self, pacman, ghosts):
        pac_grid_x = pacman.x // TILE_SIZE
        pac_grid_y = pacman.y // TILE_SIZE
        
        if self.mode == 'eaten':
            return (14, 14)  # Return to ghost house
        
        if self.mode == 'scatter':
            # Each ghost has a corner to scatter to
            corners = {
                'blinky': (25, 0),
                'pinky': (2, 0),
                'inky': (27, 30),
                'clyde': (0, 30)
            }
            return corners.get(self.personality, (0, 0))
        
        # Chase mode - each ghost has unique behavior
        if self.personality == 'blinky':
            # Red ghost - directly chases Pac-Man
            return (pac_grid_x, pac_grid_y)
        
        elif self.personality == 'pinky':
            # Pink ghost - targets 4 tiles ahead of Pac-Man
            dx, dy = pacman.direction.value
            return (pac_grid_x + dx * 4, pac_grid_y + dy * 4)
        
        elif self.personality == 'inky':
            # Cyan ghost - uses Blinky's position for flanking
            blinky = next((g for g in ghosts if g.personality == 'blinky'), None)
            if blinky:
                dx, dy = pacman.direction.value
                pivot_x = pac_grid_x + dx * 2
                pivot_y = pac_grid_y + dy * 2
                target_x = pivot_x + (pivot_x - blinky.grid_x)
                target_y = pivot_y + (pivot_y - blinky.grid_y)
                return (target_x, target_y)
            return (pac_grid_x, pac_grid_y)
        
        elif self.personality == 'clyde':
            # Orange ghost - chases if far, scatters if close
            dist = ((self.grid_x - pac_grid_x) ** 2 + (self.grid_y - pac_grid_y) ** 2) ** 0.5
            if dist > 8:
                return (pac_grid_x, pac_grid_y)
            else:
                return (0, 30)  # Scatter corner
        
        return (pac_grid_x, pac_grid_y)
    
    def get_possible_directions(self, maze):
        possible = []
        opposite = {
            Direction.UP: Direction.DOWN,
            Direction.DOWN: Direction.UP,
            Direction.LEFT: Direction.RIGHT,
            Direction.RIGHT: Direction.LEFT,
            Direction.NONE: Direction.NONE
        }
        
        for direction in [Direction.UP, Direction.DOWN, Direction.LEFT, Direction.RIGHT]:
            if direction == opposite[self.direction]:
                continue  # Ghosts can't reverse
            
            dx, dy = direction.value
            next_x = self.grid_x + dx
            next_y = self.grid_y + dy
            
            # Check bounds
            if next_y < 0 or next_y >= MAZE_HEIGHT:
                continue
            if next_x < 0 or next_x >= MAZE_WIDTH:
                possible.append(direction)
                continue
            
            # Check if wall
            tile = maze[next_y][next_x]
            if tile != 1:
                possible.append(direction)
        
        return possible if possible else [self.direction]
    
    def make_frightened(self, duration):
        if self.mode != 'eaten':
            self.mode = 'frightened'
            self.frightened_timer = duration
            # Reverse direction
            reverse = {
                Direction.UP: Direction.DOWN,
                Direction.DOWN: Direction.UP,
                Direction.LEFT: Direction.RIGHT,
                Direction.RIGHT: Direction.LEFT,
                Direction.NONE: Direction.NONE
            }
            self.direction = reverse[self.direction]
    
    def get_eaten(self):
        self.mode = 'eaten'
    
    def draw(self, screen):
        center_x = self.x + TILE_SIZE // 2
        center_y = self.y + TILE_SIZE // 2
        radius = TILE_SIZE // 2 - 2
        
        # Choose color based on mode
        if self.mode == 'frightened':
            if self.frightened_timer > 120 or (self.frightened_timer // 10) % 2 == 0:
                color = DARK_BLUE
            else:
                color = WHITE
        elif self.mode == 'eaten':
            # Draw eyes only
            self.draw_eyes(screen, center_x, center_y)
            return
        else:
            color = self.color
        
        # Draw ghost body (rounded top, wavy bottom)
        # Top half circle
        pygame.draw.circle(screen, color, (center_x, center_y - 2), radius)
        
        # Bottom rectangle with waves
        rect_height = radius + 2
        pygame.draw.rect(screen, color, 
                        (center_x - radius, center_y - 2, radius * 2, rect_height))
        
        # Draw wavy bottom
        wave_width = radius * 2 // 3
        for i in range(3):
            wave_x = center_x - radius + i * wave_width
            if self.animation_frame < 10:
                pygame.draw.circle(screen, color, (wave_x + wave_width // 2, center_y + rect_height - 2), wave_width // 2)
            else:
                pygame.draw.circle(screen, color, (wave_x + wave_width // 2, center_y + rect_height - 4), wave_width // 2)
        
        # Cover bottom with black to create wave effect
        points = []
        for i in range(4):
            wave_x = center_x - radius + i * wave_width
            if self.animation_frame < 10:
                points.append((wave_x, center_y + rect_height))
            else:
                points.append((wave_x, center_y + rect_height - 2))
        points.append((center_x + radius, center_y + rect_height + 5))
        points.append((center_x - radius, center_y + rect_height + 5))
        pygame.draw.polygon(screen, BLACK, points)
        
        # Draw eyes
        self.draw_eyes(screen, center_x, center_y)
    
    def draw_eyes(self, screen, center_x, center_y):
        eye_radius = 4
        pupil_radius = 2
        
        # Determine pupil direction based on movement
        pupil_offset_x = 0
        pupil_offset_y = 0
        if self.direction == Direction.LEFT:
            pupil_offset_x = -2
        elif self.direction == Direction.RIGHT:
            pupil_offset_x = 2
        elif self.direction == Direction.UP:
            pupil_offset_y = -2
        elif self.direction == Direction.DOWN:
            pupil_offset_y = 2
        
        # Left eye
        pygame.draw.circle(screen, WHITE, (center_x - 6, center_y - 4), eye_radius)
        pygame.draw.circle(screen, BLUE, 
                         (center_x - 6 + pupil_offset_x, center_y - 4 + pupil_offset_y), 
                         pupil_radius)
        
        # Right eye
        pygame.draw.circle(screen, WHITE, (center_x + 6, center_y - 4), eye_radius)
        pygame.draw.circle(screen, BLUE, 
                         (center_x + 6 + pupil_offset_x, center_y - 4 + pupil_offset_y), 
                         pupil_radius)

class Fruit:
    def __init__(self):
        self.active = False
        self.x = 14 * TILE_SIZE
        self.y = 17 * TILE_SIZE
        self.timer = 0
        self.type = 'cherry'
        self.points = 100
        
    def spawn(self, level):
        self.active = True
        self.timer = 600  # 10 seconds
        fruits = ['cherry', 'strawberry', 'orange', 'apple', 'melon']
        self.type = fruits[min(level - 1, 4)]
        self.points = 100 * min(level, 5)
        print(f"Fruit spawned: {self.type} worth {self.points} points")
    
    def update(self):
        if self.active:
            self.timer -= 1
            if self.timer <= 0:
                self.active = False
                print("Fruit despawned")
    
    def collect(self):
        if self.active:
            self.active = False
            print(f"Fruit collected! +{self.points} points")
            return self.points
        return 0
    
    def draw(self, screen):
        if not self.active:
            return
        
        center_x = self.x + TILE_SIZE // 2
        center_y = self.y + TILE_SIZE // 2
        
        # Draw simple fruit representation
        if self.type == 'cherry':
            pygame.draw.circle(screen, RED, (center_x - 4, center_y + 2), 6)
            pygame.draw.circle(screen, RED, (center_x + 4, center_y + 2), 6)
        elif self.type == 'strawberry':
            points = [(center_x, center_y - 6), (center_x - 6, center_y + 6), 
                     (center_x + 6, center_y + 6)]
            pygame.draw.polygon(screen, RED, points)
        elif self.type == 'orange':
            pygame.draw.circle(screen, ORANGE, (center_x, center_y), 8)
        elif self.type == 'apple':
            pygame.draw.circle(screen, RED, (center_x, center_y), 8)
        elif self.type == 'melon':
            pygame.draw.ellipse(screen, (0, 255, 0), 
                              (center_x - 10, center_y - 6, 20, 12))

class Game:
    def __init__(self):
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("PAC-MAN")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 36)
        self.small_font = pygame.font.Font(None, 24)
        
        self.reset_game()
        
    def reset_game(self):
        self.state = GameState.READY
        self.pacman = Pacman()
        self.ghosts = [
            Ghost(RED, 14, 14, 'blinky'),
            Ghost(PINK, 14, 14, 'pinky'),
            Ghost(CYAN, 13, 14, 'inky'),
            Ghost(ORANGE, 15, 14, 'clyde')
        ]
        self.fruit = Fruit()
        
        # Initialize maze with dots
        self.maze = [row[:] for row in MAZE]
        self.dots_remaining = sum(row.count(2) for row in self.maze)
        self.total_dots = self.dots_remaining
        
        self.score = 0
        self.lives = 3
        self.level = 1
        self.ready_timer = 120
        self.power_mode = False
        self.power_timer = 0
        self.ghost_combo = 0
        self.dots_eaten_this_level = 0
        
        print(f"Game initialized - Level {self.level}, Lives: {self.lives}, Dots: {self.dots_remaining}")
    
    def reset_level(self):
        self.state = GameState.READY
        self.pacman.reset()
        for ghost in self.ghosts:
            ghost.reset()
        self.ready_timer = 120
        self.power_mode = False
        self.power_timer = 0
        self.dots_eaten_this_level = 0
        print(f"Level {self.level} started")
    
    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            
            if event.type == pygame.KEYDOWN:
                if self.state == GameState.GAME_OVER:
                    if event.key == pygame.K_SPACE:
                        self.reset_game()
                
                if event.key == pygame.K_UP:
                    self.pacman.set_next_direction(Direction.UP)
                elif event.key == pygame.K_DOWN:
                    self.pacman.set_next_direction(Direction.DOWN)
                elif event.key == pygame.K_LEFT:
                    self.pacman.set_next_direction(Direction.LEFT)
                elif event.key == pygame.K_RIGHT:
                    self.pacman.set_next_direction(Direction.RIGHT)
        
        return True
    
    def update(self):
        if self.state == GameState.READY:
            self.ready_timer -= 1
            if self.ready_timer <= 0:
                self.state = GameState.PLAYING
                print("Game started!")
        
        elif self.state == GameState.PLAYING:
            # Update Pac-Man
            self.pacman.update(self.maze)
            
            # Check dot collection
            grid_x = self.pacman.x // TILE_SIZE
            grid_y = self.pacman.y // TILE_SIZE
            
            if 0 <= grid_x < MAZE_WIDTH and 0 <= grid_y < MAZE_HEIGHT:
                tile = self.maze[grid_y][grid_x]
                
                if tile == 2:  # Regular dot
                    self.maze[grid_y][grid_x] = 0
                    self.score += 10
                    self.dots_remaining -= 1
                    self.dots_eaten_this_level += 1
                    
                    # Spawn fruit at certain dot counts
                    if self.dots_eaten_this_level == 70 or self.dots_eaten_this_level == 170:
                        self.fruit.spawn(self.level)
                    
                elif tile == 3:  # Power pellet
                    self.maze[grid_y][grid_x] = 0
                    self.score += 50
                    self.dots_remaining -= 1
                    self.power_mode = True
                    self.power_timer = 360  # 6 seconds
                    self.ghost_combo = 0
                    print("POWER MODE ACTIVATED!")
                    
                    for ghost in self.ghosts:
                        ghost.make_frightened(360)
            
            # Update power mode
            if self.power_mode:
                self.power_timer -= 1
                if self.power_timer <= 0:
                    self.power_mode = False
                    print("Power mode ended")
            
            # Update ghosts
            for ghost in self.ghosts:
                ghost.update(self.pacman, self.ghosts, self.maze)
            
            # Update fruit
            self.fruit.update()
            
            # Check collisions with ghosts
            for ghost in self.ghosts:
                if ghost.in_house:
                    continue
                    
                pac_rect = pygame.Rect(self.pacman.x + 4, self.pacman.y + 4, 
                                      TILE_SIZE - 8, TILE_SIZE - 8)
                ghost_rect = pygame.Rect(ghost.x + 4, ghost.y + 4, 
                                        TILE_SIZE - 8, TILE_SIZE - 8)
                
                if pac_rect.colliderect(ghost_rect):
                    if ghost.mode == 'frightened':
                        # Eat ghost
                        self.ghost_combo += 1
                        points = 200 * (2 ** (self.ghost_combo - 1))
                        self.score += points
                        ghost.get_eaten()
                        print(f"Ghost eaten! +{points} points (combo: {self.ghost_combo})")
                    elif ghost.mode != 'eaten':
                        # Pac-Man dies
                        self.pacman.alive = False
                        self.state = GameState.DYING
                        print("Pac-Man died!")
                        break
            
            # Check fruit collection
            if self.fruit.active:
                pac_rect = pygame.Rect(self.pacman.x + 4, self.pacman.y + 4, 
                                      TILE_SIZE - 8, TILE_SIZE - 8)
                fruit_rect = pygame.Rect(self.fruit.x + 4, self.fruit.y + 4, 
                                        TILE_SIZE - 8, TILE_SIZE - 8)
                if pac_rect.colliderect(fruit_rect):
                    self.score += self.fruit.collect()
            
            # Check level complete
            if self.dots_remaining == 0:
                self.state = GameState.LEVEL_COMPLETE
                print(f"Level {self.level} complete!")
        
        elif self.state == GameState.DYING:
            self.pacman.death_animation_frame += 1
            if self.pacman.death_animation_frame > 60:
                self.lives -= 1
                print(f"Lives remaining: {self.lives}")
                if self.lives <= 0:
                    self.state = GameState.GAME_OVER
                    print("GAME OVER!")
                else:
                    self.reset_level()
        
        elif self.state == GameState.LEVEL_COMPLETE:
            self.ready_timer -= 1
            if self.ready_timer <= 0:
                self.level += 1
                # Reset maze
                self.maze = [row[:] for row in MAZE]
                self.dots_remaining = sum(row.count(2) for row in self.maze)
                self.reset_level()
    
    def draw(self):
        self.screen.fill(BLACK)
        
        # Draw maze
        for y in range(MAZE_HEIGHT):
            for x in range(MAZE_WIDTH):
                tile = self.maze[y][x]
                px = x * TILE_SIZE
                py = y * TILE_SIZE
                
                if tile == 1:  # Wall
                    # Draw blue wall with border
                    pygame.draw.rect(self.screen, BLUE, 
                                   (px + 1, py + 1, TILE_SIZE - 2, TILE_SIZE - 2))
                    pygame.draw.rect(self.screen, PELLET_BLUE, 
                                   (px + 1, py + 1, TILE_SIZE - 2, TILE_SIZE - 2), 1)
                
                elif tile == 2:  # Dot
                    pygame.draw.circle(self.screen, WHITE, 
                                     (px + TILE_SIZE // 2, py + TILE_SIZE // 2), 2)
                
                elif tile == 3:  # Power pellet
                    # Animated power pellet
                    radius = 6 if (pygame.time.get_ticks() // 200) % 2 == 0 else 5
                    pygame.draw.circle(self.screen, WHITE, 
                                     (px + TILE_SIZE // 2, py + TILE_SIZE // 2), radius)
                
                elif tile == 4:  # Ghost house
                    pygame.draw.rect(self.screen, PINK, 
                                   (px + 2, py + 2, TILE_SIZE - 4, TILE_SIZE - 4), 1)
        
        # Draw fruit
        self.fruit.draw(self.screen)
        
        # Draw ghosts
        for ghost in self.ghosts:
            ghost.draw(self.screen)
        
        # Draw Pac-Man
        self.pacman.draw(self.screen)
        
        # Draw UI
        score_text = self.font.render(f"SCORE: {self.score}", True, WHITE)
        self.screen.blit(score_text, (10, 10))
        
        level_text = self.font.render(f"LEVEL: {self.level}", True, WHITE)
        self.screen.blit(level_text, (SCREEN_WIDTH - 200, 10))
        
        # Draw lives
        for i in range(self.lives):
            pygame.draw.circle(self.screen, YELLOW, 
                             (30 + i * 30, SCREEN_HEIGHT - 30), 10)
        
        # Draw game state messages
        if self.state == GameState.READY:
            ready_text = self.font.render("READY!", True, YELLOW)
            text_rect = ready_text.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2))
            self.screen.blit(ready_text, text_rect)
        
        elif self.state == GameState.GAME_OVER:
            game_over_text = self.font.render("GAME OVER", True, RED)
            text_rect = game_over_text.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2))
            self.screen.blit(game_over_text, text_rect)
            
            restart_text = self.small_font.render("Press SPACE to restart", True, WHITE)
            text_rect = restart_text.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 + 40))
            self.screen.blit(restart_text, text_rect)
        
        elif self.state == GameState.LEVEL_COMPLETE:
            complete_text = self.font.render("LEVEL COMPLETE!", True, CYAN)
            text_rect = complete_text.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2))
            self.screen.blit(complete_text, text_rect)
        
        pygame.display.flip()
    
    def run(self):
        print("=== PAC-MAN GAME STARTED ===")
        print("Controls: Arrow keys to move")
        print("Eat all dots to complete the level!")
        print("Eat power pellets to turn ghosts blue and eat them!")
        
        running = True
        while running:
            running = self.handle_events()
            self.update()
            self.draw()
            self.clock.tick(60)
        
        pygame.quit()
        sys.exit()

if __name__ == "__main__":
    game = Game()
    game.run()