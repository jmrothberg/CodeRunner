
import pygame
import random
import math

# Initialize pygame
pygame.init()

# Constants
SCREEN_WIDTH = 672
SCREEN_HEIGHT = 744
CELL_SIZE = 24
MAZE_WIDTH = SCREEN_WIDTH // CELL_SIZE
MAZE_HEIGHT = (SCREEN_HEIGHT - 96) // CELL_SIZE  # Extra space for UI

# Colors
BLACK = (0, 0, 0)
BLUE = (33, 33, 255)
YELLOW = (255, 255, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
PINK = (255, 184, 255)
CYAN = (0, 255, 255)
ORANGE = (255, 184, 82)

# Game states
READY = 0
PLAYING = 1
GAME_OVER = 2

class PacMan:
    def __init__(self):
        self.reset()
        self.mouth_open = True
        self.animation_counter = 0
        
    def reset(self):
        self.x = 14 * CELL_SIZE
        self.y = 22 * CELL_SIZE
        self.direction = "STOP"
        self.next_direction = "STOP"
        self.speed = 1
        self.alive = True
        
    def update(self, maze):
        # Update animation
        self.animation_counter += 1
        if self.animation_counter >= 5:
            self.mouth_open = not self.mouth_open
            self.animation_counter = 0
            
        # Try to change direction if requested
        if self.next_direction != self.direction:
            new_x, new_y = self.x, self.y
            
            if self.next_direction == "LEFT":
                new_x -= self.speed
            elif self.next_direction == "RIGHT":
                new_x += self.speed
            elif self.next_direction == "UP":
                new_y -= self.speed
            elif self.next_direction == "DOWN":
                new_y += self.speed
                
            # Check if the new position is valid
            grid_x = round(new_x / CELL_SIZE)
            grid_y = round(new_y / CELL_SIZE)
            
            if 0 <= grid_y < MAZE_HEIGHT and 0 <= grid_x < MAZE_WIDTH:
                if maze[grid_y][grid_x] != '#':
                    self.direction = self.next_direction
        
        # Move in current direction
        old_x, old_y = self.x, self.y
        
        if self.direction == "LEFT":
            self.x -= self.speed
        elif self.direction == "RIGHT":
            self.x += self.speed
        elif self.direction == "UP":
            self.y -= self.speed
        elif self.direction == "DOWN":
            self.y += self.speed
            
        # Handle tunnel wrapping
        if self.x < -CELL_SIZE:
            self.x = SCREEN_WIDTH
        elif self.x > SCREEN_WIDTH:
            self.x = -CELL_SIZE
            
        # Check for collisions with walls
        grid_x =round(self.x / CELL_SIZE)
        grid_y = round(self.y / CELL_SIZE)
       
        
        if 0 <= grid_y < MAZE_HEIGHT and 0 <= grid_x < MAZE_WIDTH:
            if maze[grid_y][grid_x] == '#':
                # Revert to previous position
                self.x, self.y = old_x, old_y
                
    def draw(self, screen):
        # Draw Pac-Man with animated mouth
        center_x = self.x + CELL_SIZE // 2
        center_y = self.y + CELL_SIZE // 2
        radius = CELL_SIZE // 2 - 1
        
        if self.mouth_open:
            # Draw open mouth based on direction
            start_angle = 0
            end_angle = 360
            
            if self.direction == "RIGHT":
                start_angle = 45
                end_angle = 315
            elif self.direction == "LEFT":
                start_angle = 225
                end_angle = 495  # Same as 135, but to make arc draw correctly
            elif self.direction == "DOWN":
                start_angle = 135
                end_angle = 405
            elif self.direction == "UP":
                start_angle = 315
                end_angle = 585
            
            pygame.draw.circle(screen, BLACK, (center_x, center_y), radius)
            
            # Draw the mouth
            points = [(center_x, center_y)]
            for angle in range(start_angle, end_angle + 1, 5):
                rad = math.radians(angle)
                x = center_x + radius * math.cos(rad)
                y = center_y + radius * math.sin(rad)
                points.append((x, y))
            
            pygame.draw.polygon(screen, YELLOW, points)
        else:
            # Draw closed mouth (full circle)
            pygame.draw.circle(screen, BLACK, (center_x, center_y), radius)

class Ghost:
    def __init__(self, color, x, y, name):
        self.color = color
        self.x = x
        self.y = y
        self.direction = "LEFT"
        self.speed = 1.5
        self.name = name
        self.mode = "SCATTER"  # SCATTER, CHASE, FRIGHTENED
        self.frightened_timer = 0
        self.returning_home = False
        self.animation_counter = 0
        self.wave = 1
        
    def update(self, maze, pacman, ghosts):
        # Update frightened timer
        if self.mode == "FRIGHTENED":
            self.frightened_timer -= 1
            if self.frightened_timer <= 0:
                self.mode = "CHASE"
                
        # Update animation counter
        self.animation_counter += 1
        if self.animation_counter >= 10:
            self.animation_counter = 0
            
        # Handle ghost returning home
        if self.returning_home:
            # Move toward center of ghost house (14, 14)
            target_x = 14 * CELL_SIZE
            target_y = 17 * CELL_SIZE  # Slightly below the actual center to enter from bottom
            
            dx = target_x - self.x
            dy = target_y - self.y
            
            if abs(dx) > abs(dy):
                if dx > 0:
                    new_direction = "RIGHT"
                else:
                    new_direction = "LEFT"
            else:
                if dy > 0:
                    new_direction = "DOWN"
                else:
                    new_direction = "UP"
                    
            # Check if reached home
            distance = math.sqrt(dx*dx + dy*dy)
            if distance < CELL_SIZE:
                self.returning_home = False
                self.mode = "SCATTER"
                
            self.direction = new_direction
        else:
            # Choose direction based on AI
            self.choose_direction(maze, pacman, ghosts)
            
        # Move in current direction
        old_x, old_y = self.x, self.y
        
        if self.direction == "LEFT":
            self.x -= self.speed
        elif self.direction == "RIGHT":
            self.x += self.speed
        elif self.direction == "UP":
            self.y -= self.speed
        elif self.direction == "DOWN":
            self.y += self.speed
            
        # Handle tunnel wrapping
        if self.x < -CELL_SIZE:
            self.x = SCREEN_WIDTH
        elif self.x > SCREEN_WIDTH:
            self.x = -CELL_SIZE
            
        # Check for collisions with walls and revert if necessary
        grid_x = round(self.x / CELL_SIZE)
        grid_y = round(self.y / CELL_SIZE)
        
        if 0 <= grid_y < MAZE_HEIGHT and 0 <= grid_x < MAZE_WIDTH:
            if maze[grid_y][grid_x] == '#':
                # Revert to previous position
                self.x, self.y = old_x, old_y
                
    def choose_direction(self, maze, pacman, ghosts):
        # Define target based on ghost personality
        if self.mode == "FRIGHTENED":
            # Random movement when frightened
            possible_directions = []
            for direction in ["UP", "DOWN", "LEFT", "RIGHT"]:
                if direction == "UP" and self.direction != "DOWN":
                    new_y = self.y - self.speed
                    grid_x = round(self.x / CELL_SIZE)
                    grid_y = round(new_y / CELL_SIZE)
                    if 0 <= grid_y < MAZE_HEIGHT and 0 <= grid_x < MAZE_WIDTH:
                        if maze[grid_y][grid_x] != '#':
                            possible_directions.append("UP")
                elif direction == "DOWN" and self.direction != "UP":
                    new_y = self.y + self.speed
                    grid_x = round(self.x / CELL_SIZE)
                    grid_y = round(new_y / CELL_SIZE)
                    if 0 <= grid_y < MAZE_HEIGHT and 0 <= grid_x < MAZE_WIDTH:
                        if maze[grid_y][grid_x] != '#':
                            possible_directions.append("DOWN")
                elif direction == "LEFT" and self.direction != "RIGHT":
                    new_x = self.x - self.speed
                    grid_x = round(new_x / CELL_SIZE)
                    grid_y = round(self.y / CELL_SIZE)
                    if 0 <= grid_y < MAZE_HEIGHT and 0 <= grid_x < MAZE_WIDTH:
                        if maze[grid_y][grid_x] != '#':
                            possible_directions.append("LEFT")
                elif direction == "RIGHT" and self.direction != "LEFT":
                    new_x = self.x + self.speed
                    grid_x = round(new_x / CELL_SIZE)
                    grid_y = round(self.y / CELL_SIZE)
                    if 0 <= grid_y < MAZE_HEIGHT and 0 <= grid_x < MAZE_WIDTH:
                        if maze[grid_y][grid_x] != '#':
                            possible_directions.append("RIGHT")
            
            if possible_directions:
                self.direction = random.choice(possible_directions)
            return
            
        # For non-frightened modes
        target_x, target_y = 0, 0
        
        if self.name == "BLINKY":  # Red ghost - chases directly
            target_x = pacman.x
            target_y = pacman.y
        elif self.name == "PINKY":  # Pink ghost - ambushes ahead of Pac-Man
            target_x = pacman.x
            target_y = pacman.y
            
            # Look 4 tiles ahead of Pac-Man's direction
            if pacman.direction == "LEFT":
                target_x -= 4 * CELL_SIZE
            elif pacman.direction == "RIGHT":
                target_x += 4 * CELL_SIZE
            elif pacman.direction == "UP":
                target_y -= 4 * CELL_SIZE
            elif pacman.direction == "DOWN":
                target_y += 4 * CELL_SIZE
        elif self.name == "INKY":  # Cyan ghost - flanks Pac-Man
            # Target is determined by vector from Blinky to a point ahead of Pac-Man
            ahead_x, ahead_y = pacman.x, pacman.y
            
            if pacman.direction == "LEFT":
                ahead_x -= 2 * CELL_SIZE
            elif pacman.direction == "RIGHT":
                ahead_x += 2 * CELL_SIZE
            elif pacman.direction == "UP":
                ahead_y -= 2 * CELL_SIZE
            elif pacman.direction == "DOWN":
                ahead_y += 2 * CELL_SIZE
                
            # Find Blinky (red ghost)
            blinky = None
            for ghost in ghosts:
                if ghost.name == "BLINKY":
                    blinky = ghost
                    break
                    
            if blinky:
                # Vector from Blinky to ahead point, doubled
                vec_x = ahead_x - blinky.x
                vec_y = ahead_y - blinky.y
                
                target_x = ahead_x + vec_x
                target_y = ahead_y + vec_y
        elif self.name == "CLYDE":  # Orange ghost - erratic behavior
            # Calculate distance to Pac-Man
            dx = pacman.x - self.x
            dy = pacman.y - self.y
            distance = math.sqrt(dx*dx + dy*dy)
            
            # If close to Pac-Man, target scatter corner (bottom left)
            if distance < 8 * CELL_SIZE:
                target_x = 0
                target_y = MAZE_HEIGHT * CELL_SIZE
            else:
                # Otherwise chase Pac-Man directly
                target_x = pacman.x
                target_y = pacman.y
                
        # Calculate the best direction to move toward target
        possible_directions = []
        
        # Check each possible direction
        for direction in ["UP", "DOWN", "LEFT", "RIGHT"]:
            if direction == "UP" and self.direction != "DOWN":
                new_y = self.y - self.speed
                grid_x = round(self.x / CELL_SIZE)
                grid_y = round(new_y / CELL_SIZE)
                if 0 <= grid_y < MAZE_HEIGHT and 0 <= grid_x < MAZE_WIDTH:
                    if maze[grid_y][grid_x] != '#':
                        possible_directions.append(("UP", new_y))
            elif direction == "DOWN" and self.direction != "UP":
                new_y = self.y + self.speed
                grid_x = round(self.x / CELL_SIZE)
                grid_y = round(new_y / CELL_SIZE)
                if 0 <= grid_y < MAZE_HEIGHT and 0 <= grid_x < MAZE_WIDTH:
                    if maze[grid_y][grid_x] != '#':
                        possible_directions.append(("DOWN", new_y))
            elif direction == "LEFT" and self.direction != "RIGHT":
                new_x = self.x - self.speed
                grid_x = round(new_x / CELL_SIZE)
                grid_y = round(self.y / CELL_SIZE)
                if 0 <= grid_y < MAZE_HEIGHT and 0 <= grid_x < MAZE_WIDTH:
                    if maze[grid_y][grid_x] != '#':
                        possible_directions.append(("LEFT", new_x))
            elif direction == "RIGHT" and self.direction != "LEFT":
                new_x = self.x + self.speed
                grid_x = round(new_x / CELL_SIZE)
                grid_y = round(self.y / CELL_SIZE)
                if 0 <= grid_y < MAZE_HEIGHT and 0 <= grid_x < MAZE_WIDTH:
                    if maze[grid_y][grid_x] != '#':
                        possible_directions.append(("RIGHT", new_x))
                        
        # Choose the direction that minimizes distance to target
        if possible_directions:
            best_direction = possible_directions[0][0]
            min_distance = float('inf')
            
            for direction, new_pos in possible_directions:
                # Calculate distance to target based on movement direction
                if direction in ["LEFT", "RIGHT"]:
                    distance = abs(target_x - new_pos)
                else:  # UP or DOWN
                    distance = abs(target_y - new_pos)
                    
                if distance < min_distance:
                    min_distance = distance
                    best_direction = direction
                    
            self.direction = best_direction
            
    def draw(self, screen):
        # Draw ghost body
        center_x = self.x + CELL_SIZE // 2
        center_y = self.y + CELL_SIZE // 2
        radius = CELL_SIZE // 2 - 1
        
        if self.mode == "FRIGHTENED":
            # Draw blue frightened ghost with white eyes
            pygame.draw.circle(screen, (33, 50, 99), (center_x, center_y), radius)
            
            # Draw wavy bottom
            points = [(self.x, center_y)]
            for i in range(1, CELL_SIZE):
                wave = radius if self.animation_counter < 5 else -radius
                points.append((self.x + i, center_y + wave * math.sin(i * 2)))
            points.append((self.x + CELL_SIZE, center_y))
            points.append((self.x + CELL_SIZE, self.y + CELL_SIZE))
            points.append((self.x, self.y + CELL_SIZE))
            pygame.draw.polygon(screen, (33, 50, 99), points)
            
            # Draw eyes
            eye_radius = radius // 3
            pygame.draw.circle(screen, WHITE, (center_x - radius//2, center_y - radius//4), eye_radius)
            pygame.draw.circle(screen, WHITE, (center_x + radius//2, center_y - radius//4), eye_radius)
            
            # Draw pupils looking in current direction
            pupil_radius = eye_radius // 2
            pupil_offset = eye_radius // 3
            
            left_pupil_x = center_x - radius//2
            right_pupil_x = center_x + radius//2
            pupil_y = center_y - radius//4
            
            if self.direction == "LEFT":
                pygame.draw.circle(screen, BLUE, (left_pupil_x - pupil_offset, pupil_y), pupil_radius)
                pygame.draw.circle(screen, BLUE, (right_pupil_x - pupil_offset, pupil_y), pupil_radius)
            elif self.direction == "RIGHT":
                pygame.draw.circle(screen, BLUE, (left_pupil_x + pupil_offset, pupil_y), pupil_radius)
                pygame.draw.circle(screen, BLUE, (right_pupil_x + pupil_offset, pupil_y), pupil_radius)
            elif self.direction == "UP":
                pygame.draw.circle(screen, BLUE, (left_pupil_x, pupil_y - pupil_offset), pupil_radius)
                pygame.draw.circle(screen, BLUE, (right_pupil_x, pupil_y - pupil_offset), pupil_radius)
            else:  # DOWN
                pygame.draw.circle(screen, BLUE, (left_pupil_x, pupil_y + pupil_offset), pupil_radius)
                pygame.draw.circle(screen, BLUE, (right_pupil_x, pupil_y + pupil_offset), pupil_radius)
        else:
            # Draw normal colored ghost with eyes
            pygame.draw.circle(screen, self.color, (center_x, center_y), radius)
            
            # Draw wavy bottom
            points = [(self.x, center_y)]
            for i in range(1, CELL_SIZE):
                wave = radius if self.animation_counter < 5 else -radius
                points.append((self.x + i, center_y + wave * math.sin(i * 2)))
            points.append((self.x + CELL_SIZE, center_y))
            points.append((self.x + CELL_SIZE, self.y + CELL_SIZE))
            points.append((self.x, self.y + CELL_SIZE))
            pygame.draw.polygon(screen, self.color, points)
            
            # Draw eyes
            eye_radius = radius // 3
            pygame.draw.circle(screen, WHITE, (center_x - radius//2, center_y - radius//4), eye_radius)
            pygame.draw.circle(screen, WHITE, (center_x + radius//2, center_y - radius//4), eye_radius)
            
            # Draw pupils looking in current direction
            pupil_radius = eye_radius // 2
            pupil_offset = eye_radius // 3
            
            left_pupil_x = center_x - radius//2
            right_pupil_x = center_x + radius//2
            pupil_y = center_y - radius//4
            
            if self.direction == "LEFT":
                pygame.draw.circle(screen, BLACK, (left_pupil_x - pupil_offset, pupil_y), pupil_radius)
                pygame.draw.circle(screen, BLACK, (right_pupil_x - pupil_offset, pupil_y), pupil_radius)
            elif self.direction == "RIGHT":
                pygame.draw.circle(screen, BLACK, (left_pupil_x + pupil_offset, pupil_y), pupil_radius)
                pygame.draw.circle(screen, BLACK, (right_pupil_x + pupil_offset, pupil_y), pupil_radius)
            elif self.direction == "UP":
                pygame.draw.circle(screen, BLACK, (left_pupil_x, pupil_y - pupil_offset), pupil_radius)
                pygame.draw.circle(screen, BLACK, (right_pupil_x, pupil_y - pupil_offset), pupil_radius)
            else:  # DOWN
                pygame.draw.circle(screen, BLACK, (left_pupil_x, pupil_y + pupil_offset), pupil_radius)
                pygame.draw.circle(screen, BLACK, (right_pupil_x, pupil_y + pupil_offset), pupil_radius)

class Game:
    def __init__(self):
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Pac-Man")
        self.clock = pygame.time.Clock()
        
        # Initialize game objects
        self.pacman = PacMan()
        self.ghosts = [
            Ghost(RED, 14*CELL_SIZE, 14*CELL_SIZE, "BLINKY"),    # Blinky (red)
            Ghost(PINK, 13*CELL_SIZE, 18*CELL_SIZE, "PINKY"),       # Pinky (pink)
            Ghost(CYAN, 15*CELL_SIZE, 18*CELL_SIZE, "INKY"),         # Inky (cyan)
            Ghost(ORANGE, 14*CELL_SIZE, 18*CELL_SIZE, "CLYDE")       # Clyde (orange)
        ]
        
        # Game state
        self.state = READY
        self.score = 0
        self.lives = 3
        self.level = 1
        self.dots_eaten = 0
        self.fruit_visible = False
        self.fruit_timer = 0
        self.fruit_type = 0
        
        # Maze layout (28x36)
        self.maze = [
            "############################",
            "#............##............#",
            "#.####.#####.##.#####.####.#",
            "#o####.#####.##.#####.####o#",
            "#..........................#",
            "#.####.##.########.##.####.#",
            "#......##....##....##......#",
            "######.##### ## #####.######",
            "     #.##          ##.#     ",
            "     #.## ###--### ##.#     ",
            "######.## #      # ##.######",
            "      .   #      #   .      ",
            "######.## #      # ##.######",
            "     #.## ######## ##.#     ",
            "     #.##          ##.#     ",
            "######.## ######## ##.######",
            "#............##............#",
            "#.####.#####.##.#####.####.#",
            "#o..##....... ........##..o#",
            "###.##.##.########.##.##.###",
            "#......##....##....##......#",
            "#.##########.##.##########.#",
            "#..........................#",
            "############################"
        ]
        
        # Create dot positions
        self.dots = []
        self.power_pellets = []
        for y in range(len(self.maze)):
            for x in range(len(self.maze[y])):
                if self.maze[y][x] == '.':
                    self.dots.append((x * CELL_SIZE + CELL_SIZE // 2, 
 y * CELL_SIZE + CELL_SIZE // 2))
                elif self.maze[y][x] == 'o':
                    self.power_pellets.append((x * CELL_SIZE + CELL_SIZE // 2, 
 y * CELL_SIZE + CELL_SIZE // 2))
        
        # Fruit types
        self.fruits = ["Cherry", "Strawberry", "Orange", "Apple", "Melon", "Galaxian", "Bell", "Key"]
        self.fruit_points = [100, 300, 500, 700, 1000, 2000, 3000, 5000]
        
    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return False
                elif event.key == pygame.K_UP:
                    self.pacman.next_direction = "UP"
                elif event.key == pygame.K_DOWN:
                    self.pacman.next_direction = "DOWN"
                elif event.key == pygame.K_LEFT:
                    self.pacman.next_direction = "LEFT"
                elif event.key == pygame.K_RIGHT:
                    self.pacman.next_direction = "RIGHT"
                elif event.key == pygame.K_SPACE and self.state != PLAYING:
                    if self.state == GAME_OVER:
                        self.reset_game()
                    self.state = PLAYING
        return True
        
    def update(self):
        if self.state != PLAYING:
            return
            
        # Update Pac-Man
        self.pacman.update(self.maze)
        
        # Update ghosts
        for ghost in self.ghosts:
            ghost.update(self.maze, self.pacman, self.ghosts)
            
        # Check dot collisions
        pacman_grid_x = round(self.pacman.x / CELL_SIZE)
        pacman_grid_y = round(self.pacman.y / CELL_SIZE)
        
        # Check regular dots
        for dot in self.dots[:]:
            dot_x, dot_y = dot
            dot_grid_x = round((dot_x - CELL_SIZE // 2) / CELL_SIZE)
            dot_grid_y = round((dot_y - CELL_SIZE // 2) / CELL_SIZE)
            
            if pacman_grid_x == dot_grid_x and pacman_grid_y == dot_grid_y:
                self.dots.remove(dot)
                self.score += 10
                self.dots_eaten += 1
                
        # Check power pellets
        for pellet in self.power_pellets[:]:
            pellet_x, pellet_y = pellet
            pellet_grid_x = round((pellet_x - CELL_SIZE // 2) / CELL_SIZE)
            pellet_grid_y = round((pellet_y - CELL_SIZE // 2) / CELL_SIZE)
            
            if pacman_grid_x == pellet_grid_x and pacman_grid_y == pellet_grid_y:
                self.power_pellets.remove(pellet)
                self.score += 50
                self.dots_eaten += 1
                
                # Make ghosts frightened
                for ghost in self.ghosts:
                    if not ghost.returning_home:
                        ghost.mode = "FRIGHTENED"
                        ghost.frightened_timer = 10 * 60  # 10 seconds at 60 FPS
                        
        # Check ghost collisions
        for ghost in self.ghosts:
            distance = math.sqrt((self.pacman.x - ghost.x)**2 + (self.pacman.y - ghost.y)**2)
            if distance < CELL_SIZE:
                if ghost.mode == "FRIGHTENED":
                    # Pac-Man eats ghost
                    ghost.returning_home = True
                    self.score += 200 * ((sum(1 for g in self.ghosts if g.mode == "FRIGHTENED")) + 1)
                elif not ghost.returning_home:
                    # Ghost catches Pac-Man
                    self.lives -= 1
                    if self.lives <= 0:
                        self.state = GAME_OVER
                    else:
                        self.reset_level()
                        
        # Handle fruit appearance
        if not self.fruit_visible and self.dots_eaten in [70, 170]:
            self.fruit_visible = True
            self.fruit_timer = 60 * 5  # 5 seconds at 60 FPS
            self.fruit_type = min(self.level - 1, len(self.fruits) - 1)
            
        if self.fruit_visible:
            self.fruit_timer -= 1
            if self.fruit_timer <= 0:
                self.fruit_visible = False
                
            # Check if Pac-Man eats fruit
            if self.fruit_visible:
                fruit_x = 14 * CELL_SIZE
                fruit_y = 18 * CELL_SIZE
                distance = math.sqrt((self.pacman.x - fruit_x)**2 + (self.pacman.y - fruit_y)**2)
                if distance < CELL_SIZE:
                    self.score += self.fruit_points[self.fruit_type]
                    self.fruit_visible = False
                    
        # Check if level is complete
        if len(self.dots) == 0 and len(self.power_pellets) == 0:
            self.level += 1
            self.reset_level()
            
    def reset_level(self):
        # Reset Pac-Man
        self.pacman.reset()
        
        # Reset ghosts
        positions = [
            (14*CELL_SIZE, 14*CELL_SIZE),  # Blinky
            (13*CELL_SIZE, 17*CELL_SIZE),  # Pinky
            (15*CELL_SIZE, 17*CELL_SIZE),  # Inky
            (14*CELL_SIZE, 20*CELL_SIZE)   # Clyde
        ]
        
        for i, ghost in enumerate(self.ghosts):
            ghost.x, ghost.y = positions[i]
            ghost.direction = "LEFT" if i != 3 else "UP"
            ghost.mode = "SCATTER"
            ghost.returning_home = False
            
    def reset_game(self):
        self.score = 0
        self.lives = 3
        self.level = 1
        self.dots_eaten = 0
        self.fruit_visible = False
        self.reset_level()
        
    def draw_maze(self):
        for y in range(len(self.maze)):
            for x in range(len(self.maze[y])):
                if self.maze[y][x] == '#':
                    pygame.draw.rect(self.screen, BLUE, 
 (x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE))

    def draw_dots(self):
        # Draw regular dots
        for dot_x, dot_y in self.dots:
            pygame.draw.circle(self.screen, WHITE, (dot_x, dot_y), 3)
            
        # Draw power pellets
        for pellet_x, pellet_y in self.power_pellets:
            pygame.draw.circle(self.screen, WHITE, (pellet_x, pellet_y), 8)
            
    def draw_fruit(self):
        if self.fruit_visible:
            fruit_x = 14 * CELL_SIZE
            fruit_y = 18 * CELL_SIZE
            
            # Draw fruit based on type
            if self.fruit_type < len(self.fruits):
                # Simple representation of fruits
                pygame.draw.circle(self.screen, RED, (fruit_x, fruit_y), 10)
                
    def draw_ui(self):
        # Draw score
        font = pygame.font.Font(None, 36)
        score_text = font.render(f"SCORE: {self.score}", True, WHITE)
        self.screen.blit(score_text, (10, SCREEN_HEIGHT - 40))
        
        # Draw lives
        lives_text = font.render(f"LIVES: {self.lives}", True, WHITE)
        self.screen.blit(lives_text, (SCREEN_WIDTH - 150, SCREEN_HEIGHT - 40))
        
        # Draw level
        level_text = font.render(f"LEVEL: {self.level}", True, WHITE)
        self.screen.blit(level_text, (SCREEN_WIDTH // 2 - 50, SCREEN_HEIGHT - 40))
        
        # Draw READY text or GAME OVER
        if self.state == READY:
            ready_font = pygame.font.Font(None, 72)
            ready_text = ready_font.render("READY!", True, YELLOW)
            text_rect = ready_text.get_rect(center=(SCREEN_WIDTH//2, SCREEN_HEIGHT//2))
            self.screen.blit(ready_text, text_rect)
        elif self.state == GAME_OVER:
            game_over_font = pygame.font.Font(None, 72)
            game_over_text = game_over_font.render("GAME OVER", True, RED)
            text_rect = game_over_text.get_rect(center=(SCREEN_WIDTH//2, SCREEN_HEIGHT//2))
            self.screen.blit(game_over_text, text_rect)
            
            # Draw restart instruction
            restart_font = pygame.font.Font(None, 36)
            restart_text = restart_font.render("Press SPACE to restart", True, WHITE)
            restart_rect = restart_text.get_rect(center=(SCREEN_WIDTH//2, SCREEN_HEIGHT//2 + 60))
            self.screen.blit(restart_text, restart_rect)
            
    def draw(self):
        # Fill background
        self.screen.fill(BLACK)
        
        # Draw maze
        self.draw_maze()
        
        # Draw dots and power pellets
        self.draw_dots()
        
        # Draw fruit
        self.draw_fruit()
        
        # Draw Pac-Man
        self.pacman.draw(self.screen)
        
        # Draw ghosts
        for ghost in self.ghosts:
            ghost.draw(self.screen)
            
        # Draw UI elements
        self.draw_ui()
        
        # Update display
        pygame.display.flip()
        
    def run(self):
        running = True
        while running:
            running = self.handle_events()
            self.update()
            self.draw()
            self.clock.tick(60)  # 60 FPS
            
        pygame.quit()

# Run the game
if __name__ == "__main__":
    game = Game()
    game.run()
