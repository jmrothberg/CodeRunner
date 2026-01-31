import pygame
import random
import math
import sys

# Initialize pygame
pygame.init()

# Screen dimensions
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Frogger - Classic Arcade Game")

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
BRIGHT_GREEN = (50, 255, 100)
DARK_GREEN = (30, 80, 40)
BLUE = (0, 90, 187)
PURPLE = (128, 64, 160)
BROWN = (139, 69, 19)
YELLOW = (255, 255, 0)
RED = (255, 0, 0)
ORANGE = (255, 165, 0)
GREENISH_BLUE = (0, 150, 200)  # For safe crocodiles

# Game settings
GRID_SIZE = 40
LANE_HEIGHT = GRID_SIZE
FROG_SIZE = GRID_SIZE - 4
FPS = 60

# Frogger game class
class FroggerGame:
    def __init__(self):
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont(None, 36)
        self.small_font = pygame.font.SysFont(None, 28)
        
        # Game state
        self.reset_game()
        
        # Create game objects
        self.create_objects()
        
    def reset_game(self):
        self.frog_x = SCREEN_WIDTH // 2
        self.frog_y = SCREEN_HEIGHT - GRID_SIZE
        self.lives = 3
        self.score = 0
        self.level = 1
        self.timer = 60 * FPS  # REVERT TIME: Back to 60 seconds (was doubled to 120)
        self.game_over = False
        self.win = False
        self.occupied_homes = [False] * 10  # DOUBLE HOME SLOTS: 10 slots instead of 5
        
    def create_objects(self):
        # Traffic lanes (bottom half)
        self.cars = []
        self.trucks = []
        self.bulldozers = []
        
        # River objects (top half)
        self.logs = []
        self.turtles = []
        self.alligators = []
        self.crocodiles = []
        
        # Create traffic
        for i in range(5):
            lane_y = SCREEN_HEIGHT - GRID_SIZE * 2 - i * LANE_HEIGHT
            
            # Alternate directions for lanes
            direction = 1 if i % 2 == 0 else -1
            speed = random.uniform(1.0, 3.0) * self.level
            
            # Create different vehicles based on lane
            if i == 0 or i == 4:
                # Cars in first and last lanes - HALF BAD THINGS: 2 cars per lane (was 3)
                for j in range(2):
                    x_pos = random.randint(-200, SCREEN_WIDTH + 100)
                    car = {
                        'x': x_pos,
                        'y': lane_y,
                        'width': GRID_SIZE * 2,
                        'height': LANE_HEIGHT - 4,
                        'speed': speed * direction,
                        'color': random.choice([RED, YELLOW, BLUE])
                    }
                    self.cars.append(car)
            elif i == 1 or i == 3:
                # Trucks in second and fourth lanes - HALF BAD THINGS: 1 truck per lane (was 2)
                for j in range(1):
                    x_pos = random.randint(-300, SCREEN_WIDTH + 200)
                    truck = {
                        'x': x_pos,
                        'y': lane_y,
                        'width': GRID_SIZE * 4,
                        'height': LANE_HEIGHT - 6,
                        'speed': speed * direction,
                        'color': random.choice([ORANGE, BLUE])
                    }
                    self.trucks.append(truck)
            else:
                # Bulldozers in middle lane - HALF BAD THINGS: 1 bulldozer per lane (was 2)
                for j in range(1):
                    x_pos = random.randint(-400, SCREEN_WIDTH + 300)
                    bulldozer = {
                        'x': x_pos,
                        'y': lane_y,
                        'width': GRID_SIZE * 5,
                        'height': LANE_HEIGHT - 8,
                        'speed': speed * direction,
                        'color': ORANGE  # MAKE BULLDOZERS ORANGE (was PURPLE)
                    }
                    self.bulldozers.append(bulldozer)
        
        # Create river objects
        for i in range(1, 6):
            lane_y = SCREEN_HEIGHT // 2 - GRID_SIZE * 3 - i * LANE_HEIGHT
            lane_y = lane_y + 180
            # Alternate directions for lanes
            direction = -1 if i % 2 == 0 else 1
            speed = random.uniform(1.5, 3.5) * self.level
            
            # Create different river objects based on lane
            if i in [1, 4]:
                # Logs in first and fourth lanes - HALF BAD THINGS: 2 logs per lane (was 3)

                for j in range(2):
                    x_pos = random.randint(-200, SCREEN_WIDTH + 100)
                    log = {
                        'x': x_pos,
                        'y': lane_y,
                        'width': GRID_SIZE * 5,
                        'height': LANE_HEIGHT - 8,
                        'speed': speed * direction,
                        'color': BROWN
                    }
                    self.logs.append(log)
            elif i in [2, 3]:
                # Turtles in second and third lanes - DOUBLE TURTLES: 4 turtles per lane (was 2)
                for j in range(4):
                    x_pos = random.randint(-150, SCREEN_WIDTH + 50)
                    turtle = {
                        'x': x_pos,
                        'y': lane_y,
                        'width': GRID_SIZE * 2,
                        'height': LANE_HEIGHT - 8,
                        'speed': speed * direction,
                        'color': GREEN,
                        'diving': False,
                        'dive_timer': random.randint(200, 600)  # DOUBLE TIME: 200-600 frames (was 100-300)
                    }
                    self.turtles.append(turtle)
            else:
                # Alligators in fifth lane - HALF BAD THINGS: 1 alligator per lane (was 2)
                for j in range(1):
                    x_pos = random.randint(-300, SCREEN_WIDTH + 200)
                    alligator = {
                        'x': x_pos,
                        'y': lane_y,
                        'width': GRID_SIZE * 6,
                        'height': LANE_HEIGHT - 4,
                        'speed': speed * direction,
                        'color': GREEN,
                        'open_jaw': False,
                        'jaw_timer': random.randint(60, 240)  # DOUBLE TIME: 60-240 frames (was 30-120)
                    }
                    self.alligators.append(alligator)
                
                # Crocodiles on logs in fifth lane - MAKE THEM RIDEABLE INSTEAD OF DANGEROUS
                for j in range(1):
                    x_pos = random.randint(-250, SCREEN_WIDTH + 150)
                    crocodile = {
                        'x': x_pos,
                        'y': lane_y,
                        'width': GRID_SIZE * 5,
                        'height': LANE_HEIGHT - 8,
                        'speed': speed * direction,
                        'color': GREENISH_BLUE,  # SAFE CROCODILES: Greenish-blue to distinguish from dangerous green alligators
                        'open_jaw': False,
                        'jaw_timer': random.randint(40, 200)  # DOUBLE TIME: 40-200 frames (was 20-100)
                    }
                    # TREAT CROCODILES AS RIDEABLE LOGS INSTEAD OF DANGEROUS
                    self.logs.append(crocodile)
                    # DON'T add to dangerous crocodiles list
                    # self.crocodiles.append(crocodile)

    def draw_background(self):
        # Draw grass areas - ADJUST FOR HOME SLOTS: Top grass starts below home slots
        pygame.draw.rect(screen, DARK_GREEN, (0, GRID_SIZE, SCREEN_WIDTH, GRID_SIZE))  # Top grass (below home slots)
        pygame.draw.rect(screen, DARK_GREEN, (0, SCREEN_HEIGHT - GRID_SIZE, SCREEN_WIDTH, GRID_SIZE))  # Bottom grass
        
        # Draw home slots at top - LOWER POSITION: Move below score display
        for i in range(10):
            x_pos = GRID_SIZE//2 + i * (SCREEN_WIDTH - GRID_SIZE) // 9  # Evenly space 10 slots across screen
            color = WHITE if self.occupied_homes[i] else DARK_GREEN
            pygame.draw.rect(screen, color, (x_pos - GRID_SIZE//3, GRID_SIZE//2, GRID_SIZE*2//3, GRID_SIZE//2))
            pygame.draw.rect(screen, BLACK, (x_pos - GRID_SIZE//4, GRID_SIZE//2 + GRID_SIZE//8, GRID_SIZE//2, GRID_SIZE//4), 1)
        
        # Draw road
        road_top = SCREEN_HEIGHT // 2
        road_height = SCREEN_HEIGHT // 4
        pygame.draw.rect(screen, PURPLE, (0, road_top, SCREEN_WIDTH, road_height))
        
        # Draw lane markings on road
        for i in range(1, 5):
            y_pos = road_top + i * LANE_HEIGHT
            for j in range(0, SCREEN_WIDTH, GRID_SIZE * 2):
                pygame.draw.rect(screen, WHITE, (j, y_pos - 2, GRID_SIZE, 4))
        
        # Draw river
        river_top = SCREEN_HEIGHT // 4
        river_height = SCREEN_HEIGHT // 3
        pygame.draw.rect(screen, BLUE, (0, river_top, SCREEN_WIDTH, river_height))
        
        # Draw water ripples in river
        for i in range(20):
            x = random.randint(0, SCREEN_WIDTH)
            y = random.randint(river_top, river_top + river_height)
            pygame.draw.circle(screen, (0, 100, 200), (x, y), 2)

    def draw_frog(self):
        # Draw frog body
        pygame.draw.rect(screen, BRIGHT_GREEN, 
                         (self.frog_x - FROG_SIZE//2, self.frog_y - FROG_SIZE//2, 
                          FROG_SIZE, FROG_SIZE), border_radius=10)
        
        # Draw eyes
        pygame.draw.circle(screen, BLACK, (self.frog_x - 5, self.frog_y - 8), 4)
        pygame.draw.circle(screen, BLACK, (self.frog_x + 5, self.frog_y - 8), 4)
        pygame.draw.circle(screen, WHITE, (self.frog_x - 4, self.frog_y - 9), 1)
        pygame.draw.circle(screen, WHITE, (self.frog_x + 6, self.frog_y - 9), 1)
        
        # Draw legs
        # Front left leg
        pygame.draw.rect(screen, BRIGHT_GREEN, (self.frog_x - 8, self.frog_y - 4, 4, 8))
        # Front right leg
        pygame.draw.rect(screen, BRIGHT_GREEN, (self.frog_x + 6, self.frog_y - 2, 3, 6))
        # Back left leg
        pygame.draw.rect(screen, BRIGHT_GREEN, (self.frog_x - 10, self.frog_y + 4, 5, 7))
        # Back right leg
        pygame.draw.rect(screen, BRIGHT_GREEN, (self.frog_x + 6, self.frog_y + 6, 4, 6))

    def draw_traffic(self):
        # Draw cars
        for car in self.cars:
            pygame.draw.rect(screen, car['color'], 
                             (car['x'], car['y'] - car['height']//2, 
                              car['width'], car['height']), border_radius=5)
            # Car details
            pygame.draw.rect(screen, BLACK, 
                             (car['x'] + 5, car['y'] - car['height']//2 + 3, 
                              car['width'] - 10, car['height'] - 6), 2)
        
        # Draw trucks
        for truck in self.trucks:
            pygame.draw.rect(screen, truck['color'], 
                             (truck['x'], truck['y'] - truck['height']//2, 
                              truck['width'], truck['height']), border_radius=3)
            # Truck cabin
            pygame.draw.rect(screen, BLACK, 
                             (truck['x'] + truck['width'] - 20, truck['y'] - truck['height']//2, 
                              18, truck['height']), border_radius=2)
        
        # Draw bulldozers
        for bulldozer in self.bulldozers:
            pygame.draw.rect(screen, bulldozer['color'], 
                             (bulldozer['x'], bulldozer['y'] - bulldozer['height']//2, 
                              bulldozer['width'], bulldozer['height']), border_radius=3)
            # Bulldozer blade
            pygame.draw.rect(screen, BLACK, 
                             (bulldozer['x'] - 5, bulldozer['y'] - bulldozer['height']//2 + 2, 
                              8, bulldozer['height'] - 4), border_radius=1)

    def draw_river_objects(self):
        # Draw logs
        for log in self.logs:
            pygame.draw.rect(screen, log['color'],  # Use log's color (BROWN for logs, GREENISH_BLUE for safe crocodiles) 
                             (log['x'], log['y'] - log['height']//2, 
                              log['width'], log['height']), border_radius=8)
            # Log details
            for i in range(1, int(log['width']/10)):
                pygame.draw.line(screen, BLACK, 
                                 (log['x'] + i*10, log['y'] - log['height']//2), 
                                 (log['x'] + i*10, log['y'] + log['height']//2 - 1))
        
        # Draw turtles
        for turtle in self.turtles:
            if not turtle['diving']:
                pygame.draw.ellipse(screen, turtle['color'], 
                                    (turtle['x'], turtle['y'] - turtle['height']//3, 
                                     turtle['width'], turtle['height']*2//3))
                
                # Turtle head
                pygame.draw.circle(screen, turtle['color'], 
                                   (turtle['x'] + turtle['width'] - 5, turtle['y']), 6)
        
        # Draw alligators
        for alligator in self.alligators:
            # Alligator body
            pygame.draw.ellipse(screen, GREEN, 
                                (alligator['x'], alligator['y'] - alligator['height']//2, 
                                 alligator['width'], alligator['height']))
            
            # Alligator tail
            points = [
                (alligator['x'], alligator['y']),
                (alligator['x'] - 15, alligator['y'] - 8),
                (alligator['x'] - 15, alligator['y'] + 6)
            ]
            pygame.draw.polygon(screen, GREEN, points)
            
            # Alligator eyes
            eye_offset = 10 if alligator['open_jaw'] else 5
            pygame.draw.circle(screen, RED, (alligator['x'] + alligator['width'] - 20, alligator['y'] - 5), 3)
            pygame.draw.circle(screen, RED, (alligator['x'] + alligator['width'] - 15, alligator['y'] + 5), 4)
            
            # Alligator jaw
            if alligator['open_jaw']:
                pygame.draw.rect(screen, DARK_GREEN, 
                                 (alligator['x'] + alligator['width'] - 25, alligator['y'] - 8, 
                                  20, 16), border_radius=3)
        
        # Draw crocodiles on logs
        for crocodile in self.crocodiles:
            # Crocodile body (on log)
            pygame.draw.ellipse(screen, GREEN, 
                                (crocodile['x'], crocodile['y'] - crocodile['height']//2 + 5, 
                                 crocodile['width'], crocodile['height'] - 10))
            
            # Crocodile eyes
            eye_offset = 8 if crocodile['open_jaw'] else 4
            pygame.draw.circle(screen, RED, (crocodile['x'] + crocodile['width'] - 15, crocodile['y'] - 3), 2)
            pygame.draw.circle(screen, RED, (crocodile['x'] + crocodile['width'] - 8, crocodile['y'] + 3), 3)
            
            # Crocodile jaw
            if crocodile['open_jaw']:
                pygame.draw.rect(screen, DARK_GREEN, 
                                 (crocodile['x'] + crocodile['width'] - 20, crocodile['y'] - 6, 
                                  15, 8), border_radius=3)

    def update_objects(self):
        # Update car positions
        for car in self.cars:
            car['x'] += car['speed']
            if car['speed'] > 0 and car['x'] > SCREEN_WIDTH + 200:
                car['x'] = -200
            elif car['speed'] < 0 and car['x'] < -200:
                car['x'] = SCREEN_WIDTH + 150
        
        # Update truck positions
        for truck in self.trucks:
            truck['x'] += truck['speed']
            if truck['speed'] > 0 and truck['x'] > SCREEN_WIDTH + 300:
                truck['x'] = -300
            elif truck['speed'] < 0 and truck['x'] < -300:
                truck['x'] = SCREEN_WIDTH + 250
        
        # Update bulldozer positions
        for bulldozer in self.bulldozers:
            bulldozer['x'] += bulldozer['speed']
            if bulldozer['speed'] > 0 and bulldozer['x'] > SCREEN_WIDTH + 400:
                bulldozer['x'] = -450
            elif bulldozer['speed'] < 0 and bulldozer['x'] < -450:
                bulldozer['x'] = SCREEN_WIDTH + 350
        
        # Update log positions
        for log in self.logs:
            log['x'] += log['speed']
            if log['speed'] > 0 and log['x'] > SCREEN_WIDTH + 200:
                log['x'] = -200
            elif log['speed'] < 0 and log['x'] < -350:
                log['x'] = SCREEN_WIDTH + 150
        
        # Update turtle positions and diving behavior
        for turtle in self.turtles:
            turtle['x'] += turtle['speed']
            
            # Update dive timer
            turtle['dive_timer'] -= 1
            if turtle['dive_timer'] <= 0:
                turtle['diving'] = not turtle['diving']
                turtle['dive_timer'] = random.randint(200, 600)  # DOUBLE TIME: 200-600 frames (was 100-300)
                
            if turtle['speed'] > 0 and turtle['x'] > SCREEN_WIDTH + 250:
                turtle['x'] = -150
            elif turtle['speed'] < 0 and turtle['x'] < -200:
                turtle['x'] = SCREEN_WIDTH + 150
        
        # Update alligator positions and jaw behavior
        for alligator in self.alligators:
            alligator['x'] += alligator['speed']
            
            # Update jaw timer
            alligator['jaw_timer'] -= 1
            if alligator['jaw_timer'] <= 0:
                alligator['open_jaw'] = not alligator['open_jaw']
                alligator['jaw_timer'] = random.randint(60, 400)  # DOUBLE TIME: 60-400 frames (was 30-200)
                
            if alligator['speed'] > 0 and alligator['x'] > SCREEN_WIDTH + 450:
                alligator['x'] = -300
            elif alligator['speed'] < 0 and alligator['x'] < -350:
                alligator['x'] = SCREEN_WIDTH + 300
        
        # Update crocodile positions and jaw behavior
        for crocodile in self.crocodiles:
            crocodile['x'] += crocodile['speed']
            
            # Update jaw timer
            crocodile['jaw_timer'] -= 1
            if crocodile['jaw_timer'] <= 0:
                crocodile['open_jaw'] = not crocodile['open_jaw']
                crocodile['jaw_timer'] = random.randint(40, 160)  # DOUBLE TIME: 40-160 frames (was 20-80)
                
            if crocodile['speed'] > 0 and crocodile['x'] > SCREEN_WIDTH + 350:
                crocodile['x'] = -300
            elif crocodile['speed'] < 0 and crocodile['x'] < -400:
                crocodile['x'] = SCREEN_WIDTH + 250

    def check_collisions(self):
        # Check if frog is on a vehicle (death)
        frog_rect = pygame.Rect(self.frog_x - FROG_SIZE//3, self.frog_y - FROG_SIZE//3, 
                                FROG_SIZE*2//3, FROG_SIZE*2//3)
        
        # Check car collisions
        for car in self.cars:
            car_rect = pygame.Rect(car['x'], car['y'] - car['height']//4, 
                                   car['width'], car['height']//2)
            if frog_rect.colliderect(car_rect):
                return True
        
        # Check truck collisions
        for truck in self.trucks:
            truck_rect = pygame.Rect(truck['x'], truck['y'] - truck['height']//4, 
                                     truck['width'], truck['height']//2)
            if frog_rect.colliderect(truck_rect):
                return True
        
        # Check bulldozer collisions
        for bulldozer in self.bulldozers:
            bulldozer_rect = pygame.Rect(bulldozer['x'], bulldozer['y'] - bulldozer['height']//4, 
                                         bulldozer['width'], bulldozer['height']//2)
            if frog_rect.colliderect(bulldozer_rect):
                return True
        
        # Check water collisions (unless on log or turtle)
        if SCREEN_HEIGHT // 4 <= self.frog_y <= SCREEN_HEIGHT // 4 + SCREEN_HEIGHT // 3:
            on_log_or_turtle = False
            
            # Check if on a log
            for log in self.logs:
                log_rect = pygame.Rect(log['x'], log['y'] - log['height']//2, 
                                       log['width'], log['height'])
                if frog_rect.colliderect(log_rect):
                    on_log_or_turtle = True
                    # Move frog with the log
                    self.frog_x += log['speed']
            
            # Check if on a turtle (and not diving)
            for turtle in self.turtles:
                if not turtle['diving']:
                    turtle_rect = pygame.Rect(turtle['x'], turtle['y'] - turtle['height']//3, 
                                              turtle['width'], turtle['height']*2//3)
                    if frog_rect.colliderect(turtle_rect):
                        on_log_or_turtle = True
                        # Move frog with the turtle
                        self.frog_x += turtle['speed']
            
            # Check if touching an alligator or crocodile (even if open jaw)
            for alligator in self.alligators:
                alligator_rect = pygame.Rect(alligator['x'], alligator['y'] - alligator['height']//2, 
                                             alligator['width'], alligator['height'])
                if frog_rect.colliderect(alligator_rect):
                    return True
            
            for crocodile in self.crocodiles:
                crocodile_rect = pygame.Rect(crocodile['x'], crocodile['y'] - crocodile['height']//4,
                                             crocodile['width'], crocodile['height']//2)
                if frog_rect.colliderect(crocodile_rect):
                    return True
            
            # If not on log or turtle, and in water area, it's death
            if not on_log_or_turtle:
                return True
        
        return False

    def check_home(self):
        # Check if frog reached a home slot - DOUBLE HOME SLOTS: Handle 10 slots
        if self.frog_y <= GRID_SIZE * 1.5:
            for i in range(10):
                x_pos = GRID_SIZE//2 + i * (SCREEN_WIDTH - GRID_SIZE) // 9  # Same positioning as drawing
                home_rect = pygame.Rect(x_pos - GRID_SIZE//3, GRID_SIZE//2,
                                        GRID_SIZE*2//3, GRID_SIZE//2)

                frog_rect = pygame.Rect(self.frog_x - FROG_SIZE//2, self.frog_y - FROG_SIZE//2,
                                        FROG_SIZE, FROG_SIZE)

                if frog_rect.colliderect(home_rect) and not self.occupied_homes[i]:
                    self.occupied_homes[i] = True
                    self.score += 50

                    # Bonus points for remaining time
                    bonus = self.timer // FPS * 10
                    self.score += bonus

                    # Check if all homes are occupied (level complete)
                    if all(self.occupied_homes):
                        self.level_up()
                    else:
                        self.reset_frog_position()

                    return True
        return False

    def level_up(self):
        self.level += 1
        self.timer = 60 * FPS  # REVERT TIME: Reset timer to 60 seconds (was doubled to 120)
        self.occupied_homes = [False] * 10  # DOUBLE HOME SLOTS: Reset to 10 slots instead of 5
        
        # Create new objects for next level (faster)
        self.create_objects()
        
        if self.level > 5:  # Win condition after 5 levels
            self.win = True

    def reset_frog_position(self):
        self.frog_x = SCREEN_WIDTH // 2
        self.frog_y = SCREEN_HEIGHT - GRID_SIZE

    def draw_ui(self):
        # Draw score
        score_text = self.font.render(f"Score: {self.score}", True, WHITE)
        screen.blit(score_text, (10, 10))
        
        # Draw lives
        lives_text = self.font.render(f"Lives: {self.lives}", True, WHITE)
        screen.blit(lives_text, (SCREEN_WIDTH - 150, 10))
        
        # Draw level
        level_text = self.font.render(f"Level: {self.level}", True, WHITE)
        screen.blit(level_text, (SCREEN_WIDTH // 2 - 50, 10))
        
        # Draw timer bar
        pygame.draw.rect(screen, RED, (10, SCREEN_HEIGHT - 30, SCREEN_WIDTH - 20, 20), border_radius=5)
        time_ratio = self.timer / (60 * FPS)
        pygame.draw.rect(screen, GREEN, 
                         (10, SCREEN_HEIGHT - 30, int((SCREEN_WIDTH - 20) * time_ratio), 20), 
                         border_radius=5)

    def draw_game_over(self):
        overlay = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
        overlay.set_alpha(180)
        overlay.fill(BLACK)
        screen.blit(overlay, (0, 0))
        
        if self.win:
            game_over_text = self.font.render("YOU WIN!", True, YELLOW)
        else:
            game_over_text = self.font.render("GAME OVER", True, RED)
            
        score_text = self.font.render(f"Final Score: {self.score}", True, WHITE)
        restart_text = self.small_font.render("Press SPACE to play again", True, WHITE)
        
        screen.blit(game_over_text, (SCREEN_WIDTH//2 - game_over_text.get_width()//2, SCREEN_HEIGHT//2 - 60))
        screen.blit(score_text, (SCREEN_WIDTH//2 - score_text.get_width()//2, SCREEN_HEIGHT//2))
        screen.blit(restart_text, (SCREEN_WIDTH//2 - restart_text.get_width()//2, SCREEN_HEIGHT//2 + 60))

    def run(self):
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                
                # Handle frog movement
                elif event.type == pygame.KEYDOWN and not self.game_over:
                    if event.key == pygame.K_UP:
                        self.frog_y -= GRID_SIZE
                    elif event.key == pygame.K_DOWN:
                        self.frog_y += GRID_SIZE
                    elif event.key == pygame.K_LEFT:
                        self.frog_x -= GRID_SIZE
                    elif event.key == pygame.K_RIGHT:
                        self.frog_x += GRID_SIZE
                    
                    # Keep frog on screen
                    if self.frog_x < 0:
                        self.frog_x = 0
                    elif self.frog_x > SCREEN_WIDTH:
                        self.frog_x = SCREEN_WIDTH
                    
                    if self.frog_y < 0:
                        self.frog_y = 0
                    elif self.frog_y > SCREEN_HEIGHT - GRID_SIZE:
                        self.frog_y = SCREEN_HEIGHT - GRID_SIZE
                
                # Restart game after game over
                elif event.type == pygame.KEYDOWN and self.game_over:
                    if event.key == pygame.K_SPACE:
                        self.reset_game()
                        self.create_objects()
            
            if not self.game_over:
                # Update game objects
                self.update_objects()
                
                # Check for collisions
                if self.check_collisions():
                    self.lives -= 1
                    if self.lives <= 0:
                        self.game_over = True
                    else:
                        self.reset_frog_position()
                
                # Check if frog reached home
                self.check_home()
                
                # Update timer
                self.timer -= 1
                if self.timer <= 0:
                    self.lives -= 1
                    if self.lives <= 0:
                        self.game_over = True
                    else:
                        self.reset_frog_position()
                        self.timer = 60 * FPS  # REVERT TIME: Reset timer to 60 seconds (was doubled to 120)
            
            # Draw everything
            screen.fill(BLACK)
            self.draw_background()
            self.draw_traffic()
            self.draw_river_objects()
            self.draw_frog()
            self.draw_ui()
            
            if self.game_over or self.win:
                self.draw_game_over()
            
            pygame.display.flip()
            self.clock.tick(FPS)

# Run the game
if __name__ == "__main__":
    game = FroggerGame()
    game.run()
    pygame.quit()
    sys.exit()