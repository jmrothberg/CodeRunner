import pygame
import random
import math
from enum import Enum

# Initialize pygame
pygame.init()

# Screen dimensions
WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Alien Abduction Defense")

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 50, 50)
GREEN = (50, 255, 100)
BLUE = (50, 150, 255)
YELLOW = (255, 255, 50)
PURPLE = (200, 50, 255)
CYAN = (50, 255, 255)
ORANGE = (255, 150, 50)

# Game constants
FPS = 60
GRAVITY = 0.2
PLAYER_SPEED = 5
LASER_SPEED = 8
BOMB_SPEED = 3
HUMANOID_SPEED = 1
ALIEN_SPEED = 1.5

class GameState(Enum):
    PLAYING = 1
    GAME_OVER = 2
    VICTORY = 3

class Player:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.width = 40
        self.height = 25
        self.speed = PLAYER_SPEED
        self.lives = 3
        self.score = 0
        self.bombs = 3
        self.shoot_cooldown = 0
        self.invincible_timer = 0
        
    def move(self, dx, dy=0):
        self.x += dx * self.speed
        self.y += dy * self.speed // 2  # Vertical speed is half of horizontal
        
        # Keep player on screen
        if self.x < 0:
            self.x = 0
        elif self.x > WIDTH - self.width:
            self.x = WIDTH - self.width
            
        # Keep vertical movement within bounds
        if self.y < HEIGHT // 2:  # Don't go too high
            self.y = HEIGHT // 2
        elif self.y > HEIGHT - self.height - 30:  # Don't go below playable area
            self.y = HEIGHT - self.height - 30
            
    def shoot_laser(self):
        if self.shoot_cooldown <= 0:
            self.shoot_cooldown = 10
            return Laser(self.x + self.width // 2, self.y)
        return None
        
    def drop_bomb(self):
        if self.bombs > 0:
            self.bombs -= 1
            return Bomb(self.x + self.width // 2, self.y + self.height)
        return None
        
    def update(self):
        if self.shoot_cooldown > 0:
            self.shoot_cooldown -= 1
        if self.invincible_timer > 0:
            self.invincible_timer -= 1
            
    def draw(self, surface):
        # Draw ship body
        pygame.draw.polygon(surface, BLUE, [
            (self.x, self.y + self.height),
            (self.x + self.width // 2, self.y),
            (self.x + self.width, self.y + self.height)
        ])
        
        # Draw cockpit
        pygame.draw.circle(surface, CYAN, (self.x + self.width // 2, self.y + 10), 6)
        
        # Draw engine glow
        if random.random() > 0.3:
            pygame.draw.polygon(surface, ORANGE, [
                (self.x + self.width // 2 - 5, self.y + self.height),
                (self.x + self.width // 2, self.y + self.height + 10),
                (self.x + self.width // 2 + 5, self.y + self.height)
            ])
            
        # Draw invincibility effect
        if self.invincible_timer > 0 and self.invincible_timer % 4 < 2:
            pygame.draw.rect(surface, YELLOW, 
                            (self.x - 5, self.y - 5, self.width + 10, self.height + 30), 2)

class Laser:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.width = 3
        self.height = 15
        self.speed = LASER_SPEED
        
    def update(self):
        self.y -= self.speed
        
    def draw(self, surface):
        pygame.draw.rect(surface, GREEN, (self.x - self.width//2, self.y, self.width, self.height))
        # Add glow effect
        pygame.draw.circle(surface, WHITE, (self.x, self.y), 3)

class Bomb:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.radius = 8
        self.speed = BOMB_SPEED
        self.exploding = False
        self.explosion_radius = 0
        self.max_explosion_radius = 50
        
    def update(self):
        if not self.exploding:
            self.y += self.speed
            # Add some horizontal drift for visual effect
            self.x += random.randint(-1, 1)
            
            # Check if bomb is off screen
            if self.y > HEIGHT + 50:
                return False
        else:
            self.explosion_radius += 3
            if self.explosion_radius > self.max_explosion_radius:
                return False
                
        return True
        
    def explode(self):
        self.exploding = True
        
    def draw(self, surface):
        if not self.exploding:
            # Draw bomb body
            pygame.draw.circle(surface, RED, (self.x, self.y), self.radius)
            # Draw fuse
            pygame.draw.line(surface, YELLOW, 
                            (self.x, self.y - self.radius), 
                            (self.x + 5, self.y - self.radius - 8), 2)
        else:
            # Draw explosion
            pygame.draw.circle(surface, ORANGE, (self.x, self.y), self.explosion_radius)
            pygame.draw.circle(surface, YELLOW, (self.x, self.y), self.explosion_radius * 0.7)
            pygame.draw.circle(surface, RED, (self.x, self.y), self.explosion_radius * 0.4)

class Humanoid:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.width = 12
        self.height = 20
        self.speed = HUMANOID_SPEED
        self.direction = random.choice([-1, 1])  # -1 for left, 1 for right
        self.abducted = False
        self.rescued = False
        self.abduction_progress = 0
        
    def update(self):
        if not self.abducted and not self.rescued:
            self.x += self.speed * self.direction
            
            # Change direction at screen edges
            if self.x < 10 or self.x > WIDTH - 10:
                self.direction *= -1
                
    def draw(self, surface):
        if self.rescued:
            return  # Don't draw rescued humanoids
            
        color = GREEN if not self.abducted else RED
        
        # Draw body
        pygame.draw.rect(surface, color, (self.x - self.width//2, self.y, self.width, self.height))
        
        # Draw head
        pygame.draw.circle(surface, color, (self.x, self.y), 6)
        
        # Draw arms
        arm_y = self.y + 8
        pygame.draw.line(surface, color, 
                        (self.x - self.width//2, arm_y), 
                        (self.x - self.width, arm_y), 2)
        pygame.draw.line(surface, color, 
                        (self.x + self.width//2, arm_y), 
                        (self.x + self.width, arm_y), 2)
        
        # Draw legs
        leg_y = self.y + self.height
        pygame.draw.line(surface, color, 
                        (self.x - 3, leg_y), 
                        (self.x - 8, leg_y + 10), 2)
        pygame.draw.line(surface, color, 
                        (self.x + 3, leg_y), 
                        (self.x + 8, leg_y + 10), 2)
        
        # Draw abduction beam if being abducted
        if self.abducted and self.abduction_progress > 0:
            beam_height = min(self.abduction_progress * 5, HEIGHT - self.y)
            pygame.draw.line(surface, PURPLE, 
                            (self.x, self.y), 
                            (self.x, self.y - beam_height), 2)
            
            # Draw abduction particles
            for i in range(5):
                offset = random.randint(-3, 3)
                size = random.randint(1, 3)
                pygame.draw.circle(surface, PURPLE, 
                                  (self.x + offset, self.y - i * 10), size)

class AlienLander:
    def __init__(self, x):
        self.x = x
        self.y = -50
        self.width = 25
        self.height = 15
        self.speed = ALIEN_SPEED
        self.landed = False
        self.abducting = False
        self.mutated = False
        self.health = 3
        
    def update(self, humanoids):
        if not self.landed:
            self.y += self.speed
            
            # Land when reaching ground level
            if self.y >= HEIGHT - 100:
                self.landed = True
                self.y = HEIGHT - 95
                
        elif not self.abducting and not self.mutated:
            # Find a nearby humanoid to abduct
            for h in humanoids:
                if (not h.abducted and not h.rescued and 
                    abs(h.x - self.x) < 100):
                    self.abducting = True
                    h.abducted = True
                    break
                    
        elif self.abducting:
            # Find the abducted humanoid
            for h in humanoids:
                if h.abducted and not h.rescued:
                    # Increase abduction progress
                    h.abduction_progress += 1
                    
                    # If abduction is complete, mutate the lander
                    if h.abduction_progress >= 20:
                        self.mutated = True
                        self.abducting = False
                        h.abducted = False  # Remove the humanoid (abducted)
                        
    def draw(self, surface):
        color = PURPLE if not self.mutated else RED
        
        # Draw lander body
        pygame.draw.ellipse(surface, color, 
                           (self.x - self.width//2, self.y, self.width, self.height))
        
        # Draw landing gear
        pygame.draw.line(surface, color, 
                        (self.x - 10, self.y + self.height), 
                        (self.x - 5, self.y + self.height + 8), 3)
        pygame.draw.line(surface, color, 
                        (self.x + 10, self.y + self.height), 
                        (self.x + 5, self.y + self.height + 8), 3)
        
        # Draw cockpit
        pygame.draw.circle(surface, CYAN, (self.x, self.y + 5), 4)
        
        # Draw abduction beam if abducting
        if self.abducting:
            for h in humanoids:
                if h.abducted and not h.rescued:
                    pygame.draw.line(surface, PURPLE, 
                                    (self.x, self.y), 
                                    (h.x, h.y), 2)
                    
                    # Draw abduction particles
                    for i in range(5):
                        offset = random.randint(-3, 3)
                        size = random.randint(1, 4)
                        pygame.draw.circle(surface, PURPLE, 
                                          (self.x + offset, self.y + i * 10), size)
        
        # Draw mutation effect
        if self.mutated:
            # Draw mutated appendages
            pygame.draw.line(surface, RED, 
                            (self.x - 15, self.y + 5), 
                            (self.x - 25, self.y - 5), 3)
            pygame.draw.line(surface, RED, 
                            (self.x + 15, self.y + 5), 
                            (self.x + 25, self.y - 5), 3)
            
            # Draw eyes
            pygame.draw.circle(surface, YELLOW, (self.x - 6, self.y + 3), 3)
            pygame.draw.circle(surface, YELLOW, (self.x + 6, self.y + 3), 3)

class MutantAlien:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.width = 40
        self.height = 25
        self.speed_x = random.choice([-1, 1]) * (1 + random.random())
        self.speed_y = -2 - random.random() * 2
        self.health = 5
        
    def update(self):
        self.x += self.speed_x
        self.y += self.speed_y
        
        # Apply gravity
        self.speed_y += GRAVITY * 0.5
        
        # Bounce off walls
        if self.x < 0 or self.x > WIDTH:
            self.speed_x *= -1
            
        # Remove if off screen
        return self.y < HEIGHT + 50
        
    def draw(self, surface):
        # Draw mutated alien body
        pygame.draw.ellipse(surface, RED, 
                           (self.x - self.width//2, self.y, self.width, self.height))
        
        # Draw tentacles
        for i in range(4):
            angle = math.pi/2 + (i-1.5) * 0.5
            end_x = self.x + math.cos(angle) * 20
            end_y = self.y + self.height + math.sin(angle) * 20
            pygame.draw.line(surface, RED, 
                            (self.x + (i-1.5)*8, self.y + self.height), 
                            (end_x, end_y), 3)
        
        # Draw eyes
        pygame.draw.circle(surface, YELLOW, (self.x - 8, self.y + 8), 4)
        pygame.draw.circle(surface, YELLOW, (self.x + 8, self.y + 8), 4)
        pygame.draw.circle(surface, BLACK, (self.x - 8, self.y + 8), 2)
        pygame.draw.circle(surface, BLACK, (self.x + 8, self.y + 8), 2)

class Terrain:
    def __init__(self):
        self.points = []
        self.generate_terrain()
        
    def generate_terrain(self):
        # Generate initial terrain points
        self.points = []
        for x in range(0, WIDTH + 50, 50):
            y = HEIGHT - 50 + random.randint(-20, 10)
            self.points.append((x, y))
            
    def update(self):
        # Scroll terrain to the left
        for i in range(len(self.points)):
            x, y = self.points[i]
            self.points[i] = (x - 1, y)
            
        # Add new point when needed
        if self.points[-1][0] < WIDTH:
            last_y = self.points[-1][1]
            new_y = max(HEIGHT - 80, min(last_y + random.randint(-5, 5), HEIGHT - 30))
            self.points.append((self.points[-1][0] + 20, new_y))
            
        # Remove points that are off screen
        while self.points and self.points[0][0] < -50:
            self.points.pop(0)
            
    def draw(self, surface):
        if len(self.points) > 1:
            pygame.draw.lines(surface, GREEN, False, self.points, 3)
            
            # Draw ground details
            for i in range(len(self.points) - 1):
                x1, y1 = self.points[i]
                x2, y2 = self.points[i+1]
                
                # Draw grass tufts
                if random.random() > 0.7:
                    mid_x = (x1 + x2) / 2
                    pygame.draw.line(surface, GREEN, 
                                    (mid_x, y1), 
                                    (mid_x, y1 - random.randint(5, 15)), 2)

# Create game objects
player = Player(WIDTH // 2, HEIGHT - 80)
lasers = []
bombs = []
humanoids = [Humanoid(random.randint(50, WIDTH-50), HEIGHT - 30) for _ in range(8)]
alien_landers = []
mutant_aliens = []
terrain = Terrain()

# Game state
game_state = GameState.PLAYING
level = 1
spawn_timer = 0
font = pygame.font.SysFont(None, 36)

# Main game loop
clock = pygame.time.Clock()
running = True

while running:
    # Event handling
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE and game_state == GameState.PLAYING:
                laser = player.shoot_laser()
                if laser:
                    lasers.append(laser)
            elif event.key == pygame.K_b and game_state == GameState.PLAYING:
                bomb = player.drop_bomb()
                if bomb:
                    bombs.append(bomb)
            elif event.key == pygame.K_r and game_state != GameState.PLAYING:
                # Reset game
                player = Player(WIDTH // 2, HEIGHT - 80)
                lasers = []
                bombs = []
                humanoids = [Humanoid(random.randint(50, WIDTH-50), HEIGHT - 30) for _ in range(8)]
                alien_landers = []
                mutant_aliens = []
                terrain = Terrain()
                game_state = GameState.PLAYING
                level = 1
                spawn_timer = 0
    
    if game_state == GameState.PLAYING:
        # Player movement
        keys = pygame.key.get_pressed()
        dx, dy = 0, 0
        if keys[pygame.K_LEFT]:
            dx = -1
        if keys[pygame.K_RIGHT]:
            dx = 1
        if keys[pygame.K_UP]:
            dy = -1
        if keys[pygame.K_DOWN]:
            dy = 1
        player.move(dx, dy)
            
        # Update player
        player.update()
        
        # Spawn alien landers
        spawn_timer += 1
        if spawn_timer > max(60 - level * 5, 20):  # Increase spawn rate with level
            spawn_timer = 0
            alien_landers.append(AlienLander(random.randint(50, WIDTH-50)))
            
        # Update lasers
        for laser in lasers[:]:
            laser.update()
            if laser.y < 0:
                lasers.remove(laser)
                
        # Update bombs
        for bomb in bombs[:]:
            if not bomb.update():
                bombs.remove(bomb)
            else:
                # Check if bomb hit ground or aliens to explode
                if not bomb.exploding and bomb.y >= HEIGHT - 50:  # Hit ground
                    bomb.explode()
                        
        # Update humanoids
        for humanoid in humanoids:
            humanoid.update()
            
        # Update alien landers
        for lander in alien_landers[:]:
            lander.update(humanoids)
            
            # Remove landers that have mutated and spawned mutants
            if lander.mutated:
                mutant_aliens.append(MutantAlien(lander.x, lander.y))
                alien_landers.remove(lander)
                
        # Update mutant aliens
        for mutant in mutant_aliens[:]:
            if not mutant.update():
                mutant_aliens.remove(mutant)
                
        # Update terrain
        terrain.update()
        
        # Collision detection: lasers vs aliens
        for laser in lasers[:]:
            hit = False
            
            # Check against landers
            for lander in alien_landers:
                if (abs(laser.x - lander.x) < 20 and 
                    abs(laser.y - lander.y) < 15):
                    lander.health -= 1
                    if lander.health <= 0:
                        alien_landers.remove(lander)
                        player.score += 100
                    lasers.remove(laser)
                    hit = True
                    break
                    
            # Check against mutants
            if not hit:
                for mutant in mutant_aliens:
                    if (abs(laser.x - mutant.x) < 25 and 
                        abs(laser.y - mutant.y) < 15):
                        mutant.health -= 1
                        if mutant.health <= 0:
                            mutant_aliens.remove(mutant)
                            player.score += 200
                        lasers.remove(laser)
                        hit = True
                        break
                        
        # Collision detection: bombs vs aliens
        for bomb in bombs[:]:
            if bomb.exploding:
                # Check against landers
                for lander in alien_landers[:]:
                    distance = math.sqrt((bomb.x - lander.x)**2 + (bomb.y - lander.y)**2)
                    if distance < bomb.explosion_radius + 20:
                        alien_landers.remove(lander)
                        player.score += 100
                        
                # Check against mutants
                for mutant in mutant_aliens[:]:
                    distance = math.sqrt((bomb.x - mutant.x)**2 + (bomb.y - mutant.y)**2)
                    if distance < bomb.explosion_radius + 30:
                        mutant_aliens.remove(mutant)
                        player.score += 200
                        
        # Collision detection: player vs aliens
        for lander in alien_landers:
            if (abs(player.x + player.width//2 - lander.x) < 30 and 
                abs(player.y + player.height//2 - lander.y) < 30):
                if player.invincible_timer <= 0:
                    player.lives -= 1
                    player.invincible_timer = 60  # Invincible for 1 second
                    
        for mutant in mutant_aliens:
            if (abs(player.x + player.width//2 - mutant.x) < 30 and 
                abs(player.y + player.height//2 - mutant.y) < 25):
                if player.invincible_timer <= 0:
                    player.lives -= 1
                    player.invincible_timer = 60
                    
        # Collision detection: bombs vs landers (trigger explosion)
        for bomb in bombs[:]:
            if not bomb.exploding:
                for lander in alien_landers:
                    if (abs(bomb.x - lander.x) < 25 and 
                        abs(bomb.y - lander.y) < 15):
                        bomb.explode()
                        
                for mutant in mutant_aliens:
                    if (abs(bomb.x - mutant.x) < 30 and 
                        abs(bomb.y - mutant.y) < 20):
                        bomb.explode()
        
        # Check for rescued humanoids
        for humanoid in humanoids[:]:
            for humanoid in humanoids[:]:
               if not humanoid.rescued and not humanoid.abducted:
                # Improved collision detection with player
                player_center_x = player.x + player.width // 2
                player_bottom_y = player.y + player.height
                
                distance = math.sqrt((player_center_x - humanoid.x)**2 + 
                                   (player_bottom_y - humanoid.y)**2)
                
                if distance < 30:  # Adjust this value to tune collision sensitivity
                    humanoid.rescued = True
                    player.score += 50
                    
        # Check game over conditions
        if player.lives <= 0:
            game_state = GameState.GAME_OVER
            
        # Check victory condition (rescue all humanoids)
        rescued_count = sum(1 for h in humanoids if h.rescued)
        if rescued_count >= len(humanoids) and not any(h.abducted for h in humanoids):
            level += 1
            player.bombs = min(player.bombs + 2, 5)  # Add bombs each level
            # Spawn new humanoids for next level
            humanoids = [Humanoid(random.randint(50, WIDTH-50), HEIGHT - 30) for _ in range(8)]
            
        # Increase score over time
        player.score += 1
        
    # Drawing
    screen.fill(BLACK)
    
    # Draw stars in background
    for i in range(100):
        x = (i * 73) % WIDTH
        y = (i * 57) % HEIGHT
        size = (i % 3) + 1
        brightness = 200 - (i % 50)
        pygame.draw.circle(screen, (brightness, brightness, brightness), (x, y), size)
    
    # Draw terrain
    terrain.draw(screen)
    
    # Draw game objects
    for humanoid in humanoids:
        humanoid.draw(screen)
        
    for lander in alien_landers:
        lander.draw(screen)
        
    for mutant in mutant_aliens:
        mutant.draw(screen)
        
    for laser in lasers:
        laser.draw(screen)
        
    for bomb in bombs:
        bomb.draw(screen)
        
    player.draw(screen)
    
    # Draw UI
    score_text = font.render(f"Score: {player.score}", True, WHITE)
    lives_text = font.render(f"Lives: {player.lives}", True, WHITE)
    bombs_text = font.render(f"Bombs: {player.bombs}", True, WHITE)
    level_text = font.render(f"Level: {level}", True, WHITE)
    
    screen.blit(score_text, (10, 10))
    screen.blit(lives_text, (10, 50))
    screen.blit(bombs_text, (WIDTH - 150, 10))
    screen.blit(level_text, (WIDTH // 2 - 50, 10))
    
    # Draw rescued count
    rescued_count = sum(1 for h in humanoids if h.rescued)
    rescued_text = font.render(f"Rescued: {rescued_count}/{len(humanoids)}", True, GREEN)
    screen.blit(rescued_text, (WIDTH // 2 - 80, 50))
    
    # Draw game over or victory message
    if game_state == GameState.GAME_OVER:
        game_over_text = font.render("GAME OVER! Press R to restart", True, RED)
        text_rect = game_over_text.get_rect(center=(WIDTH//2, HEIGHT//2))
        screen.blit(game_over_text, text_rect)
        
    elif game_state == GameState.VICTORY:
        victory_text = font.render("VICTORY! Press R to play again", True, GREEN)
        text_rect = victory_text.get_rect(center=(WIDTH//2, HEIGHT//2))
        screen.blit(victory_text, text_rect)
    
    pygame.display.flip()
    clock.tick(FPS)

pygame.quit()