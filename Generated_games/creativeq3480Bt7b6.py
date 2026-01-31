import pygame
import sys
import random
import math

# Initialize Pygame
pygame.init()

# Screen dimensions
SCREEN_WIDTH = 1024
SCREEN_HEIGHT = 768
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Adventure Quest: The Crystal Caverns")

# Colors
SKY_BLUE = (135, 206, 235)
GRASS_GREEN = (34, 139, 34)
DIRT_BROWN = (139, 69, 19)
STONE_GRAY = (128, 128, 128)
CRYSTAL_BLUE = (0, 150, 255)
GOLD = (255, 215, 0)
RED = (220, 60, 40)
PLAYER_COLOR = (70, 90, 200)
ENEMY_RED = (200, 50, 30)
HEART_RED = (255, 50, 100)
TEXT_COLOR = (240, 240, 240)

# Game variables
clock = pygame.time.Clock()
FPS = 60

# Player class
class Player:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.width = 30
        self.height = 50
        self.vel_x = 0
        self.vel_y = 0
        self.jumping = False
        self.direction = 1  # 1 for right, -1 for left
        self.health = 3
        self.coins = 0
        self.invincible = 0
        self.animation_counter = 0
        
    def update(self, platforms):
        # Apply gravity
        self.vel_y += 0.8
        
        # Limit fall speed
        if self.vel_y > 15:
            self.vel_y = 15
            
        # Update position
        self.x += self.vel_x
        self.y += self.vel_y
        
        # Check for collisions with platforms
        self.jumping = True
        for platform in platforms:
            if (self.y + self.height >= platform.y and 
                self.y + self.height <= platform.y + 20 and
                self.x + self.width > platform.x and 
                self.x < platform.x + platform.width and
                self.vel_y > 0):
                
                self.y = platform.y - self.height
                self.vel_y = 0
                self.jumping = False
                
        # Boundary checks
        if self.x < 0:
            self.x = 0
        if self.x > SCREEN_WIDTH - self.width:
            self.x = SCREEN_WIDTH - self.width
            
        # Death condition (falling off screen)
        if self.y > SCREEN_HEIGHT + 100:
            self.health = 0
            
        # Update invincibility timer
        if self.invincible > 0:
            self.invincible -= 1
            
        # Update animation counter
        self.animation_counter = (self.animation_counter + 1) % 20
    
    def jump(self):
        if not self.jumping:
            self.vel_y = -18
            self.jumping = True
    
    def draw(self, screen):
        # Draw player with simple animation
        color = PLAYER_COLOR
        if self.invincible > 0 and self.invincible % 4 < 2:
            color = (min(color[0] + 50, 255), min(color[1] + 30, 255), min(color[2] + 30, 255))
            
        pygame.draw.rect(screen, color, (self.x, self.y, self.width, self.height))
        
        # Draw eyes
        eye_offset = 8 if self.direction > 0 else -8
        pygame.draw.circle(screen, (240, 240, 10), 
                          (int(self.x + self.width/2 + eye_offset), int(self.y + 15)), 5)
        pygame.draw.circle(screen, (30, 70, 120), 
                          (int(self.x + self.width/2 + eye_offset), int(self.y + 15)), 3)
        
        # Draw simple animation for legs
        leg_offset = math.sin(self.animation_counter * 0.3) * 3 if abs(self.vel_x) > 0 else 0
        pygame.draw.line(screen, (40, 60, 200), 
                        (self.x + self.width/3, self.y + self.height), 
                        (self.x + self.width/3 - leg_offset, self.y + self.height + 15), 4)
        pygame.draw.line(screen, (40, 60, 200), 
                        (self.x + 2*self.width/3, self.y + self.height), 
                        (self.x + 2*self.width/3 - leg_offset, self.y + self.height + 15), 4)
        
        # Draw arms
        arm_offset = math.sin(self.animation_counter * 0.3) * 2 if abs(self.vel_x) > 0 else 0
        pygame.draw.line(screen, (40, 60, 200), 
                        (self.x + self.width/3, self.y + 20), 
                        (self.x - arm_offset, self.y + 15), 4)
        pygame.draw.line(screen, (40, 60, 200), 
                        (self.x + 2*self.width/3, self.y + 20), 
                        (self.x + self.width + arm_offset, self.y + 15), 4)

# Platform class
class Platform:
    def __init__(self, x, y, width, height, color=GRASS_GREEN):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.color = color
        
    def draw(self, screen):
        pygame.draw.rect(screen, self.color, (self.x, self.y, self.width, self.height))
        # Add some texture to platforms
        for i in range(0, int(self.width), 10):
            pygame.draw.line(screen, (min(self.color[0], 255), 
                                     min(self.color[1], 255), 
                                     min(self.color[2], 255)), 
                            (self.x + i, self.y), 
                            (self.x + i, self.y + 5), 2)

# Coin class
class Coin:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.radius = 10
        self.animation_counter = random.randint(0, 10)
        
    def update(self):
        self.animation_counter += 0.2
        
    def draw(self, screen):
        # Pulsing effect
        pulse = math.sin(self.animation_counter) * 3
        color = (min(GOLD[0] + int(pulse*5), 255), 
                min(GOLD[1] + int(pulse*5), 255), 
                GOLD[2])
        
        pygame.draw.circle(screen, color, (int(self.x), int(self.y)), self.radius)
        pygame.draw.circle(screen, (0, 0, 0), (int(self.x), int(self.y)), self.radius, 1)
        
        # Draw $ symbol
        font = pygame.font.SysFont(None, 24)
        text = font.render("$", True, (50, 30, 0))
        screen.blit(text, (self.x - text.get_width()//2, self.y - text.get_height()//2))

# Enemy class
class Enemy:
    def __init__(self, x, y, patrol_distance=100):
        self.x = x
        self.y = y
        self.width = 35
        self.height = 35
        self.vel_x = random.choice([-1, 1]) * 2
        self.start_x = x
        self.patrol_distance = patrol_distance
        self.animation_counter = 0
        
    def update(self, platforms):
        # Move enemy
        self.x += self.vel_x
        self.animation_counter += 0.2
        
        # Patrol behavior
        if abs(self.x - self.start_x) > self.patrol_distance:
            self.vel_x *= -1
            
        # Boundary checks
        if self.x < 50:
            self.x = 50
            self.vel_x *= -1
        if self.x > SCREEN_WIDTH - self.width - 50:
            self.x = SCREEN_WIDTH - self.width - 50
            self.vel_x *= -1
            
        # Simple gravity for enemies too
        self.y += 3
        
        # Check for collisions with platforms
        for platform in platforms:
            if (self.y + self.height >= platform.y and 
                self.y + self.height <= platform.y + 15 and
                self.x + self.width > platform.x and 
                self.x < platform.x + platform.width):
                
                self.y = platform.y - self.height
                
    def draw(self, screen):
        # Draw enemy body with simple animation
        offset = math.sin(self.animation_counter) * 2
        
        pygame.draw.rect(screen, ENEMY_RED, (self.x, self.y, self.width, self.height))
        
        # Draw eyes
        eye_direction = -5 if self.vel_x > 0 else 5
        pygame.draw.circle(screen, (240, 10, 30), 
                          (int(self.x + self.width/2 + eye_direction), int(self.y + 15)), 6)
        pygame.draw.circle(screen, (0, 0, 0), 
                          (int(self.x + self.width/2 + eye_direction), int(self.y + 15)), 3)
        
        # Draw legs
        leg_offset = math.sin(self.animation_counter * 3) * 3
        for i in range(3):
            pygame.draw.line(screen, (80, 20, 20), 
                            (self.x + self.width/4 + i*self.width/4, self.y + self.height), 
                            (self.x + self.width/4 + i*self.width/4 - leg_offset, self.y + self.height + 15), 3)
        
        # Draw spikes
        for i in range(5):
            spike_x = self.x + i * (self.width / 4) - offset
            pygame.draw.polygon(screen, (200, 0, 80), [
                (spike_x, self.y),
                (spike_x - 3, self.y - 10),
                (spike_x + 3, self.y - 10)
            ])

# Crystal class
class Crystal:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.width = 25
        self.height = 40
        self.animation_counter = random.randint(0, 10)
        
    def update(self):
        self.animation_counter += 0.1
        
    def draw(self, screen):
        # Pulsing effect for crystal
        pulse = math.sin(self.animation_counter) * 5 + 5
        
        # Draw crystal with glow effect
        points = [
            (self.x + self.width/2, self.y),
            (self.x + self.width, self.y + self.height/3),
            (self.x + 3*self.width/4, self.y + self.height),
            (self.x + self.width/4, self.y + self.height),
            (self.x, self.y + self.height/3)
        ]
        
        # Draw glow
        pygame.draw.polygon(screen, (100, 200, 255, 100), points)
        
        # Draw crystal body with gradient effect
        for i in range(5):
            offset = i * 2
            color_value = min(100 + int(pulse) + offset*3, 240)
            pygame.draw.polygon(screen, (color_value, color_value+30, 255), [
                (p[0], p[1] + offset) for p in points
            ])
            
        # Draw crystal tip
        pygame.draw.circle(screen, (200, 90, 60), 
                          (int(self.x + self.width/2), int(self.y)), 5)

# Particle system for effects
class Particle:
    def __init__(self, x, y, color):
        self.x = x
        self.y = y
        self.vel_x = random.uniform(-3, 3)
        self.vel_y = random.uniform(-5, -1)
        self.size = random.randint(2, 6)
        self.color = color
        self.life = 30
        
    def update(self):
        self.x += self.vel_x
        self.y += self.vel_y
        self.vel_y += 0.1  # Gravity
        self.life -= 1
        
    def draw(self, screen):
        alpha = min(255, self.life * 8)
        pygame.draw.circle(screen, self.color, (int(self.x), int(self.y)), self.size)

# Game class to manage levels and game state
class Game:
    def __init__(self):
        self.player = Player(100, SCREEN_HEIGHT - 200)
        self.platforms = []
        self.coins = []
        self.enemies = []
        self.crystals = []
        self.particles = []
        self.level = 1
        self.game_state = "playing"  # playing, game_over, level_complete
        self.camera_x = 0
        self.camera_y = 0
        self.background_offset = 0
        self.create_level()
        
    def create_level(self):
        self.platforms = []
        self.coins = []
        self.enemies = []
        self.crystals = []
        self.particles = []
        
        # Create platforms for level 1 (ground and floating platforms)
        if self.level == 1:
            # Ground
            self.platforms.append(Platform(0, SCREEN_HEIGHT - 50, SCREEN_WIDTH*2, 50, DIRT_BROWN))
            
            # Floating platforms
            self.platforms.append(Platform(300, SCREEN_HEIGHT - 180, 200, 20, GRASS_GREEN))
            self.platforms.append(Platform(600, SCREEN_HEIGHT - 280, 150, 20, GRASS_GREEN))
            self.platforms.append(Platform(900, SCREEN_HEIGHT - 380, 180, 20, GRASS_GREEN))
            self.platforms.append(Platform(1200, SCREEN_HEIGHT - 280, 150, 20, GRASS_GREEN))
            self.platforms.append(Platform(1500, SCREEN_HEIGHT - 180, 200, 20, GRASS_GREEN))
            
            # Coins
            for i in range(10):
                x = random.randint(200, SCREEN_WIDTH*3)
                y = random.randint(SCREEN_HEIGHT - 450, SCREEN_HEIGHT - 100)
                self.coins.append(Coin(x, y))
                
            # Enemies
            self.enemies.append(Enemy(500, SCREEN_HEIGHT - 100))
            self.enemies.append(Enemy(800, SCREEN_HEIGHT - 320, 80))
            self.enemies.append(Enemy(1300, SCREEN_HEIGHT - 160, 120))
            
            # Crystals
            self.crystals.append(Crystal(SCREEN_WIDTH*1.5, SCREEN_HEIGHT - 400))
            
        elif self.level == 2:
            # More complex level with different platforms and enemies
            self.platforms.append(Platform(0, SCREEN_HEIGHT - 30, SCREEN_WIDTH*2, 30, DIRT_BROWN))
            
            # Stepped platforms
            for i in range(8):
                self.platforms.append(Platform(250 + i*150, SCREEN_HEIGHT - 100 - i*50, 100, 20, GRASS_GREEN))
                
            # Additional platforms
            self.platforms.append(Platform(700, SCREEN_HEIGHT - 300, 180, 20, GRASS_GREEN))
            self.platforms.append(Platform(1050, SCREEN_HEIGHT - 200, 160, 20, GRASS_GREEN))
            self.platforms.append(Platform(1400, SCREEN_HEIGHT - 350, 200, 20, GRASS_GREEN))
            
            # Coins
            for i in range(15):
                x = random.randint(200, SCREEN_WIDTH*3)
                y = random.randint(SCREEN_HEIGHT - 450, SCREEN_HEIGHT - 100)
                self.coins.append(Coin(x, y))
                
            # Enemies
            self.enemies.append(Enemy(400, SCREEN_HEIGHT - 80, 100))
            self.enemies.append(Enemy(700, SCREEN_HEIGHT - 340, 60))
            self.enemies.append(Enemy(950, SCREEN_HEIGHT - 240, 100))
            self.enemies.append(Enemy(1200, SCREEN_HEIGHT - 80, 150))
            
            # Crystals
            self.crystals.append(Crystal(SCREEN_WIDTH*1.3, SCREEN_HEIGHT - 400))
            self.crystals.append(Crystal(SCREEN_WIDTH*1.7, SCREEN_HEIGHT - 200))
        
        elif self.level == 3:
            # Final level with more challenging layout
            self.platforms.append(Platform(0, SCREEN_HEIGHT - 30, SCREEN_WIDTH*2, 30, DIRT_BROWN))
            
            # Complex platform arrangement
            for i in range(5):
                self.platforms.append(Platform(150 + i*300, SCREEN_HEIGHT - 150 - i*40, 180, 20, GRASS_GREEN))
                
            # Additional platforms
            self.platforms.append(Platform(400, SCREEN_HEIGHT - 350, 150, 20, GRASS_GREEN))
            self.platforms.append(Platform(750, SCREEN_HEIGHT - 280, 200, 20, GRASS_GREEN))
            self.platforms.append(Platform(1100, SCREEN_HEIGHT - 400, 160, 20, GRASS_GREEN))
            self.platforms.append(Platform(1500, SCREEN_HEIGHT - 250, 180, 20, GRASS_GREEN))
            
            # Coins
            for i in range(20):
                x = random.randint(200, SCREEN_WIDTH*3)
                y = random.randint(SCREEN_HEIGHT - 450, SCREEN_HEIGHT - 70)
                self.coins.append(Coin(x, y))
                
            # Enemies
            self.enemies.append(Enemy(300, SCREEN_HEIGHT - 80, 120))
            self.enemies.append(Enemy(600, SCREEN_HEIGHT - 450, 80))
            self.enemies.append(Enemy(900, SCREEN_HEIGHT - 320, 100))
            self.enemies.append(Enemy(1200, SCREEN_HEIGHT - 80, 70))
            self.enemies.append(Enemy(1400, SCREEN_HEIGHT - 290, 150))
            
            # Crystals
            self.crystals.append(Crystal(SCREEN_WIDTH*1.2, SCREEN_HEIGHT - 430))
            self.crystals.append(Crystal(SCREEN_WIDTH*1.6, SCREEN_HEIGHT - 280))
            self.crystals.append(Crystal(SCREEN_WIDTH*1.9, SCREEN_HEIGHT - 200))
        
        # Reset player position
        self.player.x = 50
        self.player.y = SCREEN_HEIGHT - 150
    
    def update(self):
        if self.game_state != "playing":
            return
            
        # Update background offset for parallax effect
        self.background_offset = (self.camera_x // 3) % 200
        
        # Update player
        self.player.update(self.platforms)
        
        # Update coins
        for coin in self.coins:
            coin.update()
            
        # Update enemies
        for enemy in self.enemies:
            enemy.update(self.platforms)
            
        # Update crystals
        for crystal in self.crystals:
            crystal.update()
            
        # Update particles
        for particle in self.particles[:]:
            particle.update()
            if particle.life <= 0:
                self.particles.remove(particle)
        
        # Check coin collisions
        for coin in self.coins[:]:
            distance = math.sqrt((self.player.x + self.player.width/2 - coin.x)**2 + 
                                (self.player.y + self.player.height/2 - coin.y)**2)
            if distance < 30:
                self.coins.remove(coin)
                self.player.coins += 1
                # Add particle effect
                for _ in range(15):
                    self.particles.append(Particle(coin.x, coin.y, GOLD))
        
        # Check crystal collisions (level completion)
        for crystal in self.crystals[:]:
            if (self.player.x < crystal.x + crystal.width and 
                self.player.x + self.player.width > crystal.x and
                self.player.y < crystal.y + crystal.height and
                self.player.y + self.player.height > crystal.y):
                
                # Remove collected crystal
                self.crystals.remove(crystal)
                
                # Add particle effect
                for _ in range(30):
                    self.particles.append(Particle(
                        crystal.x + crystal.width/2, 
                        crystal.y + crystal.height/2, 
                        (100, 200, 255)
                    ))
                    
                # If all crystals collected, level complete
                if len(self.crystals) == 0:
                    self.game_state = "level_complete"
        
        # Check enemy collisions
        for enemy in self.enemies:
            if (self.player.x < enemy.x + enemy.width and 
                self.player.x + self.player.width > enemy.x and
                self.player.y < enemy.y + enemy.height and
                self.player.y + self.player.height > enemy.y):
                
                # Only take damage if not invincible
                if self.player.invincible == 0:
                    self.player.health -= 1
                    self.player.invincible = 60  # Invincibility frames
                    
                    # Add particle effect
                    for _ in range(20):
                        self.particles.append(Particle(
                            self.player.x + self.player.width/2, 
                            self.player.y + self.player.height/2, 
                            RED
                        ))
                    
                    # Knockback player
                    if enemy.x < self.player.x:
                        self.player.vel_x = 5
                    else:
                        self.player.vel_x = -5
                    
                    # Check for game over
                    if self.player.health <= 0:
                        self.game_state = "game_over"
        
        # Update camera to follow player
        if self.player.x > SCREEN_WIDTH / 2:
            self.camera_x = max(0, self.player.x - SCREEN_WIDTH / 2)
            
    def draw(self, screen):
        # Draw sky background with gradient
        for y in range(SCREEN_HEIGHT):
            color_value = max(100, 50 + int(y * (200/SCREEN_HEIGHT)))
            pygame.draw.line(screen, (color_value, color_value, 240), 
                            (0, y), (SCREEN_WIDTH, y))
        
        # Draw distant mountains for parallax effect
        for i in range(10):
            x = (i * 150 - self.background_offset) % (SCREEN_WIDTH + 150)
            pygame.draw.polygon(screen, (80, 100, 90), [
                (x, SCREEN_HEIGHT),
                (x + 75, SCREEN_HEIGHT - 120),
                (x + 150, SCREEN_HEIGHT)
            ])
        
        # Draw clouds
        for i in range(5):
            x = (i * 300 - self.background_offset//4) % (SCREEN_WIDTH + 300)
            pygame.draw.circle(screen, (240, 240, 248), (x, 100), 30)
            pygame.draw.circle(screen, (240, 240, 248), (x + 25, 90), 25)
            pygame.draw.circle(screen, (240, 240, 248), (x - 15, 95), 20)
        
        # Draw platforms
        for platform in self.platforms:
            if platform.x + platform.width > self.camera_x and platform.x < self.camera_x + SCREEN_WIDTH:
                screen.blit(screen, (-self.camera_x, 0))
                platform.draw(screen)
                screen.blit(screen, (0, 0))
        
        # Draw coins
        for coin in self.coins:
            if coin.x > self.camera_x - 50 and coin.x < self.camera_x + SCREEN_WIDTH + 50:
                coin.draw(screen)
                
        # Draw enemies
        for enemy in self.enemies:
            if enemy.x > self.camera_x - 100 and enemy.x < self.camera_x + SCREEN_WIDTH + 100:
                enemy.draw(screen)
        
        # Draw crystals
        for crystal in self.crystals:
            if crystal.x > self.camera_x - 50 and crystal.x < self.camera_x + SCREEN_WIDTH + 50:
                crystal.draw(screen)
                
        # Draw particles
        for particle in self.particles:
            particle.draw(screen)
        
        # Draw player (always visible)
        screen.blit(screen, (-self.camera_x, 0))
        self.player.draw(screen)
        screen.blit(screen, (0, 0))
        
        # Draw UI
        font = pygame.font.SysFont(None, 36)
        
        # Health display
        for i in range(self.player.health):
            pygame.draw.circle(screen, HEART_RED, (30 + i*45, 40), 15)
            pygame.draw.circle(screen, (200, 0, 0), (30 + i*45, 40), 8)
        
        # Coins display
        coin_icon = pygame.Surface((36, 36), pygame.SRCALPHA)
        pygame.draw.circle(coin_icon, GOLD, (18, 18), 12)
        screen.blit(coin_icon, (SCREEN_WIDTH - 90, 25))
        text = font.render(f"{self.player.coins}", True, TEXT_COLOR)
        screen.blit(text, (SCREEN_WIDTH - 40, 30))
        
        # Level display
        level_text = font.render(f"Level: {self.level}", True, TEXT_COLOR)
        screen.blit(level_text, (SCREEN_WIDTH // 2 - level_text.get_width() // 2, 20))
        
        # Game state messages
        if self.game_state == "game_over":
            overlay = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            screen.blit(overlay, (0, 0))
            
            game_over_text = font.render("GAME OVER", True, (255, 50, 30))
            restart_text = font.render("Press R to Restart", True, TEXT_COLOR)
            screen.blit(game_over_text, 
                       (SCREEN_WIDTH // 2 - game_over_text.get_width() // 2, 
                        SCREEN_HEIGHT // 2 - 40))
            screen.blit(restart_text, 
                       (SCREEN_WIDTH // 2 - restart_text.get_width() // 2, 
                        SCREEN_HEIGHT // 2 + 10))
            
        elif self.game_state == "level_complete":
            overlay = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 50))
            screen.blit(overlay, (0, 0))
            
            level_text = font.render(f"LEVEL {self.level} COMPLETE!", True, GOLD)
            coins_text = font.render(f"Coins Collected: {self.player.coins}", True, TEXT_COLOR)
            
            if self.level < 3:
                next_text = font.render("Press SPACE for Next Level", True, TEXT_COLOR)
            else:
                next_text = font.render("Congratulations! You Win!", True, GOLD)
                
            screen.blit(level_text, 
                       (SCREEN_WIDTH // 2 - level_text.get_width() // 2, 
                        SCREEN_HEIGHT // 3))
            screen.blit(coins_text, 
                       (SCREEN_WIDTH // 2 - coins_text.get_width() // 2, 
                        SCREEN_HEIGHT // 3 + 50))
            screen.blit(next_text, 
                       (SCREEN_WIDTH // 2 - next_text.get_width() // 2, 
                        SCREEN_HEIGHT // 3 + 100))

# Create game instance
game = Game()

# Main game loop
running = True
while running:
    # Handle events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
            
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE and not game.player.jumping:
                game.player.jump()
                
            if event.key == pygame.K_r and game.game_state == "game_over":
                # Reset game
                game.level = 1
                game.create_level()
                game.player.health = 3
                game.player.coins = 0
                game.game_state = "playing"
                
            if (event.key == pygame.K_SPACE and 
                game.game_state == "level_complete" and 
                game.level < 3):
                # Go to next level
                game.level += 1
                game.create_level()
                game.game_state = "playing"
                
            if event.key == pygame.K_ESCAPE:
                running = False
                
    # Handle continuous key presses for movement
    keys = pygame.key.get_pressed()
    game.player.vel_x = 0
    
    if keys[pygame.K_LEFT] or keys[pygame.K_a]:
        game.player.vel_x = -5
        game.player.direction = -1
        
    if keys[pygame.K_RIGHT] or keys[pygame.K_d]:
        game.player.vel_x = 5
        game.player.direction = 1
    
    # Update game state
    game.update()
    
    # Draw everything
    screen.fill(SKY_BLUE)
    game.draw(screen)
    
    # Update display
    pygame.display.flip()
    
    # Cap the frame rate
    clock.tick(FPS)

# Quit Pygame
pygame.quit()
sys.exit()