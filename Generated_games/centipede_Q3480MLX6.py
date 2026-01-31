from icecream import ic
import pygame
import random
import math

# Initialize pygame
pygame.init()

# Screen dimensions
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
GRID_SIZE = 20
GRID_WIDTH = SCREEN_WIDTH // GRID_SIZE
GRID_HEIGHT = SCREEN_HEIGHT // GRID_SIZE

# Colors (ultra-bright vivid colors as requested)
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)
YELLOW = (255, 255, 0)
RED = (255, 0, 100)  # Magenta-red for spiders
PURPLE = (200, 0, 255)  # Purple for fleas
CYAN = (0, 255, 255)  # Cyan for scorpions
WHITE = (255, 255, 255)
ORANGE = (255, 165, 0)
DARK_GREEN = (0, 100, 0)

# Game settings
PLAYER_SPEED = 5
BULLET_SPEED = 8
CENTIPEDE_SPEED = 3
FLEA_SPEED = 2
SPIDER_SPEED = 3
SCORPION_SPEED = 2

# Create the screen
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Centipede - Faithful to the 1981 Atari Original")
clock = pygame.time.Clock()

# Font setup
font = pygame.font.SysFont(None, 36)
small_font = pygame.font.SysFont(None, 24)

class Player:
    def __init__(self):
        self.width = GRID_SIZE
        self.height = GRID_SIZE
        self.x = SCREEN_WIDTH // 2 - self.width // 2
        self.y = SCREEN_HEIGHT - GRID_SIZE * 3
        self.speed = PLAYER_SPEED
        self.lives = 3
        self.score = 0
        self.next_life_score = 12000
        
    def move(self, dx, dy=0):
        # Constrain player to bottom area (last 4 rows)
        new_x = self.x + dx * self.speed
        new_y = self.y + dy * self.speed
        
        # Horizontal constraint
        if 0 <= new_x // GRID_SIZE < GRID_WIDTH:
            self.x = new_x
            
        # Vertical constraint - limit to bottom area
        grid_y = new_y // GRID_SIZE
        if GRID_HEIGHT - 4 <= grid_y < GRID_HEIGHT - 1:
            self.y = new_y
    
    def draw(self):
        # Draw a simple ship-like shape for the player
        pygame.draw.rect(screen, GREEN, (self.x, self.y, self.width, self.height))
        pygame.draw.polygon(screen, YELLOW, [
            (self.x + self.width // 2, self.y - 5),
            (self.x, self.y + self.height),
            (self.x + self.width, self.y + self.height)
        ])
        
    def shoot(self):
        return Bullet(self.x + self.width // 2, self.y)

class Bullet:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.speed = BULLET_SPEED
        self.radius = 3
        
    def move(self):
        self.y -= self.speed
        
    def draw(self):
        pygame.draw.circle(screen, YELLOW, (int(self.x), int(self.y)), self.radius)
        
    def is_off_screen(self):
        return self.y < 0

class Mushroom:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.hits = 4  # Takes 4 hits to destroy
        self.poisoned = False
        
    def hit(self):
        self.hits -= 1
        if self.hits <= 0:
            return True
        elif self.hits == 3:
            self.poisoned = random.random() < 0.2  # 20% chance to become poisoned when damaged
        return False
        
    def draw(self):
        color = CYAN if self.poisoned else RED
        size_factor = self.hits / 4.0
        pygame.draw.circle(screen, color, (self.x + GRID_SIZE//2, self.y + GRID_SIZE//2), 
                          int(GRID_SIZE * 0.4 * size_factor))
        
        # Draw spots on mushroom cap
        if not self.poisoned:
            for i in range(3):
                spot_x = self.x + GRID_SIZE//2 + random.randint(-5, 5)
                spot_y = self.y + GRID_SIZE//2 + random.randint(-5, 5)
                pygame.draw.circle(screen, WHITE, (spot_x, spot_y), 2)

class CentipedeSegment:
    def __init__(self, x, y, is_head=False):
        self.x = x
        self.y = y
        self.width = GRID_SIZE
        self.height = GRID_SIZE
        self.is_head = is_head
        self.direction = 1  # 1 for right, -1 for left
        self.speed = CENTIPEDE_SPEED
        
    def move(self, mushrooms):
        new_x = self.x + self.direction * self.speed
        
        # Check if we need to drop down
        drop_down = False
        if new_x < 0 or new_x >= SCREEN_WIDTH - self.width:
            drop_down = True
        else:
            # Check for mushroom collision
            grid_x = (new_x + self.width // 2) // GRID_SIZE
            grid_y = self.y // GRID_SIZE
            for mushroom in mushrooms:
                if mushroom.x // GRID_SIZE == grid_x and mushroom.y // GRID_SIZE == grid_y:
                    drop_down = True
                    break
        
        if drop_down:
            self.direction *= -1
            self.y += GRID_SIZE
        else:
            self.x = new_x
            
    def draw(self):
        # Draw head differently from body segments
        color = YELLOW if self.is_head else GREEN
        pygame.draw.rect(screen, color, (self.x, self.y, self.width, self.height))
        
        # Add details to make it look more segmented
        segment_width = self.width // 3
        for i in range(3):
            offset_x = i * segment_width
            pygame.draw.line(screen, BLACK, 
                            (self.x + offset_x, self.y), 
                            (self.x + offset_x, self.y + self.height), 1)
        
        # Draw eyes on head
        if self.is_head:
            eye_offset = 5
            pygame.draw.circle(screen, RED, (self.x + eye_offset, self.y + eye_offset), 3)
            pygame.draw.circle(screen, RED, (self.x + self.width - eye_offset, self.y + eye_offset), 3)

class Centipede:
    def __init__(self, length=12):
        self.segments = []
        self.length = length
        # Start at top of screen
        for i in range(length):
            segment_x = i * GRID_SIZE
            segment_y = 0
            is_head = (i == 0)
            self.segments.append(CentipedeSegment(segment_x, segment_y, is_head))
            
    def move(self, mushrooms):
        # Move segments from tail to head so they follow correctly
        for i in range(len(self.segments) - 1, -1, -1):
            if i == 0:  # Head moves normally
                self.segments[i].move(mushrooms)
            else:  # Other segments follow the previous segment
                prev_segment = self.segments[i-1]
                self.segments[i].x = prev_segment.x
                self.segments[i].y = prev_segment.y
                self.segments[i].direction = prev_segment.direction
                
    def draw(self):
        for segment in self.segments:
            segment.draw()
            
    def split_at(self, index):
        # Split the centipede at the given index
        if index <= 0 or index >= len(self.segments) - 1:
            return None
            
        # Create new centipede from segments after the split point
        new_segments = self.segments[index:]
        self.segments = self.segments[:index]
        
        # Reverse direction of new centipede head
        if new_segments:
            new_segments[0].direction *= -1
            
        # Create new centipede object
        new_centipede = Centipede(0)
        new_centipede.segments = new_segments
        return new_centipede

class Flea:
    def __init__(self):
        self.x = random.randint(0, SCREEN_WIDTH - GRID_SIZE)
        self.y = 0
        self.width = GRID_SIZE
        self.height = GRID_SIZE
        self.speed = FLEA_SPEED
        
    def move(self):
        self.y += self.speed
        
    def draw(self):
        # Draw a simple flea shape
        pygame.draw.rect(screen, PURPLE, (self.x, self.y, self.width, self.height))
        # Legs
        for i in range(3):
            leg_y = self.y + i * GRID_SIZE // 2
            pygame.draw.line(screen, PURPLE, 
                            (self.x - 5, leg_y), 
                            (self.x, leg_y), 1)
            pygame.draw.line(screen, PURPLE, 
                            (self.x + self.width + 5, leg_y), 
                            (self.x + self.width, leg_y), 1)
        
    def is_off_screen(self):
        return self.y > SCREEN_HEIGHT

class Spider:
    def __init__(self):
        # Start from either left or right side
        if random.random() < 0.5:
            self.x = -GRID_SIZE
            self.direction = 1
        else:
            self.x = SCREEN_WIDTH
            self.direction = -1
            
        self.y = random.randint(SCREEN_HEIGHT // 2, SCREEN_HEIGHT - GRID_SIZE * 4)
        self.width = GRID_SIZE
        self.height = GRID_SIZE
        self.speed_x = SPIDER_SPEED
        self.speed_y = random.choice([-SPIDER_SPEED, SPIDER_SPEED])
        self.move_counter = 0
        
    def move(self):
        self.x += self.direction * self.speed_x
        self.y += self.speed_y
        self.move_counter += 1
        
        # Change vertical direction periodically for zigzag movement
        if self.move_counter % 20 == 0:
            self.speed_y *= -1
            
    def draw(self):
        # Draw spider body
        pygame.draw.circle(screen, RED, (self.x + self.width//2, self.y + self.height//2), 
                          self.width // 2)
        
        # Draw legs
        for i in range(4):
            leg_y = self.y + i * GRID_SIZE // 3
            pygame.draw.line(screen, RED, 
                            (self.x - 10, leg_y), 
                            (self.x, leg_y), 2)
            pygame.draw.line(screen, RED, 
                            (self.x + self.width + 10, leg_y), 
                            (self.x + self.width, leg_y), 2)
        
    def is_off_screen(self):
        return (self.x < -GRID_SIZE or self.x > SCREEN_WIDTH) and self.y > SCREEN_HEIGHT

class Scorpion:
    def __init__(self):
        # Start from either left or right side
        if random.random() < 0.5:
            self.x = -GRID_SIZE
            self.direction = 1
        else:
            self.x = SCREEN_WIDTH
            self.direction = -1
            
        self.y = random.randint(0, SCREEN_HEIGHT // 2)
        self.width = GRID_SIZE * 2
        self.height = GRID_SIZE
        self.speed = SCORPION_SPEED
        
    def move(self):
        self.x += self.direction * self.speed
        
    def draw(self):
        # Draw scorpion body
        pygame.draw.ellipse(screen, CYAN, (self.x, self.y, self.width, self.height))
        
        # Tail segments
        tail_x = self.x + self.width if self.direction == 1 else self.x
        for i in range(3):
            offset = i * 5 * self.direction
            pygame.draw.line(screen, CYAN, 
                            (tail_x, self.y + self.height//2), 
                            (tail_x + offset, self.y + self.height//2 - 10), 2)
        
        # Tail stinger
        stinger_x = tail_x + 3 * 5 * self.direction if self.direction == 1 else tail_x - 3 * 5 * self.direction
        pygame.draw.line(screen, CYAN, 
                        (stinger_x, self.y + self.height//2 - 10), 
                        (stinger_x + 5 * self.direction, self.y + self.height//2 - 15), 3)
        
    def is_off_screen(self):
        return self.x < -self.width or self.x > SCREEN_WIDTH

class Game:
    def __init__(self):
        self.player = Player()
        self.bullets = []
        self.mushrooms = []
        self.centipedes = [Centipede()]
        self.fleas = []
        self.spiders = []
        self.scorpions = []
        self.wave = 1
        self.game_over = False
        self.paused = False
        
        # Initialize with some mushrooms
        self.create_initial_mushrooms()
        
    def create_initial_mushrooms(self):
        # Create a random pattern of mushrooms at the start
        for _ in range(30):
            x = random.randint(0, GRID_WIDTH - 1) * GRID_SIZE
            y = random.randint(0, GRID_HEIGHT - 5) * GRID_SIZE
            self.mushrooms.append(Mushroom(x, y))
            
    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
                
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE and not self.game_over:
                    self.bullets.append(self.player.shoot())
                if event.key == pygame.K_p:  # Pause toggle
                    self.paused = not self.paused
                if event.key == pygame.K_r and self.game_over:
                    self.__init__()  # Reset game
                    
        return True
        
    def update(self):
        if self.paused or self.game_over:
            return
            
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
        self.player.move(dx, dy)
            
        # Update bullets
        for bullet in self.bullets[:]:
            bullet.move()
            if bullet.is_off_screen():
                self.bullets.remove(bullet)
                
        # Update centipedes
        for centipede in self.centipedes[:]:
            centipede.move(self.mushrooms)
            
            # Check if any segment is off screen at the bottom
            for segment in centipede.segments:
                if segment.y >= SCREEN_HEIGHT - GRID_SIZE * 3:
                    self.game_over = True
                    
        # Update fleas
        for flea in self.fleas[:]:
            flea.move()
            if flea.is_off_screen():
                self.fleas.remove(flea)
                
            # Flea drops mushrooms occasionally
            if random.random() < 0.02:  # 2% chance each frame
                grid_x = (flea.x + GRID_SIZE // 2) // GRID_SIZE * GRID_SIZE
                grid_y = (flea.y + GRID_SIZE // 2) // GRID_SIZE * GRID_SIZE
                self.mushrooms.append(Mushroom(grid_x, grid_y))
                
        # Update spiders
        for spider in self.spiders[:]:
            spider.move()
            if spider.is_off_screen():
                self.spiders.remove(spider)
                
        # Update scorpions
        for scorpion in self.scorpions[:]:
            scorpion.move()
            if scorpion.is_off_screen():
                self.scorpions.remove(scorpion)
                
        # Check bullet collisions with centipedes
        for bullet in self.bullets[:]:
            for centipede in self.centipedes[:]:
                for i, segment in enumerate(centipede.segments):
                    if (bullet.x >= segment.x and bullet.x <= segment.x + segment.width and
                        bullet.y >= segment.y and bullet.y <= segment.y + segment.height):
                        
                        # Remove the bullet
                        if bullet in self.bullets:
                            self.bullets.remove(bullet)
                            
                        # Remove the segment
                        centipede.segments.pop(i)
                        
                        # Add score (head = 100, body = 10)
                        points = 100 if segment.is_head else 10
                        self.player.score += points
                        
                        # Split centipede if it's a middle segment
                        if i > 0 and i < len(centipede.segments):
                            new_centipede = centipede.split_at(i)
                            if new_centipede:
                                self.centipedes.append(new_centipede)
                                
                        # Remove empty centipedes
                        if not centipede.segments:
                            self.centipedes.remove(centipede)
                            
                        break
                        
        # Check bullet collisions with mushrooms
        for bullet in self.bullets[:]:
            for mushroom in self.mushrooms[:]:
                if (bullet.x >= mushroom.x and bullet.x <= mushroom.x + GRID_SIZE and
                    bullet.y >= mushroom.y and bullet.y <= mushroom.y + GRID_SIZE):
                    
                    # Remove the bullet
                    if bullet in self.bullets:
                        self.bullets.remove(bullet)
                        
                    # Damage the mushroom
                    if mushroom.hit():
                        self.mushrooms.remove(mushroom)
                        self.player.score += 1
                        
        # Check bullet collisions with fleas
        for bullet in self.bullets[:]:
            for flea in self.fleas[:]:
                if (bullet.x >= flea.x and bullet.x <= flea.x + flea.width and
                    bullet.y >= flea.y and bullet.y <= flea.y + flea.height):
                    
                    # Remove the bullet
                    if bullet in self.bullets:
                        self.bullets.remove(bullet)
                        
                    # Remove the flea
                    self.fleas.remove(flea)
                    self.player.score += 200
                    
        # Check bullet collisions with spiders
        for bullet in self.bullets[:]:
            for spider in self.spiders[:]:
                if (bullet.x >= spider.x and bullet.x <= spider.x + spider.width and
                    bullet.y >= spider.y and bullet.y <= spider.y + spider.height):
                    
                    # Remove the bullet
                    if bullet in self.bullets:
                        self.bullets.remove(bullet)
                        
                    # Remove the spider
                    self.spiders.remove(spider)
                    self.player.score += 500
                    
        # Check bullet collisions with scorpions
        for bullet in self.bullets[:]:
            for scorpion in self.scorpions[:]:
                if (bullet.x >= scorpion.x and bullet.x <= scorpion.x + scorpion.width and
                    bullet.y >= scorpion.y and bullet.y <= scorpion.y + scorpion.height):
                    
                    # Remove the bullet
                    if bullet in self.bullets:
                        self.bullets.remove(bullet)
                        
                    # Remove the scorpion
                    self.scorpions.remove(scorpion)
                    self.player.score += 1000
                    
        # Check player collisions with enemies
        player_rect = pygame.Rect(self.player.x, self.player.y, self.player.width, self.player.height)
        
        # Centipede collision
        for centipede in self.centipedes:
            for segment in centipede.segments:
                segment_rect = pygame.Rect(segment.x, segment.y, segment.width, segment.height)
                if player_rect.colliderect(segment_rect):
                    self.player.lives -= 1
                    if self.player.lives <= 0:
                        self.game_over = True
                    else:
                        # Reset player position
                        self.player.x = SCREEN_WIDTH // 2 - self.player.width // 2
                        
        # Flea collision
        for flea in self.fleas:
            flea_rect = pygame.Rect(flea.x, flea.y, flea.width, flea.height)
            if player_rect.colliderect(flea_rect):
                self.player.lives -= 1
                if self.player.lives <= 0:
                    self.game_over = True
                else:
                    # Reset player position
                    self.player.x = SCREEN_WIDTH // 2 - self.player.width // 2
                    
        # Spider collision
        for spider in self.spiders:
            spider_rect = pygame.Rect(spider.x, spider.y, spider.width, spider.height)
            if player_rect.colliderect(spider_rect):
                self.player.lives -= 1
                if self.player.lives <= 0:
                    self.game_over = True
                else:
                    # Reset player position
                    self.player.x = SCREEN_WIDTH // 2 - self.player.width // 2
                    
        # Scorpion collision
        for scorpion in self.scorpions:
            scorpion_rect = pygame.Rect(scorpion.x, scorpion.y, scorpion.width, scorpion.height)
            if player_rect.colliderect(scorpion_rect):
                self.player.lives -= 1
                if self.player.lives <= 0:
                    self.game_over = True
                else:
                    # Reset player position
                    self.player.x = SCREEN_WIDTH // 2 - self.player.width // 2
                    
        # Check scorpion collisions with mushrooms (poisoning)
        for scorpion in self.scorpions:
            grid_x = (scorpion.x + scorpion.width // 2) // GRID_SIZE * GRID_SIZE
            grid_y = (scorpion.y + scorpion.height // 2) // GRID_SIZE * GRID_SIZE
            
            for mushroom in self.mushrooms:
                if mushroom.x == grid_x and mushroom.y == grid_y:
                    mushroom.poisoned = True
                    
        # Check if wave is complete (no centipedes left)
        if not self.centipedes:
            self.wave += 1
            # Add more centipedes for higher waves
            num_centipedes = min(3, self.wave // 2 + 1)
            for _ in range(num_centipedes):
                self.centipedes.append(Centipede(random.randint(8, 15)))
                
        # Spawn enemies randomly
        if random.random() < 0.005:  # Flea spawn chance
            self.fleas.append(Flea())
            
        if random.random() < 0.003:  # Spider spawn chance
            self.spiders.append(Spider())
            
        if random.random() < 0.002:  # Scorpion spawn chance
            self.scorpions.append(Scorpion())
            
        # Check for extra life
        if self.player.score >= self.player.next_life_score:
            self.player.lives += 1
            self.player.next_life_score += 12000
            
    def draw(self):
        # Fill the background
        screen.fill(BLACK)
        
        # Draw grid lines (subtle)
        for x in range(0, SCREEN_WIDTH, GRID_SIZE):
            pygame.draw.line(screen, (30, 30, 30), (x, 0), (x, SCREEN_HEIGHT))
        for y in range(0, SCREEN_HEIGHT, GRID_SIZE):
            pygame.draw.line(screen, (30, 30, 30), (0, y), (SCREEN_WIDTH, y))
            
        # Draw mushrooms
        for mushroom in self.mushrooms:
            mushroom.draw()
            
        # Draw centipedes
        for centipede in self.centipedes:
            centipede.draw()
            
        # Draw fleas
        for flea in self.fleas:
            flea.draw()
            
        # Draw spiders
        for spider in self.spiders:
            spider.draw()
            
        # Draw scorpions
        for scorpion in self.scorpions:
            scorpion.draw()
            
        # Draw bullets
        for bullet in self.bullets:
            bullet.draw()
            
        # Draw player
        self.player.draw()
        
        # Draw UI elements
        score_text = font.render(f"SCORE: {self.player.score}", True, WHITE)
        screen.blit(score_text, (10, 10))
        
        lives_text = font.render(f"LIVES: {self.player.lives}", True, WHITE)
        screen.blit(lives_text, (SCREEN_WIDTH - 150, 10))
        
        wave_text = small_font.render(f"WAVE: {self.wave}", True, WHITE)
        screen.blit(wave_text, (SCREEN_WIDTH // 2 - 40, 10))
        
        # Draw next life indicator
        next_life_text = small_font.render(f"NEXT LIFE: {max(0, self.player.next_life_score - self.player.score)}", True, WHITE)
        screen.blit(next_life_text, (SCREEN_WIDTH // 2 - 80, SCREEN_HEIGHT - 30))
        
        # Draw game over message
        if self.game_over:
            game_over_text = font.render("GAME OVER", True, RED)
            restart_text = small_font.render("Press R to Restart", True, WHITE)
            screen.blit(game_over_text, (SCREEN_WIDTH // 2 - 100, SCREEN_HEIGHT // 2 - 30))
            screen.blit(restart_text, (SCREEN_WIDTH // 2 - 80, SCREEN_HEIGHT // 2 + 10))
            
        # Draw pause message
        if self.paused:
            pause_text = font.render("PAUSED", True, YELLOW)
            screen.blit(pause_text, (SCREEN_WIDTH // 2 - 60, SCREEN_HEIGHT // 2))
            
        pygame.display.flip()

# Main game loop
def main():
    game = Game()
    running = True
    
    while running:
        running = game.handle_events()
        game.update()
        game.draw()
        clock.tick(60)  # 60 FPS
        
    pygame.quit()

if __name__ == "__main__":
    main()