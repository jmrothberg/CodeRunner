import pygame
import random
import sys

# Initialize PyGame
pygame.init()

# Constants
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
FPS = 60

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)

# Game settings
PLAYER_SPEED = 5
BULLET_SPEED = 7
ALIEN_SPEED = 1
ALIEN_DROP_DISTANCE = 20
BUNKER_SIZE = 4
BUNKER_ROWS = 4
BUNKER_COLS = 8
UFO_APPEAR_CHANCE = 0.003
UFO_SPEED = 3
UFO_POINTS = 100

# Font
FONT_PATH = None
FONT_SIZE = 24
try:
    FONT = pygame.font.Font(FONT_PATH, FONT_SIZE) if FONT_PATH else pygame.font.SysFont('Courier New', FONT_SIZE, bold=True)
except:
    FONT = pygame.font.SysFont('Courier New', FONT_SIZE, bold=True)

# Sprite sizes
PLAYER_WIDTH = 32
PLAYER_HEIGHT = 16
ALIEN_WIDTH = 24
ALIEN_HEIGHT = 16
BULLET_WIDTH = 2
BULLET_HEIGHT = 10
UFO_WIDTH = 40
UFO_HEIGHT = 16

class Player(pygame.sprite.Sprite):
    def __init__(self):
        super().__init__()
        self.image = pygame.Surface((PLAYER_WIDTH, PLAYER_HEIGHT))
        self.image.fill(GREEN)
        pygame.draw.rect(self.image, GREEN, (0, 0, PLAYER_WIDTH, PLAYER_HEIGHT))
        pygame.draw.rect(self.image, GREEN, (PLAYER_WIDTH//2 - 4, PLAYER_HEIGHT-4, 8, 4))
        self.rect = self.image.get_rect()
        self.rect.centerx = SCREEN_WIDTH // 2
        self.rect.bottom = SCREEN_HEIGHT - 10
        self.speed = PLAYER_SPEED
        self.bullet_cooldown = 0

    def update(self):
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT] and self.rect.left > 0:
            self.rect.x -= self.speed
        if keys[pygame.K_RIGHT] and self.rect.right < SCREEN_WIDTH:
            self.rect.x += self.speed
        
        # Fire upward bullet
        if keys[pygame.K_SPACE] and self.bullet_cooldown <= 0:
            bullet = Bullet(self.rect.centerx, self.rect.top - BULLET_HEIGHT, True)  # Start just above player
            all_sprites.add(bullet)
            bullets.add(bullet)
            self.bullet_cooldown = 15
        if self.bullet_cooldown > 0:
            self.bullet_cooldown -= 1

class Alien(pygame.sprite.Sprite):
    def __init__(self, row, col, alien_type):
        super().__init__()
        self.row = row
        self.col = col
        self.alien_type = alien_type
        self.animation_frame = 0
        self.animation_timer = 0
        
        if alien_type == 0:  # Squid (top)
            self.color = BLUE
            self.points = 30
        elif alien_type == 1:  # Bug (middle)
            self.color = RED
            self.points = 20
        else:  # Octopus (bottom)
            self.color = WHITE
            self.points = 10
            
        self.image = pygame.Surface((ALIEN_WIDTH, ALIEN_HEIGHT))
        self.image.fill(BLACK)
        self.image.set_colorkey(BLACK)
        self.draw_alien()
        self.rect = self.image.get_rect()

    def draw_alien(self):
        self.image.fill(BLACK)  # Clear
        
        if self.alien_type == 0:  # Squid
            pygame.draw.rect(self.image, self.color, (4, 2, 16, 8))
            pygame.draw.rect(self.image, self.color, (2, 10, 3, 4))
            pygame.draw.rect(self.image, self.color, (7, 10, 3, 4))
            pygame.draw.rect(self.image, self.color, (14, 10, 3, 4))
            pygame.draw.rect(self.image, WHITE, (6, 4, 2, 2))
            pygame.draw.rect(self.image, WHITE, (16, 4, 2, 2))
            
        elif self.alien_type == 1:  # Bug
            pygame.draw.rect(self.image, self.color, (4, 2, 16, 8))
            pygame.draw.rect(self.image, self.color, (2, 10, 2, 4))
            pygame.draw.rect(self.image, self.color, (6, 10, 2, 4))
            pygame.draw.rect(self.image, self.color, (12, 10, 2, 4))
            pygame.draw.rect(self.image, self.color, (16, 10, 2, 4))
            pygame.draw.rect(self.image, self.color, (8, 0, 2, 2))
            pygame.draw.rect(self.image, self.color, (14, 0, 2, 2))
            
        else:  # Octopus
            pygame.draw.rect(self.image, self.color, (6, 2, 12, 8))
            for i in range(5):
                x = 4 + i * 4
                if i != 2:
                    pygame.draw.rect(self.image, self.color, (x, 10, 2, 6))
            pygame.draw.rect(self.image, WHITE, (8, 4, 2, 2))
            pygame.draw.rect(self.image, WHITE, (14, 4, 2, 2))

    def update(self):
        self.animation_timer += 1
        if self.animation_timer >= 10:  # Change sprite every 10 frames
            self.animation_frame = 1 - self.animation_frame
            self.animation_timer = 0
            old_rect = self.rect.copy()  # Preserve position!
            self.draw_alien()
            self.rect = old_rect  # Restore position after redrawing image
            # IMPORTANT: We must reassign rect to preserve position after changing image!

class Bullet(pygame.sprite.Sprite):
    def __init__(self, x, y, is_player_bullet=True):
        super().__init__()
        self.image = pygame.Surface((BULLET_WIDTH, BULLET_HEIGHT))
        self.image.fill(WHITE)
        self.rect = self.image.get_rect()
        self.rect.centerx = x
        if is_player_bullet:
            # Player bullet: shoots UP → negative speed
            self.rect.bottom = y  # Start at player's top
            self.speed = -BULLET_SPEED  # ✅ FIXED: NEGATIVE = UP
        else:
            # Alien bomb: falls DOWN → positive speed
            self.rect.top = y + ALIEN_HEIGHT  # Start just below alien
            self.speed = BULLET_SPEED // 2   # ✅ FIXED: POSITIVE = DOWN (slower than player bullet)
        
        self.is_player_bullet = is_player_bullet

    def update(self):
        self.rect.y += self.speed
        if self.rect.bottom < 0 or self.rect.top > SCREEN_HEIGHT:
            self.kill()

class UFO(pygame.sprite.Sprite):
    def __init__(self, direction=1):
        super().__init__()
        self.direction = direction
        self.image = pygame.Surface((UFO_WIDTH, UFO_HEIGHT))
        self.image.fill(BLACK)
        self.image.set_colorkey(BLACK)
        
        pygame.draw.rect(self.image, RED, (8, 2, 24, 8))
        pygame.draw.rect(self.image, BLUE, (4, 10, 32, 4))
        pygame.draw.rect(self.image, WHITE, (16, 0, 8, 2))
        
        self.rect = self.image.get_rect()
        if direction == 1:
            self.rect.left = -UFO_WIDTH
        else:
            self.rect.right = SCREEN_WIDTH + UFO_WIDTH
        self.rect.top = 50
        self.speed = UFO_SPEED * direction

    def update(self):
        self.rect.x += self.speed
        if self.rect.left > SCREEN_WIDTH or self.rect.right < 0:
            self.kill()

class Bunker(pygame.sprite.Sprite):
    def __init__(self, x, y):
        super().__init__()
        self.image = pygame.Surface((BUNKER_COLS * BUNKER_SIZE, BUNKER_ROWS * BUNKER_SIZE))
        self.image.fill(GREEN)
        self.rect = self.image.get_rect()
        self.rect.x = x
        self.rect.y = y
        self.health_grid = [[1 for _ in range(BUNKER_COLS)] for _ in range(BUNKER_ROWS)]
        
    def take_damage(self, x, y):
        local_x = x - self.rect.x
        local_y = y - self.rect.y
        
        if 0 <= local_x < BUNKER_COLS * BUNKER_SIZE and 0 <= local_y < BUNKER_ROWS * BUNKER_SIZE:
            grid_x = local_x // BUNKER_SIZE
            grid_y = local_y // BUNKER_SIZE
            
            if 0 <= grid_x < BUNKER_COLS and 0 <= grid_y < BUNKER_ROWS:
                self.health_grid[grid_y][grid_x] = 0
                self.redraw_bunker()
                
    def redraw_bunker(self):
        self.image.fill(BLACK)  # Clear to black first
        for y in range(BUNKER_ROWS):
            for x in range(BUNKER_COLS):
                if self.health_grid[y][x] == 1:
                    pygame.draw.rect(self.image, GREEN, 
                                   (x * BUNKER_SIZE, y * BUNKER_SIZE, BUNKER_SIZE, BUNKER_SIZE))
        if sum(sum(row) for row in self.health_grid) == 0:
            self.kill()

def create_aliens():
    aliens = pygame.sprite.Group()
    for row in range(5):
        for col in range(11):
            if row < 2:
                alien_type = 0
            elif row < 4:
                alien_type = 1
            else:
                alien_type = 2
            alien = Alien(row, col, alien_type)
            alien.rect.x = 80 + col * (ALIEN_WIDTH + 10)
            alien.rect.y = 60 + row * (ALIEN_HEIGHT + 15)  # Start high enough to allow bunker space
            aliens.add(alien)
    return aliens

def main():
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Space Invaders - 1978 Taito Original")
    clock = pygame.time.Clock()
    
    global all_sprites, bullets
    all_sprites = pygame.sprite.Group()
    aliens = create_aliens()
    all_sprites.add(aliens)
    
    player = Player()
    all_sprites.add(player)
    
    bullets = pygame.sprite.Group()
    bunkers = pygame.sprite.Group()
    ufo = None
    
    # ✅ FIXED: Bunkers now placed 60px above player (original spacing)
    bunker_spacing = (SCREEN_WIDTH - 4 * (BUNKER_COLS * BUNKER_SIZE)) // 5
    for i in range(4):
        x = bunker_spacing + i * (bunker_spacing + BUNKER_COLS * BUNKER_SIZE)
        y = SCREEN_HEIGHT - 80  # ✅ FIXED: Now 80px from bottom → perfect spacing!
        bunker = Bunker(x, y)
        bunkers.add(bunker)
        all_sprites.add(bunker)
    
    alien_direction = 1
    alien_speed = ALIEN_SPEED
    score = 0
    lives = 3
    game_over = False
    win = False
    
    ufo_timer = 0
    ufo_spawned = False
    
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
                
        if game_over or win:
            screen.fill(BLACK)
            font_large = pygame.font.SysFont('Courier New', 48, bold=True)
            text = font_large.render("YOU WIN!" if win else "GAME OVER", True, GREEN if win else RED)
            rect = text.get_rect(center=(SCREEN_WIDTH//2, SCREEN_HEIGHT//2))
            screen.blit(text, rect)
            
            score_text = FONT.render(f"Final Score: {score}", True, WHITE)
            score_rect = score_text.get_rect(center=(SCREEN_WIDTH//2, SCREEN_HEIGHT//2 + 50))
            screen.blit(score_text, score_rect)
            
            restart_text = FONT.render("Press R to Restart or Q to Quit", True, WHITE)
            restart_rect = restart_text.get_rect(center=(SCREEN_WIDTH//2, SCREEN_HEIGHT//2 + 100))
            screen.blit(restart_text, restart_rect)
            
            keys = pygame.key.get_pressed()
            if keys[pygame.K_r]:
                main()
            elif keys[pygame.K_q]:
                pygame.quit()
                sys.exit()
                
            pygame.display.flip()
            clock.tick(FPS)
            continue
            
        # UFO spawn
        ufo_timer += 1
        if not ufo_spawned and len(aliens) > 0 and random.random() < UFO_APPEAR_CHANCE and ufo_timer > 300:
            ufo = UFO(random.choice([1, -1]))
            all_sprites.add(ufo)
            ufo_spawned = True
            
        # Alien movement
        move_down = False
        for alien in aliens:
            if (alien.rect.right >= SCREEN_WIDTH - 20 and alien_direction > 0) or \
               (alien.rect.left <= 20 and alien_direction < 0):
                move_down = True
                break
                
        if move_down:
            alien_direction *= -1
            for alien in aliens:
                alien.rect.y += ALIEN_DROP_DISTANCE
        else:
            for alien in aliens:
                alien.rect.x += alien_speed * alien_direction
                
        # Accelerate as aliens die
        if len(aliens) < 40 and alien_speed < 3:
            alien_speed = 1 + (55 - len(aliens)) / 20
            
        # Alien shooting: bombs fall DOWN
        if len(aliens) > 0 and random.random() < 0.005 * (55 / max(len(aliens), 1)):
            shooter = random.choice(list(aliens))
            bomb = Bullet(shooter.rect.centerx, shooter.rect.bottom, False)  # ✅ Start at alien bottom
            all_sprites.add(bomb)
            bullets.add(bomb)
            
        # Update everything
        player.update()
        bullets.update()
        aliens.update()  # ✅ CRITICAL: We forgot to call .update() on aliens!
        if ufo:
            ufo.update()
            
        # Collisions
        hits = pygame.sprite.groupcollide(bullets, aliens, True, True)
        for hit in hits:
            score += hits[hit][0].points
            
        # Player bullet hits bunkers
        player_bullet_hits = pygame.sprite.groupcollide(bullets, bunkers, False, False)
        for bullet, bunker_list in player_bullet_hits.items():
            if bullet.is_player_bullet:
                for bunker in bunker_list:
                    bunker.take_damage(bullet.rect.centerx, bullet.rect.centery)
                    
        # Alien bomb hits bunkers
        alien_bullet_hits = pygame.sprite.groupcollide(bullets, bunkers, True, False)
        for bullet, bunker_list in alien_bullet_hits.items():
            if not bullet.is_player_bullet:
                for bunker in bunker_list:
                    bunker.take_damage(bullet.rect.centerx, bullet.rect.centery)
                    
        # ✅ Only alien bombs kill player
        if pygame.sprite.spritecollide(player, bullets, False):
            for bullet in bullets:
                if not bullet.is_player_bullet:
                    lives -= 1
                    bullet.kill()
                    break
            if lives <= 0:
                game_over = True
            else:
                player.rect.centerx = SCREEN_WIDTH // 2
                
        # UFO hit by player bullet
        if ufo and pygame.sprite.spritecollide(ufo, bullets, True):
            score += UFO_POINTS
            ufo.kill()
            ufo_spawned = False
            
        # Aliens reached bottom?
        for alien in aliens:
            if alien.rect.bottom >= SCREEN_HEIGHT - 80:
                game_over = True
                break
                
        # Win condition
        if len(aliens) == 0:
            win = True
            
        # Draw everything
        screen.fill(BLACK)
        
        score_text = FONT.render(f"SCORE: {score}", True, WHITE)
        lives_text = FONT.render(f"LIVES: {lives}", True, WHITE)
        screen.blit(score_text, (10, 10))
        screen.blit(lives_text, (SCREEN_WIDTH - 120, 10))
        
        all_sprites.draw(screen)
        
        pygame.display.flip()
        clock.tick(FPS)

if __name__ == "__main__":
    main()