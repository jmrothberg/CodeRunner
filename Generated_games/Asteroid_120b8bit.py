import pygame
import math
import random
import sys

# Initialize Pygame
pygame.init()
pygame.mixer.init()

# Screen dimensions
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Asteroids - 1979 Atari Original")

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)

# Game constants
SHIP_SIZE = 10
SHIP_THRUST = 0.2
SHIP_ROTATION = 0.05
BULLET_SPEED = 5
BULLET_MAX_DISTANCE = 500
ASTEROID_LARGE_RADIUS = 20
ASTEROID_MEDIUM_RADIUS = 10
ASTEROID_SMALL_RADIUS = 5
UFO_LARGE_RADIUS = 15
UFO_SMALL_RADIUS = 10
UFO_SPAWN_INTERVAL = 1000
MAX_BULLETS = 10
MAX_LIVES = 5

# Sound generation functions
def generate_sound(frequency, duration, volume=0.5, decay=False, modulation_freq=None):
    sample_rate = 44100
    n_samples = int(sample_rate * duration)
    buffer = bytearray(n_samples * 2)
    for i in range(n_samples):
        if decay:
            amp = volume * (1 - i / n_samples)
        else:
            amp = volume
            
        if modulation_freq is not None:
            mod_amp = 0.5 * (1 + math.sin(2 * math.pi * modulation_freq * i / sample_rate))
            amp = amp * mod_amp
            
        value = int(32767 * amp * math.sin(2 * math.pi * frequency * i / sample_rate))
        buffer[i*2] = value & 0xFF
        buffer[i*2+1] = (value >> 8) & 0xFF
    return pygame.mixer.Sound(buffer)

# Generate all sound effects
try:
    thrust_sound = generate_sound(440, 1.0, 0.5, decay=False, modulation_freq=None)
    fire_sound = generate_sound(880, 0.1, 0.5, decay=True, modulation_freq=None)
    asteroid_explosion_sound = generate_sound(200, 0.2, 0.5, decay=True, modulation_freq=None)
    large_ufo_sound = generate_sound(100, 1.0, 0.5, decay=False, modulation_freq=2)
    small_ufo_sound = generate_sound(400, 1.0, 0.5, decay=False, modulation_freq=2)
    hyperspace_sound = generate_sound(1000, 0.1, 0.5, decay=True, modulation_freq=None)
except Exception as e:
    print(f"Sound generation failed: {e}")
    thrust_sound = fire_sound = asteroid_explosion_sound = large_ufo_sound = small_ufo_sound = hyperspace_sound = None

# Vector font for digits (5x3 grid)
DIGIT_LINES = {
    '0': [
        [(0, 0), (2, 0)], [(2, 0), (2, 4)], [(2, 4), (0, 4)], [(0, 4), (0, 0)]
    ],
    '1': [
        [(1, 0), (1, 4)]
    ],
    '2': [
        [(0, 0), (2, 0)], [(2, 0), (2, 2)], [(2, 2), (0, 2)], 
        [(0, 2), (0, 4)], [(0, 4), (2, 4)]
    ],
    '3': [
        [(0, 0), (2, 0)], [(2, 0), (2, 2)], [(2, 2), (0, 2)], 
        [(0, 2), (2, 2)], [(2, 2), (2, 4)], [(2, 4), (0, 4)]
    ],
    '4': [
        [(0, 0), (0, 2)], [(0, 2), (2, 2)], [(2, 2), (2, 0)], 
        [(2, 2), (2, 4)]
    ],
    '5': [
        [(0, 0), (2, 0)], [(2, 0), (2, 2)], [(2, 2), (0, 2)], 
        [(0, 2), (0, 4)], [(0, 4), (2, 4)]
    ],
    '6': [
        [(0, 0), (2, 0)], [(2, 0), (2, 2)], [(2, 2), (0, 2)], 
        [(0, 2), (0, 4)], [(0, 4), (2, 4)], [(2, 4), (2, 2)]
    ],
    '7': [
        [(0, 0), (2, 0)], [(2, 0), (2, 4)]
    ],
    '8': [
        [(0, 0), (2, 0)], [(2, 0), (2, 2)], [(2, 2), (0, 2)], 
        [(0, 2), (0, 0)], [(0, 2), (0, 4)], [(0, 4), (2, 4)], 
        [(2, 4), (2, 2)]
    ],
    '9': [
        [(0, 0), (2, 0)], [(2, 0), (2, 2)], [(2, 2), (0, 2)], 
        [(0, 2), (0, 4)], [(0, 4), (2, 4)], [(2, 4), (2, 2)]
    ]
}

def draw_digit(screen, digit, x, y, size=10):
    if digit not in DIGIT_LINES:
        return
    for line in DIGIT_LINES[digit]:
        start = (x + line[0][0] * size, y + line[0][1] * size)
        end = (x + line[1][0] * size, y + line[1][1] * size)
        pygame.draw.line(screen, WHITE, start, end, 1)

def draw_score(screen, score, x, y):
    score_str = str(score)
    for char in score_str:
        draw_digit(screen, char, x, y, size=10)
        x += 15

# Game classes
class Ship:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.angle = 0
        self.vx = 0
        self.vy = 0
        self.thrusting = False
        self.thrust_timer = 0
        self.flame_points = [
            (0, 10), (-5, 15), (0, 20), (5, 15)
        ]
    
    def update(self):
        # Apply inertia
        self.x += self.vx
        self.y += self.vy
        
        # Screen wraparound
        if self.x < 0: self.x = SCREEN_WIDTH
        if self.x > SCREEN_WIDTH: self.x = 0
        if self.y < 0: self.y = SCREEN_HEIGHT
        if self.y > SCREEN_HEIGHT: self.y = 0
        
        # Thrust flame animation
        self.thrust_timer = (self.thrust_timer + 0.1) % 1.0
    
    def rotate(self, direction):
        self.angle += direction * SHIP_ROTATION
    
    def apply_thrust(self):
        self.vx += SHIP_THRUST * math.sin(self.angle)
        self.vy -= SHIP_THRUST * math.cos(self.angle)
        self.thrsting = True
    
    def draw(self, screen):
        # Ship triangle
        points = [
            (0, -SHIP_SIZE),
            (-SHIP_SIZE * 0.8, SHIP_SIZE * 0.8),
            (SHIP_SIZE * 0.8, SHIP_SIZE * 0.8)
        ]
        rotated = []
        for (x, y) in points:
            new_x = x * math.cos(self.angle) - y * math.sin(self.angle)
            new_y = x * math.sin(self.angle) + y * math.cos(self.angle)
            rotated.append((new_x + self.x, new_y + self.y))
        
        # Draw ship
        pygame.draw.line(screen, WHITE, rotated[0], rotated[1], 1)
        pygame.draw.line(screen, WHITE, rotated[1], rotated[2], 1)
        pygame.draw.line(screen, WHITE, rotated[2], rotated[0], 1)
        
        # Thrust flame
        if self.thrusting:
            flame_rotated = []
            for (x, y) in self.flame_points:
                new_x = x * math.cos(self.angle) - y * math.sin(self.angle)
                new_y = x * math.sin(self.angle) + y * math.cos(self.angle)
                flame_rotated.append((new_x + self.x, new_y + self.y))
            
            # Flicker effect based on thrust_timer
            flicker = 0.5 + 0.5 * math.sin(self.thrust_timer * 10)
            for i in range(len(flame_rotated)):
                start = flame_rotated[i]
                end = flame_rotated[(i+1) % len(flame_rotated)]
                pygame.draw.line(screen, (255, 255, int(255 * flicker)), start, end, 1)

class Bullet:
    def __init__(self, x, y, angle):
        self.x = x
        self.y = y
        self.angle = angle
        self.vx = BULLET_SPEED * math.sin(angle)
        self.vy = - BULLET_SPEED * math.cos(angle)
        self.start_x = x
        self.start_y = y
        self.max_distance = BULLET_MAX_DISTANCE
    
    def update(self):
        self.x += self.vx
        self.y += self.vy
        
        # Check distance from start
        dx = self.x - self.start_x
        dy = self.y - self.start_y
        distance = math.sqrt(dx*dx + dy*dy)
        if distance > self.max_distance:
            return False
        
        # Screen wraparound
        if self.x < 0: self.x = SCREEN_WIDTH
        if self.x > SCREEN_WIDTH: self.x = 0
        if self.y < 0: self.y = SCREEN_HEIGHT
        if self.y > SCREEN_HEIGHT: self.y = 0
        
        return True
    
    def draw(self, screen):
        # Draw bullet as cross
        pygame.draw.line(screen, WHITE, (self.x-1, self.y), (self.x+1, self.y), 1)
        pygame.draw.line(screen, WHITE, (self.x, self.y-1), (self.x, self.y+1), 1)

class Asteroid:
    def __init__(self, size, x, y):
        self.size = size
        self.x = x
        self.y = y
        self.vx = random.uniform(-1, 1)
        self.vy = random.uniform(-1, 1)
        self.angle = 0
        self.rotation_speed = random.uniform(-0.05, 0.05)
        self.points = self.generate_points(size)
    
    def generate_points(self, size):
        if size == 'large':
            num_points = random.randint(8, 12)
            radius = ASTEROID_LARGE_RADIUS
        elif size == 'medium':
            num_points = random.randint(6, 8)
            radius = ASTEROID_MEDIUM_RADIUS
        else:  # small
            num_points = random.randint(4, 6)
            radius = ASTEROID_SMALL_RADIUS
        
        points = []
        for i in range(num_points):
            angle = i * (2 * math.pi / num_points) + random.uniform(-0.2, 0.2)
            r = radius + random.uniform(-radius * 0.3, radius * 0.3)
            x = r * math.cos(angle)
            y = r * math.sin(angle)
            points.append((x, y))
        return points
    
    def update(self):
        self.x += self.vx
        self.y += self.vy
        self.angle += self.rotation_speed
        
        # Screen wraparound
        if self.x < 0: self.x = SCREEN_WIDTH
        if self.x > SCREEN_WIDTH: self.x = 0
        if self.y < 0: self.y = SCREEN_HEIGHT
        if self.y > SCREEN_HEIGHT: self.y = 0
    
    def draw(self, screen):
        # Rotate and translate points
        rotated = []
        for (x, y) in self.points:
            new_x = x * math.cos(self.angle) - y * math.sin(self.angle)
            new_y = x * math.sin(self.angle) + y * math.cos(self.angle)
            rotated.append((new_x + self.x, new_y + self.y))
        
        # Draw polygon outline
        n = len(rotated)
        for i in range(n):
            pygame.draw.line(screen, WHITE, rotated[i], rotated[(i+1) % n], 1)
    
    def split(self):
        if self.size == 'large':
            new_size = 'medium'
        elif self.size == 'medium':
            new_size = 'small'
        else:
            return []
        
        asteroids = []
        for _ in range(2):
            offset_x = random.uniform(-5, 5)
            offset_y = random.uniform(-5, 5)
            new_asteroid = Asteroid(new_size, self.x + offset_x, self.y + offset_y)
            new_asteroid.vx = random.uniform(-2, 2)
            new_asteroid.vy = random.uniform(-2, 2)
            asteroids.append(new_asteroid)
        return asteroids

class UFO:
    def __init__(self, ufo_type, x, y):
        self.type = ufo_type
        self.x = x
        self.y = y
        self.vx = 0
        self.vy = 0
        self.shoot_timer = 0
        self.shoot_cooldown = 100 if ufo_type == 'large' else 50
        self.points = self.generate_points(ufo_type)
    
    def generate_points(self, ufo_type):
        if ufo_type == 'large':
            return [
                (-10, -5), (10, -5), (10, 5), (5, 5), (0, 10), (-5, 5), (-10, 5)
            ]
        else:  # small
            return [
                (-5, -3), (5, -3), (5, 3), (2, 3), (0, 6), (-2, 3), (-5, 3)
            ]
    
    def update(self, player_x, player_y):
        # Move UFO
        if self.type == 'small':
            dx = player_x - self.x
            dy = player_y - self.y
            dist = math.sqrt(dx*dx + dy*dy)
            if dist > 0:
                self.vx = (dx / dist) * 3
                self.vy = (dy / dist) * 3
        else:  # large UFO moves randomly
            if self.shoot_timer % 100 == 0:
                angle = random.uniform(0, 2 * math.pi)
                self.vx = math.cos(angle) * 1
                self.vy = math.sin(angle) * 1
        
        self.x += self.vx
        self.y += self.vy
        
        # Screen wraparound
        if self.x < 0: self.x = SCREEN_WIDTH
        if self.x > SCREEN_WIDTH: self.x = 0
        if self.y < 0: self.y = SCREEN_HEIGHT
        if self.y > SCREEN_HEIGHT: self.y = 0
    
    def draw(self, screen):
        # Draw UFO
        points = []
        for (x, y) in self.points:
            points.append((x + self.x, y + self.y))
        
        n = len(points)
        for i in range(n):
            pygame.draw.line(screen, WHITE, points[i], points[(i+1) % n], 1)
    
    def shoot(self, player_x, player_y):
        if self.type == 'large':
            angle = random.uniform(0, 2 * math.pi)
        else:
            dx = player_x - self.x
            dy = player_y - self.y
            angle = math.atan2(dy, dx)
        
        bullet = Bullet(self.x, self.y, angle)
        bullet.speed = 3
        return bullet

# Game state
def reset_game():
    ship = Ship(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2)
    asteroids = [Asteroid('large', random.randint(50, SCREEN_WIDTH-50), random.randint(50, SCREEN_HEIGHT-50)) for _ in range(4)]
    bullets = []
    ufos = []
    score = 0
    lives = 3
    level = 1
    ufo_timer = 0
    thrust_channel = None
    ufo_channels = []
    
    return ship, asteroids, bullets, ufos, score, lives, level, ufo_timer, thrust_channel, ufo_channels

# Main game loop
ship, asteroids, bullets, ufos, score, lives, level, ufo_timer, thrust_channel, ufo_channels = reset_game()
clock = pygame.time.Clock()
running = True

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
    
    keys = pygame.key.get_pressed()
    
    # Handle ship controls
    if keys[pygame.K_LEFT]:
        ship.rotate(-1)
    if keys[pygame.K_RIGHT]:
        ship.rotate(1)
    if keys[pygame.K_UP]:
        if not ship.thrusting:
            ship.apply_thrust()
            if thrust_channel is None or not thrust_channel.get_busy():
                if thrust_sound:
                    thrust_channel = thrust_sound.play(-1)
    else:
        ship.thrusting = False
        if thrust_channel and thrust_channel.get_busy():
            thrust_channel.stop()
            thrust_channel = None
    
    if keys[pygame.K_SPACE] and len(bullets) < MAX_BULLETS:
        bullet = Bullet(ship.x, ship.y, ship.angle)
        bullets.append(bullet)
        if fire_sound:
            fire_sound.play()
    
    if keys[pygame.K_h]:
        ship.x = random.randint(0, SCREEN_WIDTH)
        ship.y = random.randint(0, SCREEN_HEIGHT)
        if hyperspace_sound:
            hyperspace_sound.play()
    
    # Update game objects
    ship.update()
    
    # Update bullets
    for bullet in bullets[:]:
        if not bullet.update():
            bullets.remove(bullet)
    
    # Update asteroids
    for asteroid in asteroids[:]:
        asteroid.update()
    
    # Update UFOs
    for ufo in ufos[:]:
        ufo.update(ship.x, ship.y)
        ufo.shoot_timer += 1
        if ufo.shoot_timer >= ufo.shoot_cooldown:
            ufo.shoot_timer = 0
            bullet = ufo.shoot(ship.x, ship.y)
            bullets.append(bullet)
    
    # Spawn UFOs
    ufo_timer += 1
    if ufo_timer >= UFO_SPAWN_INTERVAL:
        ufo_timer = 0
        side = random.randint(0, 3)
        if side == 0:  # top
            x = random.randint(0, SCREEN_WIDTH)
            y = 0
        elif side == 1:  # right
            x = SCREEN_WIDTH
            y = random.randint(0, SCREEN_HEIGHT)
        elif side == 2:  # bottom
            x = random.randint(0, SCREEN_WIDTH)
            y = SCREEN_HEIGHT
        else:  # left
            x = 0
            y = random.randint(0, SCREEN_HEIGHT)
        
        ufo_type = 'large' if random.random() < 0.7 else 'small'
        ufo = UFO(ufo_type, x, y)
        ufos.append(ufo)
        if ufo_type == 'large' and large_ufo_sound:
            channel = large_ufo_sound.play(-1)
            ufo_channels.append(channel)
        elif ufo_type == 'small' and small_ufo_sound:
            channel = small_ufo_sound.play(-1)
            ufo_channels.append(channel)
    
    # Check collisions
    # Ship with asteroids
    for asteroid in asteroids[:]:
        dx = ship.x - asteroid.x
        dy = ship.y - asteroid.y
        dist = math.sqrt(dx*dx + dy*dy)
        ship_radius = SHIP_SIZE
        asteroid_radius = ASTEROID_LARGE_RADIUS if asteroid.size == 'large' else ASTEROID_MEDIUM_RADIUS if asteroid.size == 'medium' else ASTEROID_SMALL_RADIUS
        if dist < ship_radius + asteroid_radius:
            lives -= 1
            ship = Ship(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2)
            asteroids.remove(asteroid)
            if lives <= 0:
                ship, asteroids, bullets, ufos, score, lives, level, ufo_timer, thrust_channel, ufo_channels = reset_game()
            break
    
    # Bullets with asteroids
    for bullet in bullets[:]:
        for asteroid in asteroids[:]:
            dx = bullet.x - asteroid.x
            dy = bullet.y - asteroid.y
            dist = math.sqrt(dx*dx + dy*dy)
            asteroid_radius = ASTEROID_LARGE_RADIUS if asteroid.size == 'large' else ASTEROID_MEDIUM_RADIUS if asteroid.size == 'medium' else ASTEROID_SMALL_RADIUS
            if dist < 1 + asteroid_radius:
                bullets.remove(bullet)
                asteroids.remove(asteroid)
                if asteroid.size == 'large':
                    score += 20
                elif asteroid.size == 'medium':
                    score += 50
                else:
                    score += 100
                new_asteroids = asteroid.split()
                asteroids.extend(new_asteroids)
                if asteroid_explosion_sound:
                    asteroid_explosion_sound.play()
                break
    
    # Bullets with UFOs
    for bullet in bullets[:]:
        for ufo in ufos[:]:
            dx = bullet.x - ufo.x
            dy = bullet.y - ufo.y
            dist = math.sqrt(dx*dx + dy*dy)
            ufo_radius = UFO_LARGE_RADIUS if ufo.type == 'large' else UFO_SMALL_RADIUS
            if dist < ufo_radius:
                bullets.remove(bullet)
                ufos.remove(ufo)
                for channel in ufo_channels:
                    if channel.get_sound() == large_ufo_sound or channel.get_sound() == small_ufo_sound:
                        channel.stop()
                ufo_channels = [c for c in ufo_channels if c.get_busy()]
                if ufo.type == 'large':
                    score += 200
                else:
                    score += 1000
                if asteroid_explosion_sound:
                    asteroid_explosion_sound.play()
                break
    
    # Ship with UFOs
    for ufo in ufos[:]:
        dx = ship.x - ufo.x
        dy = ship.y - ufo.y
        dist = math.sqrt(dx*dx + dy*dy)
        ship_radius = SHIP_SIZE
        ufo_radius = UFO_LARGE_RADIUS if ufo.type == 'large' else UFO_SMALL_RADIUS
        if dist < ship_radius + ufo_radius:
            lives -= 1
            ship = Ship(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2)
            ufos.remove(ufo)
            for channel in ufo_channels:
                if channel.get_sound() == large_ufo_sound or channel.get_sound() == small_ufo_sound:
                    channel.stop()
            ufo_channels = [c for c in ufo_channels if c.get_busy()]
            if lives <= 0:
                ship, asteroids, bullets, ufos, score, lives, level, ufo_timer, thrust_channel, ufo_channels = reset_game()
            break
    
    # Check for new level
    if len(asteroids) == 0:
        level += 1
        num_asteroids = 4 + level
        asteroids = [Asteroid('large', random.randint(50, SCREEN_WIDTH-50), random.randint(50, SCREEN_HEIGHT-50)) for _ in range(num_asteroids)]
        ufo_timer = 0
    
    # Check for extra life
    if score >= 10000 and lives < MAX_LIVES:
        lives += 1
        score -= 10000
    
    # Draw everything
    screen.fill(BLACK)
    
    # Draw ship
    ship.draw(screen)
    
    # Draw bullets
    for bullet in bullets:
        bullet.draw(screen)
    
    # Draw asteroids
    for asteroid in asteroids:
        asteroid.draw(screen)
    
    # Draw UFOs
    for ufo in ufos:
        ufo.draw(screen)
    
    # Draw score
    draw_score(screen, score, 10, 10)
    
    # Draw lives
    for i in range(lives):
        ship_points = [
            (0, -3), (-2, 2), (2, 2)
        ]
        rotated = []
        for (x, y) in ship_points:
            new_x = x * math.cos(0) - y * math.sin(0)
            new_y = x * math.sin(0) + y * math.cos(0)
            rotated.append((new_x + SCREEN_WIDTH - 20 - 20*i, new_y + 10))
        pygame.draw.line(screen, WHITE, rotated[0], rotated[1], 1)
        pygame.draw.line(screen, WHITE, rotated[1], rotated[2], 1)
        pygame.draw.line(screen, WHITE, rotated[2], rotated[0], 1)
    
    # Draw level
    draw_digit(screen, str(level), SCREEN_WIDTH - 30, 10, size=10)
    
    pygame.display.flip()
    clock.tick(60)

pygame.quit()
sys.exit()