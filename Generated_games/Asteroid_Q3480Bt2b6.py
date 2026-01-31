import pygame
import math
import random
import sys

# Initialize pygame
pygame.init()
pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=512)

# Screen dimensions
WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Asteroids - Faithful to the 1979 Atari Original")

# Colors (pure black background with bright white vector lines)
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
ORANGE = (255, 165, 0)

# Game constants
FPS = 60
SHIP_SIZE = 30
ASTEROID_SIZES = [40, 80, 120]  # Small, medium, large
UFO_SIZE = 30
BULLET_SPEED = 10
SHIP_ACCELERATION = 0.5
ROTATION_SPEED = 6
FRICTION = 0.98
WRAP_BUFFER = 20

# Font setup
font = pygame.font.SysFont('Arial', 24)
big_font = pygame.font.SysFont('Arial', 36)

# Sound generation functions
def generate_tone(frequency, duration, volume=0.5):
    sample_rate = 22050
    frames = int(duration * sample_rate)
    arr = pygame.sndarray.array(pygame.mixer.Sound(buffer=bytes(frames*4)))
    
    for i in range(frames):
        t = float(i) / sample_rate
        wave = volume * math.sin(2 * math.pi * frequency * t)
        arr[i][0] = int(wave * 32767)
        arr[i][1] = int(wave * 32767)
        
    return pygame.sndarray.make_sound(arr)

# Generate game sounds
thrust_sound = generate_tone(250, 0.1, 0.3)
shoot_sound = generate_tone(800, 0.1, 0.4)
explosion_sound = generate_tone(220, 0.3, 0.5)
ufo_sound = generate_tone(400, 0.2, 0.3)

class Ship:
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.x = WIDTH / 2
        self.y = HEIGHT / 2
        self.angle = 0
        self.velocity_x = 0
        self.velocity_y = 0
        self.thrusting = False
        self.flame_frame = 0
        self.invincible = 0  # Invincibility frames after respawn
        
    def rotate(self, direction):
        self.angle += ROTATION_SPEED * direction
        
    def thrust(self):
        self.thrusting = True
        radians = math.radians(self.angle)
        self.velocity_x += SHIP_ACCELERATION * math.sin(radians)
        self.velocity_y -= SHIP_ACCELERATION * math.cos(radians)
        
    def update(self):
        # Apply friction to slow down ship gradually
        self.velocity_x *= FRICTION
        self.velocity_y *= FRICTION
        
        # Update position based on velocity
        self.x += self.velocity_x
        self.y += self.velocity_y
        
        # Screen wraparound
        if self.x < -WRAP_BUFFER:
            self.x = WIDTH + WRAP_BUFFER
        elif self.x > WIDTH + WRAP_BUFFER:
            self.x = -WRAP_BUFFER
            
        if self.y < -WRAP_BUFFER:
            self.y = HEIGHT + WRAP_BUFFER
        elif self.y > HEIGHT + WRAP_BUFFER:
            self.y = -WRAP_BUFFER
            
        # Update flame animation when thrusting
        if self.thrusting:
            self.flame_frame = (self.flame_frame + 1) % 4
        else:
            self.flame_frame = 0
            
        # Decrease invincibility frames
        if self.invincible > 0:
            self.invincible -= 1
            
        
    def draw(self, surface):
        # Don't draw ship if it's currently invincible (flashing effect)
        if self.invincible > 0 and self.invincible % 4 < 2:
            return
            
        radians = math.radians(self.angle)
        
        # Calculate ship points
        nose_x = self.x + SHIP_SIZE * math.sin(radians)
        nose_y = self.y - SHIP_SIZE * math.cos(radians)
        
        wing1_x = self.x + SHIP_SIZE * math.sin(radians + 2.5)
        wing1_y = self.y - SHIP_SIZE * math.cos(radians + 2.5)
        
        wing2_x = self.x + SHIP_SIZE * math.sin(radians - 2.5)
        wing2_y = self.y - SHIP_SIZE * math.cos(radians - 2.5)
        
        # Draw ship as wireframe triangle
        pygame.draw.line(surface, WHITE, (nose_x, nose_y), (wing1_x, wing1_y), 2)
        pygame.draw.line(surface, WHITE, (wing1_x, wing1_y), (wing2_x, wing2_y), 2)
        pygame.draw.line(surface, WHITE, (wing2_x, wing2_y), (nose_x, nose_y), 2)
        
        # Draw thrust flame if thrusting
        if self.thrusting:
            flame_length = 15 + random.randint(0, 10) * (self.flame_frame % 2)
             # Position flame at the back of the ship, in the opposite direction it's facing
            flame_x = self.x - (SHIP_SIZE + flame_length/3) * math.sin(radians)
            flame_y = self.y + (SHIP_SIZE + flame_length/3) * math.cos(radians)
    
            # Flame points (wider at the base, pointed at the tip)
            flame_points = [
                (flame_x, flame_y),  # Tip of flame
                (flame_x + flame_length * math.sin(radians + 0.5), 
                 flame_y - flame_length * math.cos(radians + 0.5)),  # Base point 1
                (flame_x + flame_length * math.sin(radians - 0.5), 
                 flame_y - flame_length * math.cos(radians - 0.5))   # Base point 2
            ]
    
            # Draw orange flame
            pygame.draw.polygon(surface, ORANGE, flame_points, 0)
            self.thrusting = False
            
    def get_collision_radius(self):
        return SHIP_SIZE
        
class Asteroid:
    def __init__(self, x=None, y=None, size=2):  # Size: 0=small, 1=medium, 2=large
        self.size = size
        self.radius = ASTEROID_SIZES[size] / 2
        
        if x is None or y is None:
            # Spawn at edge of screen
            side = random.randint(0, 3)
            if side == 0:  # Top
                self.x = random.randint(0, WIDTH)
                self.y = -self.radius
            elif side == 1:  # Right
                self.x = WIDTH + self.radius
                self.y = random.randint(0, HEIGHT)
            elif side == 2:  # Bottom
                self.x = random.randint(0, WIDTH)
                self.y = HEIGHT + self.radius
            else:  # Left
                self.x = -self.radius
                self.y = random.randint(0, HEIGHT)
        else:
            self.x = x
            self.y = y
            
        # Random velocity and rotation
        self.velocity_x = random.uniform(-2, 2) * (3 - size) / 2
        self.velocity_y = random.uniform(-2, 2) * (3 - size) / 2
        self.rotation = random.uniform(-3, 3)
        self.angle = random.randint(0, 360)
        
        # Create irregular shape with jagged edges
        self.points = []
        num_points = random.randint(7, 12)
        for i in range(num_points):
            distance = self.radius * random.uniform(0.8, 1.2)
            angle = math.radians(i * (360 / num_points))
            x_point = distance * math.cos(angle)
            y_point = distance * math.sin(angle)
            self.points.append((x_point, y_point))
            
    def update(self):
        # Update position
        self.x += self.velocity_x
        self.y += self.velocity_y
        
        # Update rotation
        self.angle += self.rotation
        
        # Screen wraparound
        if self.x < -self.radius:
            self.x = WIDTH + self.radius
        elif self.x > WIDTH + self.radius:
            self.x = -self.radius
            
        if self.y < -self.radius:
            self.y = HEIGHT + self.radius
        elif self.y > HEIGHT + self.radius:
            self.y = -self.radius
            
    def draw(self, surface):
        # Rotate and translate points for drawing
        rotated_points = []
        radians = math.radians(self.angle)
        
        for point in self.points:
            x_rotated = point[0] * math.cos(radians) - point[1] * math.sin(radians)
            y_rotated = point[0] * math.sin(radians) + point[1] * math.cos(radians)
            rotated_points.append((self.x + x_rotated, self.y + y_rotated))
            
        # Draw wireframe asteroid
        if len(rotated_points) > 2:
            pygame.draw.polygon(surface, WHITE, rotated_points, 1)
            
    def split(self):
        # Return new asteroids when hit (if not smallest size)
        if self.size > 0:
            return [
                Asteroid(self.x, self.y, self.size - 1),
                Asteroid(self.x, self.y, self.size - 1)
            ]
        return []
        
    def get_collision_radius(self):
        return self.radius

class Bullet:
    def __init__(self, x, y, angle):
        self.x = x
        self.y = y
        self.angle = angle
        radians = math.radians(angle)
        self.velocity_x = BULLET_SPEED * math.sin(radians)
        self.velocity_y = -BULLET_SPEED * math.cos(radians)
        self.lifetime = 60  # Frames until bullet disappears
        
    def update(self):
        self.x += self.velocity_x
        self.y += self.velocity_y
        self.lifetime -= 1
        
        # Screen wraparound
        if self.x < 0:
            self.x = WIDTH
        elif self.x > WIDTH:
            self.x = 0
            
        if self.y < 0:
            self.y = HEIGHT
        elif self.y > HEIGHT:
            self.y = 0
            
    def draw(self, surface):
        pygame.draw.circle(surface, WHITE, (int(self.x), int(self.y)), 2)
        
    def is_alive(self):
        return self.lifetime > 0

class UFO:
    def __init__(self, large=True):
        self.large = large
        self.radius = UFO_SIZE if large else UFO_SIZE/1.5
        
        # Spawn at edge of screen
        side = random.randint(0, 3)
        if side == 0:  # Top
            self.x = random.randint(0, WIDTH)
            self.y = -self.radius
        elif side == 1:  # Right
            self.x = WIDTH + self.radius
            self.y = random.randint(0, HEIGHT)
        elif side == 2:  # Bottom
            self.x = random.randint(0, WIDTH)
            self.y = HEIGHT + self.radius
        else:  # Left
            self.x = -self.radius
            self.y = random.randint(0, HEIGHT)
            
        # Set velocity based on size (large UFO is slower)
        speed = random.uniform(1.0, 2.0) if large else random.uniform(2.5, 4.0)
        angle = random.uniform(0, math.pi * 2)
        self.velocity_x = math.cos(angle) * speed
        self.velocity_y = math.sin(angle) * speed
        
        # UFO shooting behavior
        self.shoot_timer = random.randint(60, 180) if large else random.randint(30, 90)
        
    def update(self):
        # Update position
        self.x += self.velocity_x
        self.y += self.velocity_y
        
        # Screen wraparound
        if self.x < -self.radius:
            self.x = WIDTH + self.radius
        elif self.x > WIDTH + self.radius:
            self.x = -self.radius
            
        if self.y < -self.radius:
            self.y = HEIGHT + self.radius
        elif self.y > HEIGHT + self.radius:
            self.y = -self.radius
            
        # Update shooting timer
        self.shoot_timer -= 1
        
    def draw(self, surface):
        # Draw classic flying saucer shape
        center_x, center_y = int(self.x), int(self.y)
        
        # Main body (ellipse)
        pygame.draw.ellipse(surface, WHITE, 
                           (center_x - self.radius/1.5, center_y - self.radius/3,
                            self.radius*2/1.5, self.radius*2/3), 1)
        
        # Cockpit
        pygame.draw.arc(surface, WHITE, 
                       (center_x - self.radius/3, center_y - self.radius/4,
                        self.radius*2/3, self.radius/2),
                       math.pi, 0, 1)
        
        # Base details
        pygame.draw.line(surface, WHITE, 
                        (center_x - self.radius/1.5, center_y), 
                        (center_x + self.radius/1.5, center_y), 1)
                        
    def shoot(self, target_x=None, target_y=None):
        if self.shoot_timer <= 0:
            # Reset timer
            self.shoot_timer = random.randint(60, 240) if self.large else random.randint(30, 90)
            
            # For large UFO: random shot direction
            if self.large:
                angle = random.uniform(0, 360)
                return Bullet(self.x, self.y, angle)
                
            # For small UFO: aim at player (with some inaccuracy)
            else:
                if target_x is not None and target_y is not None:
                    dx = target_x - self.x
                    dy = target_y - self.y
                    angle = math.degrees(math.atan2(dy, dx)) + 90
                    
                    # Add some inaccuracy to make it challenging but beatable
                    inaccuracy = random.uniform(-15, 15)
                    return Bullet(self.x, self.y, angle + inaccuracy)
                    
        return None
        
    def get_collision_radius(self):
        return self.radius

class Particle:
    def __init__(self, x, y, color=WHITE):
        self.x = x
        self.y = y
        self.color = color
        self.velocity_x = random.uniform(-3, 3)
        self.velocity_y = random.uniform(-3, 3)
        self.lifetime = random.randint(20, 40)
        
    def update(self):
        self.x += self.velocity_x
        self.y += self.velocity_y
        self.lifetime -= 1
        
    def draw(self, surface):
        if self.lifetime > 0:
            size = max(1, self.lifetime // 10)
            pygame.draw.circle(surface, self.color, (int(self.x), int(self.y)), size)

class ShipDebris:
    def __init__(self, x, y, velocity_x, velocity_y):
        self.x = x
        self.y = y
        self.velocity_x = velocity_x + random.uniform(-1, 1)
        self.velocity_y = velocity_y + random.uniform(-1, 1)
        self.angle = random.uniform(0, 360)
        self.rotation_speed = random.uniform(-5, 5)
        self.lifetime = random.randint(60, 120)
        
        # Create geometric debris pieces
        self.pieces = []
        for _ in range(random.randint(3, 6)):
            size = random.randint(5, 15)
            points = []
            num_points = random.randint(3, 5)
            for i in range(num_points):
                angle = math.radians(i * (360 / num_points))
                px = size * math.cos(angle)
                py = size * math.sin(angle)
                points.append((px, py))
            self.pieces.append({
                'points': points,
                'offset_x': random.uniform(-20, 20),
                'offset_y': random.uniform(-20, 20)
            })
            
    def update(self):
        self.x += self.velocity_x
        self.y += self.velocity_y
        self.angle += self.rotation_speed
        self.lifetime -= 1
        
        # Screen wraparound
        if self.x < -50:
            self.x = WIDTH + 50
        elif self.x > WIDTH + 50:
            self.x = -50
            
        if self.y < -50:
            self.y = HEIGHT + 50
        elif self.y > HEIGHT + 50:
            self.y = -50
            
    def draw(self, surface):
        radians = math.radians(self.angle)
        
        for piece in self.pieces:
            rotated_points = []
            for point in piece['points']:
                # Rotate point
                x_rotated = point[0] * math.cos(radians) - point[1] * math.sin(radians)
                y_rotated = point[0] * math.sin(radians) + point[1] * math.cos(radians)
                
                # Translate to position
                rotated_points.append((
                    self.x + piece['offset_x'] + x_rotated,
                    self.y + piece['offset_y'] + y_rotated
                ))
                
            if len(rotated_points) > 2:
                pygame.draw.polygon(surface, WHITE, rotated_points, 1)
                
    def is_alive(self):
        return self.lifetime > 0

def distance(x1, y1, x2, y2):
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def main():
    clock = pygame.time.Clock()
    
    # Game objects
    ship = Ship()
    asteroids = []
    bullets = []
    ufos = []
    particles = []
    ship_debris = []
    
    # Background stars with random properties for motion effect
    stars = []
    for _ in range(100):
        x = random.randint(0, WIDTH)
        y = random.randint(0, HEIGHT)
        size = random.randint(1, 3)
        brightness = random.randint(150, 255)
        speed = random.uniform(0.1, 0.5)
        angle = random.uniform(0, math.pi * 2)
        stars.append({
            'x': x,
            'y': y,
            'size': size,
            'brightness': brightness,
            'speed': speed,
            'angle': angle
        })
    
    # Game state
    score = 0
    lives = 3
    level = 1
    game_over = False
    paused = False
    
    # Create initial asteroids
    for _ in range(4):
        asteroids.append(Asteroid(size=2))
        
    # UFO spawn timer
    ufo_timer = random.randint(600, 900)
    
    # Extra life awarded flag
    extra_life_awarded = False
    
    running = True
    while running:
        # Event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                    
                if event.key == pygame.K_p:
                    paused = not paused
                    
                if game_over and event.key == pygame.K_r:
                    # Reset game
                    ship.reset()
                    asteroids = []
                    bullets = []
                    ufos = []
                    particles = []
                    ship_debris = []
                    score = 0
                    lives = 3
                    level = 1
                    game_over = False
                    extra_life_awarded = False
                    
                    # Create initial asteroids
                    for _ in range(4):
                        asteroids.append(Asteroid(size=2))
                        
        if paused or game_over:
            # Draw pause/game over screen
            screen.fill(BLACK)
            
            if game_over:
                game_over_text = big_font.render("GAME OVER", True, WHITE)
                restart_text = font.render("Press R to Restart", True, WHITE)
                screen.blit(game_over_text, (WIDTH//2 - game_over_text.get_width()//2, HEIGHT//2 - 50))
                screen.blit(restart_text, (WIDTH//2 - restart_text.get_width()//2, HEIGHT//2 + 10))
            else:
                pause_text = big_font.render("PAUSED", True, WHITE)
                continue_text = font.render("Press P to Continue", True, WHITE)
                screen.blit(pause_text, (WIDTH//2 - pause_text.get_width()//2, HEIGHT//2 - 30))
                screen.blit(continue_text, (WIDTH//2 - continue_text.get_width()//2, HEIGHT//2 + 20))
                
            pygame.display.flip()
            clock.tick(FPS)
            continue
            
        # Player input
        keys = pygame.key.get_pressed()
        if not game_over:
            if keys[pygame.K_LEFT]:
                ship.rotate(-1)
            if keys[pygame.K_RIGHT]:
                ship.rotate(1)
            if keys[pygame.K_UP]:
                ship.thrust()
                thrust_sound.play()
            
            # Shooting (with cooldown to prevent spamming)
            if keys[pygame.K_SPACE] and len(bullets) < 4:
                # Only allow shooting every 10 frames
                if not hasattr(main, 'shoot_cooldown') or main.shoot_cooldown <= 0:
                    radians = math.radians(ship.angle)
                    bullet_x = ship.x + (SHIP_SIZE + 5) * math.sin(radians)
                    bullet_y = ship.y - (SHIP_SIZE + 5) * math.cos(radians)
                    bullets.append(Bullet(bullet_x, bullet_y, ship.angle))
                    shoot_sound.play()
                    main.shoot_cooldown = 10
            else:
                if hasattr(main, 'shoot_cooldown'):
                    main.shoot_cooldown -= 1
                    
        # Update game objects
        ship.update()
        
        for asteroid in asteroids[:]:
            asteroid.update()
            
        for bullet in bullets[:]:
            bullet.update()
            if not bullet.is_alive():
                bullets.remove(bullet)
                
        for ufo in ufos[:]:
            ufo.update()
            
            # UFO shooting logic
            new_bullet = ufo.shoot(ship.x, ship.y)
            if new_bullet:
                bullets.append(new_bullet)
                ufo_sound.play()
                
            # Remove UFOs that have left the screen significantly
            if (ufo.x < -100 or ufo.x > WIDTH + 100 or 
                ufo.y < -100 or ufo.y > HEIGHT + 100):
                ufos.remove(ufo)
                
        for particle in particles[:]:
            particle.update()
            if particle.lifetime <= 0:
                particles.remove(particle)
                
        for debris in ship_debris[:]:
            debris.update()
            if not debris.is_alive():
                ship_debris.remove(debris)
                
        # Update background stars for motion effect
        for star in stars:
            star['x'] += math.cos(star['angle']) * star['speed']
            star['y'] += math.sin(star['angle']) * star['speed']
            
            # Wrap around screen edges
            if star['x'] < 0:
                star['x'] = WIDTH
            elif star['x'] > WIDTH:
                star['x'] = 0
                
            if star['y'] < 0:
                star['y'] = HEIGHT
            elif star['y'] > HEIGHT:
                star['y'] = 0
                
            # Randomly change brightness for twinkling effect
            if random.random() < 0.02:
                star['brightness'] = random.randint(150, 255)
                
        # UFO spawning
        ufo_timer -= 1
        if ufo_timer <= 0:
            # Spawn a random type of UFO (large more common)
            if len(ufos) < 2:  # Limit number of UFOs on screen
                if random.random() < 0.7:
                    ufos.append(UFO(large=True))
                else:
                    ufos.append(UFO(large=False))
            ufo_timer = random.randint(600, 1200)
            
        # Collision detection: bullets with asteroids
        for bullet in bullets[:]:
            for asteroid in asteroids[:]:
                if distance(bullet.x, bullet.y, asteroid.x, asteroid.y) < asteroid.get_collision_radius():
                    # Create explosion particles
                    for _ in range(20):
                        particles.append(Particle(asteroid.x, asteroid.y))
                        
                    # Remove bullet and asteroid
                    bullets.remove(bullet)
                    explosion_sound.play()
                    
                    # Add score based on asteroid size (small=100, medium=50, large=20)
                    score += 20 * (3 - asteroid.size)
                    
                    # Split asteroid if not smallest size
                    new_asteroids = asteroid.split()
                    asteroids.extend(new_asteroids)
                    asteroids.remove(asteroid)
                    break
                    
        # Collision detection: bullets with UFOs
        for bullet in bullets[:]:
            for ufo in ufos[:]:
                if distance(bullet.x, bullet.y, ufo.x, ufo.y) < ufo.get_collision_radius():
                    # Create explosion particles
                    for _ in range(15):
                        particles.append(Particle(ufo.x, ufo.y))
                        
                    # Remove bullet and UFO
                    bullets.remove(bullet)
                    explosion_sound.play()
                    
                    # Add score based on UFO size (large=200, small=1000)
                    score += 200 if ufo.large else 1000
                    
                    ufos.remove(ufo)
                    break
                    
        # Collision detection: ship with asteroids
        if ship.invincible <= 0:
            for asteroid in asteroids[:]:
                if distance(ship.x, ship.y, asteroid.x, asteroid.y) < (ship.get_collision_radius() + asteroid.get_collision_radius()):
                    lives -= 1
                    ship.invincible = 90  # 1.5 seconds of invincibility
                    
                    # Create explosion particles
                    for _ in range(30):
                        particles.append(Particle(ship.x, ship.y, ORANGE))
                        
                    # Create ship debris when destroyed
                    if lives > 0:
                        for _ in range(5):
                            ship_debris.append(ShipDebris(
                                ship.x, ship.y, 
                                ship.velocity_x, ship.velocity_y
                            ))
                    
                    explosion_sound.play()
                    
                    if lives <= 0:
                        game_over = True
                        # Create more debris on final destruction
                        for _ in range(10):
                            ship_debris.append(ShipDebris(
                                ship.x, ship.y, 
                                ship.velocity_x, ship.velocity_y
                            ))
                        
        # Collision detection: ship with UFOs or their bullets
        if ship.invincible <= 0:
            for ufo in ufos[:]:
                if distance(ship.x, ship.y, ufo.x, ufo.y) < (ship.get_collision_radius() + ufo.get_collision_radius()):
                    lives -= 1
                    ship.invincible = 90
                    
                    # Create explosion particles
                    for _ in range(30):
                        particles.append(Particle(ship.x, ship.y, ORANGE))
                        
                    # Create ship debris when destroyed
                    if lives > 0:
                        for _ in range(5):
                            ship_debris.append(ShipDebris(
                                ship.x, ship.y, 
                                ship.velocity_x, ship.velocity_y
                            ))
                    
                    explosion_sound.play()
                    
                    if lives <= 0:
                        game_over = True
                        # Create more debris on final destruction
                        for _ in range(10):
                            ship_debris.append(ShipDebris(
                                ship.x, ship.y, 
                                ship.velocity_x, ship.velocity_y
                            ))
                        
        # Check if level is complete (no asteroids left)
        if len(asteroids) == 0 and not game_over:
            level += 1
            # Create new asteroids for next level
            for _ in range(min(4 + level, 8)):  # Cap at 8 asteroids
                asteroids.append(Asteroid(size=2))
                
        # Award extra life at 10,000 points
        if score >= 10000 and not extra_life_awarded:
            lives += 1
            extra_life_awarded = True
            
        # Drawing
        screen.fill(BLACK)
        
        # Draw animated background stars
        for star in stars:
            brightness = star['brightness']
            color = (brightness, brightness, brightness)
            pygame.draw.circle(screen, color, (int(star['x']), int(star['y'])), star['size'])
            
        # Draw game objects
        ship.draw(screen)
        
        for asteroid in asteroids:
            asteroid.draw(screen)
            
        for bullet in bullets:
            bullet.draw(screen)
            
        for ufo in ufos:
            ufo.draw(screen)
            
        for particle in particles:
            particle.draw(screen)
            
        for debris in ship_debris:
            debris.draw(screen)
            
        # Draw UI
        score_text = font.render(f"SCORE: {score}", True, WHITE)
        lives_text = font.render(f"LIVES: {lives}", True, WHITE)
        level_text = font.render(f"LEVEL: {level}", True, WHITE)
        
        screen.blit(score_text, (10, 10))
        screen.blit(lives_text, (10, 40))
        screen.blit(level_text, (10, 70))
        
        # Draw controls help
        controls = [
            "CONTROLS:",
            "Arrow Keys - Move",
            "Space - Shoot",
            "P - Pause",
            "ESC - Quit"
        ]
        
        for i, text in enumerate(controls):
            ctrl_text = font.render(text, True, WHITE)
            screen.blit(ctrl_text, (WIDTH - ctrl_text.get_width() - 10, 10 + i*30))
            
        pygame.display.flip()
        clock.tick(FPS)
        
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()