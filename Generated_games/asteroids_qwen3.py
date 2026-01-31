import pygame
import math
import random
import sys

# Initialize pygame
pygame.init()
pygame.mixer.init()

# Screen dimensions
WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Asteroids")

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)

# Game variables
FPS = 60
score = 0
lives = 3
level = 1
extra_life_granted = False

# Sound effects (using pygame's sound generation)
def create_thrust_sound():
    sample_rate = 22050
    duration = 0.1
    n_samples = int(round(duration * sample_rate))
    buf = pygame.sndarray.array([[0, 0]] * n_samples)
    for i in range(n_samples):
        value = random.randint(-1000, 1000)
        buf[i][0] = value
        buf[i][1] = value
    sound = pygame.mixer.Sound(buf)
    return sound

def create_fire_sound():
    sample_rate = 22050
    duration = 0.2
    n_samples = int(round(duration * sample_rate))
    buf = pygame.sndarray.array([[0, 0]] * n_samples)
    for i in range(n_samples):
        value = random.randint(-5000, 5000)
        buf[i][0] = value
        buf[i][1] = value
    sound = pygame.mixer.Sound(buf)
    return sound

def create_explosion_sound():
    sample_rate = 22050
    duration = 0.3
    n_samples = int(round(duration * sample_rate))
    buf = pygame.sndarray.array([[0, 0]] * n_samples)
    for i in range(n_samples):
        value = random.randint(-10000, 10000) * (n_samples - i) / n_samples
        buf[i][0] = value
        buf[i][1] = value
    sound = pygame.mixer.Sound(buf)
    return sound

def create_ufo_sound():
    sample_rate = 22050
    duration = 0.5
    n_samples = int(round(duration * sample_rate))
    buf = pygame.sndarray.array([[0, 0]] * n_samples)
    for i in range(n_samples):
        value = int(10000 * math.sin(2 * math.pi * 220 * i / sample_rate) * 
                     (math.sin(2 * math.pi * 2 * i / sample_rate) + 1) / 2)
        buf[i][0] = value
        buf[i][1] = value
    sound = pygame.mixer.Sound(buf)
    return sound

# Create sounds
thrust_sound = create_thrust_sound()
fire_sound = create_fire_sound()
explosion_sound = create_explosion_sound()
ufo_sound = create_ufo_sound()

# Starfield background
class Star:
    def __init__(self):
        self.x = random.randint(0, WIDTH)
        self.y = random.randint(0, HEIGHT)
        self.size = random.uniform(0.5, 2)
        self.brightness = random.randint(100, 255)

    def draw(self, surface):
        color = (self.brightness, self.brightness, self.brightness)
        pygame.draw.circle(surface, color, (int(self.x), int(self.y)), int(self.size))
 
stars = [Star() for _ in range(100)]

# Ship class
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
        
    def rotate(self, direction):
        self.angle += direction * 5
        
    def thrust(self):
        self.thrusting = True
        angle_rad = math.radians(self.angle)
        self.velocity_x += math.sin(angle_rad) * 0.2
        self.velocity_y -= math.cos(angle_rad) * 0.2
        
        # Limit maximum velocity
        speed = math.sqrt(self.velocity_x**2 + self.velocity_y**2)
        if speed > 8:
            self.velocity_x = self.velocity_x * 8 / speed
            self.velocity_y = self.velocity_y * 8 / speed
            
    def update(self):
        # Apply velocity
        self.x += self.velocity_x
        self.y += self.velocity_y
        
        # Screen wraparound
        if self.x < 0:
            self.x = WIDTH
        elif self.x > WIDTH:
            self.x = 0
        if self.y < 0:
            self.y = HEIGHT
        elif self.y > HEIGHT:
            self.y = 0
            
        # Update flame animation
        if self.thrusting:
            self.flame_frame = (self.flame_frame + 1) % 4
            
    def draw(self, surface):
        # Ship vertices
        angle_rad = math.radians(self.angle)
        size = 20
        
        # Calculate ship points
        nose_x = self.x + math.sin(angle_rad) * size
        nose_y = self.y - math.cos(angle_rad) * size
        left_x = self.x + math.sin(angle_rad + 5*math.pi/6) * size
        left_y = self.y - math.cos(angle_rad + 5*math.pi/6) * size
        right_x = self.x + math.sin(angle_rad - 5*math.pi/6) * size
        right_y = self.y - math.cos(angle_rad - 5*math.pi/6) * size
        
        # Draw ship
        pygame.draw.polygon(surface, WHITE, [
            (nose_x, nose_y),
            (left_x, left_y),
            (right_x, right_y)
        ], 1)
        
        # Draw thrust flame if thrusting
        if self.thrusting and self.flame_frame < 2:
            flame_length = random.randint(10, 20)
            flame_x = self.x - math.sin(angle_rad) * flame_length
            flame_y = self.y + math.cos(angle_rad) * flame_length
            
            pygame.draw.line(surface, WHITE, 
                               (self.x, self.y), 
                               (flame_x, flame_y), 2)
                               
    def get_vertices(self):
        angle_rad = math.radians(self.angle)
        size = 20
        
        # Calculate ship points
        nose_x = self.x + math.sin(angle_rad) * size
        nose_y = self.y - math.cos(angle_rad) * size
        left_x = self.x + math.sin(angle_rad + 5*math.pi/6) * size
        left_y = self.y - math.cos(angle_rad + 5*math.pi/6) * size
        right_x = self.x + math.sin(angle_rad - 5*math.pi/6) * size
        right_y = self.y - math.cos(angle_rad - 5*math.pi/6) * size
        
        return [(nose_x, nose_y), (left_x, left_y), (right_x, right_y)]

# Bullet class
class Bullet:
    def __init__(self, x, y, angle):
        self.x = x
        self.y = y
        self.angle = angle
        self.velocity_x = math.sin(math.radians(angle)) * 10
        self.velocity_y = -math.cos(math.radians(angle)) * 10
        self.distance_traveled = 0
        self.max_distance = 500
        
    def update(self):
        self.x += self.velocity_x
        self.y += self.velocity_y
        self.distance_traveled += math.sqrt(self.velocity_x**2 + self.velocity_y**2)
        
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
        
    def is_expired(self):
        return self.distance_traveled > self.max_distance

# Asteroid class
class Asteroid:
    def __init__(self, x, y, size):
        self.x = x
        self.y = y
        self.size = size  # 3=large, 2=medium, 1=small
        self.speed = random.uniform(1.0, 3.0) / size
        self.angle = random.uniform(0, 360)
        self.rotation_speed = random.uniform(-2, 2)
        self.rotation_angle = 0
        
        # Create irregular shape
        self.points = []
        num_points = random.randint(7, 12)
        for i in range(num_points):
            angle = 2 * math.pi * i / num_points
            distance = self.size * (15 + random.randint(-5, 5))
            x_point = distance * math.cos(angle)
            y_point = distance * math.sin(angle)
            self.points.append((x_point, y_point))
            
    def update(self):
        # Move asteroid
        self.x += math.sin(math.radians(self.angle)) * self.speed
        self.y -= math.cos(math.radians(self.angle)) * self.speed
        
        # Rotate asteroid
        self.rotation_angle += self.rotation_speed
        
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
        # Draw rotated asteroid
        rotated_points = []
        angle_rad = math.radians(self.rotation_angle)
        for point in self.points:
            x_rot = point[0] * math.cos(angle_rad) - point[1] * math.sin(angle_rad)
            y_rot = point[0] * math.sin(angle_rad) + point[1] * math.cos(angle_rad)
            rotated_points.append((self.x + x_rot, self.y + y_rot))
            
        pygame.draw.polygon(surface, WHITE, rotated_points, 1)
        
    def get_vertices(self):
        # Return current vertices for collision detection
        rotated_points = []
        angle_rad = math.radians(self.rotation_angle)
        for point in self.points:
            x_rot = point[0] * math.cos(angle_rad) - point[1] * math.sin(angle_rad)
            y_rot = point[0] * math.sin(angle_rad) + point[1] * math.cos(angle_rad)
            rotated_points.append((self.x + x_rot, self.y + y_rot))
        return rotated_points

# UFO class
class UFO:
    def __init__(self, size):
        self.size = size  # 1=small, 2=large
        self.width = 30 * size
        self.height = 15 * size
        
        # Randomly choose side to spawn on
        if random.choice([True, False]):
            self.x = -self.width
            self.direction = 1
        else:
            self.x = WIDTH + self.width
            self.direction = -1
            
        self.y = random.randint(50, HEIGHT - 50)
        self.speed = random.uniform(1.0, 3.0) * self.direction
        self.shoot_timer = 0
        self.shoot_delay = random.randint(60, 180)  # Frames between shots
        
    def update(self, ship_x, ship_y):
        self.x += self.speed
        
        # Move vertically in a sine wave pattern
        self.y += math.sin(pygame.time.get_ticks() / 500) * 0.5
        
        # UFO shooting logic
        self.shoot_timer += 1
        if self.shoot_timer >= self.shoot_delay:
            self.shoot_timer = 0
            self.shoot_delay = random.randint(60, 180)
            
            # Create bullet toward ship (small UFO) or randomly (large UFO)
            if self.size == 1:  # Small UFO - precise shooting
                dx = ship_x - self.x
                dy = ship_y - self.y
                angle = math.degrees(math.atan2(dy, dx))
            else:  # Large UFO - random shooting
                angle = random.uniform(0, 360)
                
            return Bullet(self.x, self.y, angle)
        return None
        
    def draw(self, surface):
        # Draw UFO body (ellipse)
        pygame.draw.ellipse(surface, WHITE, 
                               (self.x - self.width/2, self.y - self.height/2, 
                                self.width, self.height), 1)
        # Draw cockpit (smaller ellipse)
        pygame.draw.ellipse(surface, WHITE, 
                               (self.x - self.width/4, self.y - self.height/4, 
                                self.width/2, self.height/2), 1)
    def get_vertices(self):
        # Return bounding box vertices for collision detection
        half_width = self.width / 2
        half_height = self.height / 2
        return [
            (self.x - half_width, self.y - half_height),
            (self.x + half_width, self.y - half_height),
            (self.x + half_width, self.y + half_height),
            (self.x - half_width, self.y + half_height)
        ]

# Collision detection functions
def point_in_polygon(point, polygon):
    x, y = point
    n = len(polygon)
    inside = False

    p1x, p1y = polygon[0]
    for i in range(1, n + 1):
        p2x, p2y = polygon[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y

    return inside

def check_collision(obj1_vertices, obj2_vertices):
    # Check if any vertex of obj1 is inside obj2
    for point in obj1_vertices:
        if point_in_polygon(point, obj2_vertices):
            return True
            
    # Check if any vertex of obj2 is inside obj1
    for point in obj2_vertices:
        if point_in_polygon(point, obj1_vertices):
            return True
            
    return False

# Game objects
ship = Ship()
bullets = []
asteroids = []
ufos = []

# Create initial asteroids
def create_asteroids(count):
    new_asteroids = []
    for _ in range(count):
        # Spawn asteroids away from the ship
        while True:
            x = random.randint(0, WIDTH)
            y = random.randint(0, HEIGHT)
            distance = math.sqrt((x - ship.x)**2 + (y - ship.y)**2)
            if distance > 150:  # Minimum safe distance
                break
                
        new_asteroids.append(Asteroid(x, y, 3))  # Large asteroids
    return new_asteroids

# Initialize game objects
asteroids = create_asteroids(4)

# UFO spawn timer
ufo_spawn_timer = 0
ufo_spawn_delay = random.randint(600, 1800)  # Frames between UFO appearances

# Main game loop
clock = pygame.time.Clock()
running = True

while running:
    # Event handling
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                # Fire bullet (limit to 4 on screen)
                if len(bullets) < 4:
                    bullets.append(Bullet(ship.x, ship.y, ship.angle))
                    fire_sound.play()
            elif event.key == pygame.K_h:  # Hyperspace jump
                ship.x = random.randint(50, WIDTH - 50)
                ship.y = random.randint(50, HEIGHT - 50)
                ship.velocity_x = 0
                ship.velocity_y = 0

    # Continuous key presses
    keys = pygame.key.get_pressed()
    if keys[pygame.K_LEFT]:
        ship.rotate(-1)
    if keys[pygame.K_RIGHT]:
        ship.rotate(1)
    if keys[pygame.K_UP]:
        ship.thrust()
        thrust_sound.play()
    else:
        ship.thrusting = False

    # Update game objects
    ship.update()
    
    for bullet in bullets[:]:
        bullet.update()
        if bullet.is_expired():
            bullets.remove(bullet)
            
    for asteroid in asteroids:
        asteroid.update()
        
    for ufo in ufos[:]:
        new_bullet = ufo.update(ship.x, ship.y)
        if new_bullet:
            bullets.append(new_bullet)
        # Remove UFOs that have left the screen
        if (ufo.x < -50 and ufo.direction == -1) or \
           (ufo.x > WIDTH + 50 and ufo.direction == 1):
            ufos.remove(ufo)

    # UFO spawning
    ufo_spawn_timer += 1
    if ufo_spawn_timer >= ufo_spawn_delay:
        ufo_spawn_timer = 0
        ufo_spawn_delay = random.randint(600, 1800)
        # Randomly choose small or large UFO (small is rarer)
        ufo_size = 1 if random.random() < 0.3 else 2
        ufos.append(UFO(ufo_size))
        ufo_sound.play()

    # Collision detection - bullets with asteroids
    for bullet in bullets[:]:
        for asteroid in asteroids[:]:
            asteroid_vertices = asteroid.get_vertices()
            if point_in_polygon((bullet.x, bullet.y), asteroid_vertices):
                # Remove bullet and asteroid
                if bullet in bullets:
                    bullets.remove(bullet)
                if asteroid in asteroids:
                    asteroids.remove(asteroid)
                
                # Play explosion sound
                explosion_sound.play()
                
                # Add to score based on asteroid size
                if asteroid.size == 3:  # Large
                    score += 20
                elif asteroid.size == 2:  # Medium
                    score += 50
                elif asteroid.size == 1:  # Small
                    score += 100
                    
                # Split asteroid into smaller ones
                if asteroid.size > 1:
                    for _ in range(2):
                        new_size = asteroid.size - 1
                        # Add some randomness to the split direction
                        angle = random.uniform(0, 360)
                        asteroids.append(Asteroid(asteroid.x, asteroid.y, new_size))
                        
                break

    # Collision detection - bullets with UFOs
    for bullet in bullets[:]:
        for ufo in ufos[:]:
            ufo_vertices = ufo.get_vertices()
            if point_in_polygon((bullet.x, bullet.y), ufo_vertices):
                # Remove bullet and UFO
                if bullet in bullets:
                    bullets.remove(bullet)
                if ufo in ufos:
                    ufos.remove(ufo)
                
                # Play explosion sound
                explosion_sound.play()
                
                # Add to score based on UFO size
                if ufo.size == 2:  # Large
                    score += 200
                elif ufo.size == 1:  # Small
                    score += 1000
                break

    # Collision detection - ship with asteroids
    ship_vertices = ship.get_vertices()
    for asteroid in asteroids:
        asteroid_vertices = asteroid.get_vertices()
        if check_collision(ship_vertices, asteroid_vertices):
            # Ship destroyed
            lives -= 1
            explosion_sound.play()
            
            if lives <= 0:
                # Game over
                running = False
            else:
                # Reset ship position
                ship.reset()
                
            break

    # Collision detection - ship with UFOs or their bullets
    for ufo in ufos:
        ufo_vertices = ufo.get_vertices()
        if check_collision(ship_vertices, ufo_vertices):
            # Ship destroyed
            lives -= 1
            explosion_sound.play()
            
            if lives <= 0:
                # Game over
                running = False
            else:
                # Reset ship position
                ship.reset()
                
            break

    # Check for extra life
    if score >= 10000 and not extra_life_granted:
        lives += 1
        extra_life_granted = True

    # Check if all asteroids are destroyed (next level)
    if len(asteroids) == 0 and len(ufos) == 0:
        level += 1
        asteroids = create_asteroids(4 + level)  # More asteroids each level

    # Drawing
    screen.fill(BLACK)
    
    # Draw stars
    for star in stars:
        star.draw(screen)
    
    # Draw game objects
    ship.draw(screen)
    
    for bullet in bullets:
        bullet.draw(screen)
        
    for asteroid in asteroids:
        asteroid.draw(screen)
        
    for ufo in ufos:
        ufo.draw(screen)
    
    # Draw UI
    font = pygame.font.SysFont(None, 36)
    score_text = font.render(f"SCORE: {score}", True, WHITE)
    lives_text = font.render(f"LIVES: {lives}", True, WHITE)
    level_text = font.render(f"LEVEL: {level}", True, WHITE)
    
    screen.blit(score_text, (10, 10))
    screen.blit(lives_text, (10, 50))
    screen.blit(level_text, (10, 90))
    
    # Update display
    pygame.display.flip()
    
    # Cap the frame rate
    clock.tick(FPS)

# Quit pygame
pygame.quit()
sys.exit()