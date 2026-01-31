import pygame
import math
import random
import sys

# Constants
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
FPS = 60
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)

class Ship:
    def __init__(self):
        self.pos = [SCREEN_WIDTH / 2, SCREEN_HEIGHT / 2]
        self.vel = [0.0, 0.0]
        self.angle = 0
        self.thrusting = False
        self.max_speed = 5.0
        self.rotation_speed = 3.0
        self.acceleration = 0.1
        self.radius = 15

    def update(self):
        if self.thrusting:
            rad_angle = math.radians(self.angle)
            accel_x = self.acceleration * math.sin(rad_angle)
            accel_y = -self.acceleration * math.cos(rad_angle)
            self.vel[0] += accel_x
            self.vel[1] += accel_y

            speed = math.sqrt(self.vel[0]**2 + self.vel[1]**2)
            if speed > self.max_speed:
                scale = self.max_speed / speed
                self.vel[0] *= scale
                self.vel[1] *= scale

        self.pos[0] += self.vel[0]
        self.pos[1] += self.vel[1]

        # Screen wraparound
        if self.pos[0] < 0:
            self.pos[0] = SCREEN_WIDTH
        elif self.pos[0] > SCREEN_WIDTH:
            self.pos[0] = 0
        if self.pos[1] < 0:
            self.pos[1] = SCREEN_HEIGHT
        elif self.pos[1] > SCREEN_HEIGHT:
            self.pos[1] = 0

    def draw(self, screen):
        # Ship triangle points (relative to center)
        points = [
            (0, -20),   # Tip
            (-15, 15),
            (15, 15)
        ]
        rotated_points = []
        for x, y in points:
            rad_angle = math.radians(self.angle)
            new_x = x * math.cos(rad_angle) - y * math.sin(rad_angle)
            new_y = x * math.sin(rad_angle) + y * math.cos(rad_angle)
            rotated_points.append((new_x + self.pos[0], new_y + self.pos[1]))
        
        # Draw ship outline
        pygame.draw.lines(screen, WHITE, True, rotated_points, 2)

        if self.thrusting:
            # Thrust flame points (relative to center)
            flame_points = [
                (0, 25),   # Base at rear
                (-10, 35),
                (10, 35)
            ]
            rotated_flame = []
            for x, y in flame_points:
                rad_angle = math.radians(self.angle)
                new_x = x * math.cos(rad_angle) - y * math.sin(rad_angle)
                new_y = x * math.sin(rad_angle) + y * math.cos(rad_angle)
                rotated_flame.append((new_x + self.pos[0], new_y + self.pos[1]))
            pygame.draw.lines(screen, WHITE, True, rotated_flame, 2)

class Asteroid:
    def __init__(self, size):
        self.size = size
        # Spawn off-screen to avoid immediate collisions
        side = random.choice(['top', 'bottom', 'left', 'right'])
        if side == 'top':
            self.pos = [random.randint(0, SCREEN_WIDTH), -20]
        elif side == 'bottom':
            self.pos = [random.randint(0, SCREEN_WIDTH), SCREEN_HEIGHT + 20]
        elif side == 'left':
            self.pos = [-20, random.randint(0, SCREEN_HEIGHT)]
        else:  # right
            self.pos = [SCREEN_WIDTH + 20, random.randint(0, SCREEN_HEIGHT)]
        
        # Velocity based on size (larger asteroids move slower)
        speed = 1.0 if size == 3 else (2.0 if size == 2 else 3.0)
        angle = random.uniform(0, 2 * math.pi)
        self.vel = [speed * math.cos(angle), speed * math.sin(angle)]
        
        # Rotation properties
        self.rotation_speed = random.uniform(-1, 1)
        self.angle = random.uniform(0, 360)
        
        # Generate irregular polygon points for asteroid shape
        num_points = 8 + size * 2
        radius = 30 if size == 3 else (20 if size == 2 else 15)
        self.points = []
        for i in range(num_points):
            angle = 2 * math.pi * i / num_points
            r = radius + random.randint(-radius // 4, radius // 4)
            x = r * math.cos(angle)
            y = r * math.sin(angle)
            self.points.append((x, y))
        
        # Calculate collision radius
        max_radius = 0
        for x, y in self.points:
            r = math.sqrt(x**2 + y**2)
            if r > max_radius:
                max_radius = r
        self.radius = max_radius

    def update(self):
        self.pos[0] += self.vel[0]
        self.pos[1] += self.vel[1]
        
        # Screen wraparound
        if self.pos[0] < -self.radius:
            self.pos[0] = SCREEN_WIDTH + self.radius
        elif self.pos[0] > SCREEN_WIDTH + self.radius:
            self.pos[0] = -self.radius
        if self.pos[1] < -self.radius:
            self.pos[1] = SCREEN_HEIGHT + self.radius
        elif self.pos[1] > SCREEN_HEIGHT + self.radius:
            self.pos[1] = -self.radius
        
        # Rotate asteroid
        self.angle += self.rotation_speed

    def split(self):
        if self.size == 1:
            return []
        new_size = self.size - 1
        new_asteroids = []
        for _ in range(2):
            angle_offset = random.uniform(-math.pi / 4, math.pi / 4)
            current_angle = math.atan2(self.vel[1], self.vel[0])
            new_angle = current_angle + angle_offset
            speed = 1.5 * (self.size / 3)  # Increase speed when splitting
            new_vel_x = speed * math.cos(new_angle)
            new_vel_y = speed * math.sin(new_angle)
            new_asteroid = Asteroid(new_size)
            new_asteroid.pos = self.pos.copy()
            new_asteroid.vel = [new_vel_x, new_vel_y]
            new_asteroids.append(new_asteroid)
        return new_asteroids

    def draw(self, screen):
        rotated_points = []
        for x, y in self.points:
            rad_angle = math.radians(self.angle)
            new_x = x * math.cos(rad_angle) - y * math.sin(rad_angle)
            new_y = x * math.sin(rad_angle) + y * math.cos(rad_angle)
            rotated_points.append((new_x + self.pos[0], new_y + self.pos[1]))
        pygame.draw.lines(screen, WHITE, True, rotated_points, 2)

class Bullet:
    def __init__(self, pos, angle):
        self.pos = list(pos)
        self.angle = angle
        self.speed = 10.0
        self.lifetime = 40  # Frames until disappearance
        rad_angle = math.radians(angle)
        self.vel = [self.speed * math.sin(rad_angle), -self.speed * math.cos(rad_angle)]
        self.radius = 2

    def update(self):
        self.pos[0] += self.vel[0]
        self.pos[1] += self.vel[1]
        self.lifetime -= 1
        if self.lifetime <= 0:
            return False
        
        # Screen wraparound
        if self.pos[0] < -self.radius:
            self.pos[0] = SCREEN_WIDTH + self.radius
        elif self.pos[0] > SCREEN_WIDTH + self.radius:
            self.pos[0] = -self.radius
        if self.pos[1] < -self.radius:
            self.pos[1] = SCREEN_HEIGHT + self.radius
        elif self.pos[1] > SCREEN_HEIGHT + self.radius:
            self.pos[1] = -self.radius
        return True

    def draw(self, screen):
        pygame.draw.circle(screen, WHITE, (int(self.pos[0]), int(self.pos[1])), self.radius)

class UFO:
    def __init__(self, ufo_type):
        self.type = ufo_type
        if ufo_type == 'large':
            self.size = 30
            self.speed = 1.5
            self.fire_rate = 60
            self.shot_speed = 5.0
        else:  # small UFO
            self.size = 20
            self.speed = 3.0
            self.fire_rate = 30
            self.shot_speed = 8.0
        
        # Spawn on random edge
        side = random.choice(['top', 'bottom', 'left', 'right'])
        if side == 'top':
            self.pos = [random.randint(0, SCREEN_WIDTH), -self.size]
            self.vel = [0, self.speed]
        elif side == 'bottom':
            self.pos = [random.randint(0, SCREEN_WIDTH), SCREEN_HEIGHT + self.size]
            self.vel = [0, -self.speed]
        elif side == 'left':
            self.pos = [-self.size, random.randint(0, SCREEN_HEIGHT)]
            self.vel = [self.speed, 0]
        else:  # right
            self.pos = [SCREEN_WIDTH + self.size, random.randint(0, SCREEN_HEIGHT)]
            self.vel = [-self.speed, 0]
        
        self.shot_timer = 0

    def update(self, player_pos):
        self.pos[0] += self.vel[0]
        self.pos[1] += self.vel[1]
        
        # Screen wraparound
        if self.pos[0] < -self.size:
            self.pos[0] = SCREEN_WIDTH + self.size
        elif self.pos[0] > SCREEN_WIDTH + self.size:
            self.pos[0] = -self.size
        if self.pos[1] < -self.size:
            self.pos[1] = SCREEN_HEIGHT + self.size
        elif self.pos[1] > SCREEN_HEIGHT + self.size:
            self.pos[1] = -self.size
        
        # Shooting logic
        self.shot_timer += 1
        if self.shot_timer >= self.fire_rate:
            self.shot_timer = 0
            if self.type == 'small':
                dx = player_pos[0] - self.pos[0]
                dy = player_pos[1] - self.pos[1]
                angle = math.atan2(dy, dx)
            else:  # Large UFO fires randomly
                angle = random.uniform(0, 2 * math.pi)
            bullet = Bullet(self.pos, math.degrees(angle))
            return bullet
        return None

    def draw(self, screen):
        # Draw saucer shape (wireframe ellipse)
        pygame.draw.ellipse(screen, WHITE, (
            self.pos[0] - self.size/2,
            self.pos[1] - self.size/4,
            self.size,
            self.size/2
        ), 2)

def main():
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Asteroids")
    clock = pygame.time.Clock()
    
    # Game state variables
    score = 0
    lives = 3
    ship = Ship()
    asteroids = [Asteroid(3) for _ in range(4)]  # Start with 4 large asteroids
    bullets = []
    ufos = []
    ufo_spawn_timer = 0
    max_ufos = 2
    
    font = pygame.font.SysFont(None, 36)
    game_over_font = pygame.font.SysFont(None, 72)
    
    running = True
    while running:
        # Event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        keys = pygame.key.get_pressed()
        # Rotation controls
        if keys[pygame.K_LEFT]:
            ship.angle -= ship.rotation_speed
        if keys[pygame.K_RIGHT]:
            ship.angle += ship.rotation_speed
        # Thrust control
        if keys[pygame.K_UP]:
            ship.thrusting = True
        else:
            ship.thrusting = False
        # Shooting
        if keys[pygame.K_SPACE]:
            angle_rad = math.radians(ship.angle)
            bullet_pos = [
                ship.pos[0] + 25 * math.sin(angle_rad),
                ship.pos[1] - 25 * math.cos(angle_rad)
            ]
            bullet = Bullet(bullet_pos, ship.angle)
            bullets.append(bullet)
        
        # Update game objects
        ship.update()
        
        for asteroid in asteroids[:]:
            asteroid.update()
        
        for bullet in bullets[:]:
            result = bullet.update()
            if not result:
                bullets.remove(bullet)
        
        for ufo in ufos[:]:
            new_bullet = ufo.update(ship.pos)
            if new_bullet is not None:
                bullets.append(new_bullet)
        
        # Collision detection
        # Bullet vs Asteroid
        for bullet in bullets[:]:
            for asteroid in asteroids[:]:
                dx = bullet.pos[0] - asteroid.pos[0]
                dy = bullet.pos[1] - asteroid.pos[1]
                distance = math.sqrt(dx*dx + dy*dy)
                if distance < (bullet.radius + asteroid.radius):
                    bullets.remove(bullet)
                    asteroids.remove(asteroid)
                    score += 20 * asteroid.size  # Large:60, Medium:40, Small:20
                    new_asteroids = asteroid.split()
                    asteroids.extend(new_asteroids)
                    break
        
        # Bullet vs UFO
        for bullet in bullets[:]:
            for ufo in ufos[:]:
                dx = bullet.pos[0] - ufo.pos[0]
                dy = bullet.pos[1] - ufo.pos[1]
                distance = math.sqrt(dx*dx + dy*dy)
                if distance < (bullet.radius + ufo.size/2):
                    bullets.remove(bullet)
                    ufos.remove(ufo)
                    score += 50 if ufo.type == 'large' else 100
                    break
        
        # Ship vs Asteroid
        for asteroid in asteroids[:]:
            dx = ship.pos[0] - asteroid.pos[0]
            dy = ship.pos[1] - asteroid.pos[1]
            distance = math.sqrt(dx*dx + dy*dy)
            if distance < (ship.radius + asteroid.radius):
                lives -= 1
                asteroids.remove(asteroid)
                ship = Ship()  # Reset ship position and velocity
                break
        
        # Ship vs UFO bullets
        for bullet in bullets[:]:
            dx = ship.pos[0] - bullet.pos[0]
            dy = ship.pos[1] - bullet.pos[1]
            distance = math.sqrt(dx*dx + dy*dy)
            if distance < (ship.radius + bullet.radius):
                lives -= 1
                bullets.remove(bullet)
                ship = Ship()
                break
        
        # Spawn new UFOs
        ufo_spawn_timer += 1
        if ufo_spawn_timer >= 300 and len(ufos) < max_ufos:
            ufo_type = 'large' if random.random() > 0.5 else 'small'
            ufos.append(UFO(ufo_type))
            ufo_spawn_timer = 0
        
        # Check for cleared asteroids
        if len(asteroids) == 0:
            asteroids = [Asteroid(3) for _ in range(4)]
        
        # Extra life at 10,000 points
        if score >= 10000 and lives < 5:
            lives += 1
            score -= 10000
        
        # Draw everything
        screen.fill(BLACK)
        
        if lives <= 0:
            # Game over screen
            text = game_over_font.render("GAME OVER", True, WHITE)
            screen.blit(text, (SCREEN_WIDTH//2 - text.get_width()//2, SCREEN_HEIGHT//2))
            pygame.display.flip()
            pygame.time.wait(3000)
            running = False
        else:
            # Draw game elements
            ship.draw(screen)
            for asteroid in asteroids:
                asteroid.draw(screen)
            for bullet in bullets:
                bullet.draw(screen)
            for ufo in ufos:
                ufo.draw(screen)
            
            # Display score and lives
            score_text = font.render(f"Score: {score}", True, WHITE)
            lives_text = font.render(f"Lives: {lives}", True, WHITE)
            screen.blit(score_text, (10, 10))
            screen.blit(lives_text, (SCREEN_WIDTH - 150, 10))
            pygame.display.flip()
        
        clock.tick(FPS)
    
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()