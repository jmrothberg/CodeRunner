import pygame
import math
import random
import sys

# Initialize pygame
pygame.init()

# Constants
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
FPS = 60
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)

# Game settings
PLAYER_SPEED = 5
PLAYER_ROTATION_SPEED = 5
PLAYER_ACCELERATION = 0.5
PLAYER_DECELERATION = 0.98
BULLET_SPEED = 10
BULLET_LIFETIME = 60  # frames
ASTEROID_SIZES = [3, 2, 1]  # Large, medium, small
ASTEROID_SPEEDS = [1, 2, 3]
ASTEROID_SPAWN_RATE = 30  # frames between asteroid spawns
ASTEROID_MIN_SPAWN = 4  # minimum number of asteroids to spawn initially
MAX_ASTEROIDS = 10  # maximum number of asteroids on screen

# Set up the display
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Asteroids")
clock = pygame.time.Clock()

# Load font
font = pygame.font.SysFont('Arial', 36)

class Vector2D:
    """Simple 2D vector class for position and velocity"""
    def __init__(self, x=0, y=0):
        self.x = x
        self.y = y
    
    def __add__(self, other):
        return Vector2D(self.x + other.x, self.y + other.y)
    
    def __sub__(self, other):
        return Vector2D(self.x - other.x, self.y - other.y)
    
    def __mul__(self, scalar):
        return Vector2D(self.x * scalar, self.y * scalar)
    
    def __rmul__(self, scalar):
        return self.__mul__(scalar)
    
    def magnitude(self):
        return math.sqrt(self.x**2 + self.y**2)
    
    def normalize(self):
        mag = self.magnitude()
        if mag > 0:
            return Vector2D(self.x / mag, self.y / mag)
        return Vector2D(0, 0)
    
    def rotate(self, angle):
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        return Vector2D(
            self.x * cos_a - self.y * sin_a,
            self.x * sin_a + self.y * cos_a
        )
    
    def copy(self):
        return Vector2D(self.x, self.y)

class Player:
    def __init__(self):
        self.position = Vector2D(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2)
        self.velocity = Vector2D(0, 0)
        self.rotation = 0  # in radians
        self.rotation_speed = PLAYER_ROTATION_SPEED * (math.pi / 180)  # convert to radians
        self.speed = PLAYER_SPEED
        self.acceleration = PLAYER_ACCELERATION
        self.deceleration = PLAYER_DECELERATION
        self.radius = 15
        self.is_thrusting = False
        self.invulnerable = False
        self.invulnerable_timer = 0
        self.lives = 3
        self.score = 0
        
    def update(self):
        # Handle deceleration when not thrusting
        if not self.is_thrusting:
            self.velocity.x *= self.deceleration
            self.velocity.y *= self.deceleration
            
        # Update position based on velocity
        self.position += self.velocity
        
        # Wrap around screen edges
        self.wrap_around()
        
        # Update invulnerability timer
        if self.invulnerable:
            self.invulnerable_timer -= 1
            if self.invulnerable_timer <= 0:
                self.invulnerable = False
    
    def wrap_around(self):
        """Wrap the player around the screen edges"""
        if self.position.x < 0:
            self.position.x = SCREEN_WIDTH
        elif self.position.x > SCREEN_WIDTH:
            self.position.x = 0
        if self.position.y < 0:
            self.position.y = SCREEN_HEIGHT
        elif self.position.y > SCREEN_HEIGHT:
            self.position.y = 0
    
    def rotate_left(self):
        self.rotation -= self.rotation_speed
    
    def rotate_right(self):
        self.rotation += self.rotation_speed
    
    def thrust(self):
        self.is_thrusting = True
        # Calculate thrust vector based on current rotation
        thrust_vector = Vector2D(0, -self.acceleration).rotate(self.rotation)
        self.velocity += thrust_vector
        
        # Limit maximum speed
        if self.velocity.magnitude() > self.speed:
            self.velocity = self.velocity.normalize() * self.speed
    
    def stop_thrust(self):
        self.is_thrusting = False
    
    def shoot(self):
        # Calculate bullet direction based on player rotation
        bullet_direction = Vector2D(0, -1).rotate(self.rotation)
        bullet_velocity = bullet_direction * BULLET_SPEED
        bullet_position = self.position + bullet_direction * self.radius
        return Bullet(bullet_position, bullet_velocity)
    
    def draw(self):
    # Draw the ship as a triangle
        tip = Vector2D(0, -self.radius).rotate(self.rotation) + self.position
        left = Vector2D(-self.radius/2, self.radius/2).rotate(self.rotation) + self.position
        right = Vector2D(self.radius/2, self.radius/2).rotate(self.rotation) + self.position
    
        # Flashing effect when invulnerable
        if self.invulnerable and self.invulnerable_timer % 10 < 5:
            return  # Skip drawing during flash
    
    # Convert Vector2D objects to tuples for pygame
        pygame.draw.polygon(screen, WHITE, [
            (int(tip.x), int(tip.y)), 
            (int(left.x), int(left.y)), 
            (int(right.x), int(right.y))
        ], 2)
    
    # Draw thrust if accelerating
        if self.is_thrusting:
            thrust_tip = Vector2D(0, self.radius/2).rotate(self.rotation) + self.position
            thrust_left = Vector2D(-self.radius/4, self.radius/4).rotate(self.rotation) + self.position
            thrust_right = Vector2D(self.radius/4, self.radius/4).rotate(self.rotation) + self.position
            pygame.draw.polygon(screen, RED, [
                (int(thrust_tip.x), int(thrust_tip.y)), 
                (int(thrust_left.x), int(thrust_left.y)), 
                (int(thrust_right.x), int(thrust_right.y))
             ])
    
    def reset_position(self):
        self.position = Vector2D(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2)
        self.velocity = Vector2D(0, 0)
        self.invulnerable = True
        self.invulnerable_timer = 180  # 3 seconds at 60 FPS
    
    def get_hitbox(self):
        return pygame.Rect(
            self.position.x - self.radius,
            self.position.y - self.radius,
            self.radius * 2,
            self.radius * 2
        )

class Bullet:
    def __init__(self, position, velocity):
        self.position = position
        self.velocity = velocity
        self.lifetime = BULLET_LIFETIME
        self.radius = 2
    
    def update(self):
        self.position += self.velocity
        self.lifetime -= 1
        
        # Wrap around screen edges
        if self.position.x < 0:
            self.position.x = SCREEN_WIDTH
        elif self.position.x > SCREEN_WIDTH:
            self.position.x = 0
        if self.position.y < 0:
            self.position.y = SCREEN_HEIGHT
        elif self.position.y > SCREEN_HEIGHT:
            self.position.y = 0
    
    def draw(self):
        pygame.draw.circle(screen, WHITE, (int(self.position.x), int(self.position.y)), self.radius)
    
    def is_dead(self):
        return self.lifetime <= 0

class Asteroid:
    def __init__(self, size=3, position=None):
        self.size = size  # 1=small, 2=medium, 3=large
        self.radius = 20 * size
        self.speed = ASTEROID_SPEEDS[size - 1]
        
        # If no position provided, spawn at edge
        if position is None:
            # Randomly choose edge to spawn from
            edge = random.choice(['top', 'bottom', 'left', 'right'])
            if edge == 'top':
                position = Vector2D(random.randint(0, SCREEN_WIDTH), 0)
            elif edge == 'bottom':
                position = Vector2D(random.randint(0, SCREEN_WIDTH), SCREEN_HEIGHT)
            elif edge == 'left':
                position = Vector2D(0, random.randint(0, SCREEN_HEIGHT))
            else:  # right
                position = Vector2D(SCREEN_WIDTH, random.randint(0, SCREEN_HEIGHT))
        
        self.position = position
        # Random direction
        angle = random.uniform(0, 2 * math.pi)
        self.velocity = Vector2D(math.cos(angle), math.sin(angle)) * self.speed
        self.rotation = 0
        self.rotation_speed = random.uniform(-0.05, 0.05)
        
        # Create a list of points for the asteroid shape
        self.points = []
        for i in range(8):
            angle = i * (2 * math.pi / 8)
            # Add some randomness to make it look more asteroid-like
            radius_variation = random.uniform(0.8, 1.2)
            x = math.cos(angle) * self.radius * radius_variation
            y = math.sin(angle) * self.radius * radius_variation
            self.points.append((x, y))
    
    def update(self):
        self.position += self.velocity
        self.rotation += self.rotation_speed
        
        # Wrap around screen edges
        if self.position.x < 0:
            self.position.x = SCREEN_WIDTH
        elif self.position.x > SCREEN_WIDTH:
            self.position.x = 0
        if self.position.y < 0:
            self.position.y = SCREEN_HEIGHT
        elif self.position.y > SCREEN_HEIGHT:
            self.position.y = 0
    
    def draw(self):
        # Transform points by position and rotation
        transformed_points = []
        for point in self.points:
            # Rotate point
            x, y = point
            rotated_x = x * math.cos(self.rotation) - y * math.sin(self.rotation)
            rotated_y = x * math.sin(self.rotation) + y * math.cos(self.rotation)
            # Translate to position
            transformed_points.append((
                int(rotated_x + self.position.x),
                int(rotated_y + self.position.y)
            ))
        
        # Draw the asteroid
        pygame.draw.polygon(screen, WHITE, transformed_points, 2)
    
    def get_hitbox(self):
        return pygame.Rect(
            self.position.x - self.radius,
            self.position.y - self.radius,
            self.radius * 2,
            self.radius * 2
        )
    
    def break_apart(self):
        """Return a list of smaller asteroids if this one breaks apart"""
        if self.size == 1:  # Small asteroid, doesn't break
            return []
        
        # Create two smaller asteroids
        new_asteroids = []
        for _ in range(2):
            new_size = self.size - 1
            # Spawn slightly offset from the original position
            offset_angle = random.uniform(0, 2 * math.pi)
            offset_distance = self.radius / 2
            new_x = self.position.x + math.cos(offset_angle) * offset_distance
            new_y = self.position.y + math.sin(offset_angle) * offset_distance
            new_asteroids.append(Asteroid(new_size, Vector2D(new_x, new_y)))
        
        return new_asteroids

def create_initial_asteroids():
    """Create the initial set of asteroids"""
    asteroids = []
    for _ in range(ASTEROID_MIN_SPAWN):
        # Make sure asteroids don't spawn too close to the player
        while True:
            asteroid = Asteroid()
            distance = (asteroid.position - Vector2D(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2)).magnitude()
            if distance > 150:  # Keep asteroids away from player spawn
                asteroids.append(asteroid)
                break
    return asteroids

def main():
    # Game state
    player = Player()
    bullets = []
    asteroids = create_initial_asteroids()
    game_over = False
    level_complete = False
    spawn_timer = 0
    paused = False
    
    # Main game loop
    running = True
    while running:
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE and not game_over and not paused:
                    # Shoot bullet
                    bullets.append(player.shoot())
                elif event.key == pygame.K_p:
                    # Toggle pause
                    paused = not paused
                elif event.key == pygame.K_r and game_over:
                    # Restart game
                    player = Player()
                    bullets = []
                    asteroids = create_initial_asteroids()
                    game_over = False
                    level_complete = False
                    spawn_timer = 0
                elif event.key == pygame.K_ESCAPE:
                    running = False
        
        if game_over or paused:
            # Skip game logic if game is over or paused
            pass
        else:
            # Handle player input
            keys = pygame.key.get_pressed()
            if keys[pygame.K_LEFT] or keys[pygame.K_a]:
                player.rotate_left()
            if keys[pygame.K_RIGHT] or keys[pygame.K_d]:
                player.rotate_right()
            if keys[pygame.K_UP] or keys[pygame.K_w]:
                player.thrust()
            else:
                player.stop_thrust()
            
            # Update game objects
            player.update()
            
            # Update bullets
            for bullet in bullets[:]:
                bullet.update()
                if bullet.is_dead():
                    bullets.remove(bullet)
            
            # Update asteroids
            for asteroid in asteroids[:]:
                asteroid.update()
            
            # Check for bullet-asteroid collisions
            for bullet in bullets[:]:
                for asteroid in asteroids[:]:
                    # Simple distance check for collision
                    distance = (bullet.position - asteroid.position).magnitude()
                    if distance < asteroid.radius + bullet.radius:
                        # Remove bullet and asteroid
                        if bullet in bullets:
                            bullets.remove(bullet)
                        if asteroid in asteroids:
                            asteroids.remove(asteroid)
                            
                            # Add score based on asteroid size
                            player.score += 100 * asteroid.size
                            
                            # Break asteroid into smaller ones if not smallest
                            new_asteroids = asteroid.break_apart()
                            asteroids.extend(new_asteroids)
                            
                            # Check if all asteroids are destroyed
                            if len(asteroids) == 0:
                                level_complete = True
                            break
            
            # Check for player-asteroid collisions
            if not player.invulnerable:
                player_hitbox = player.get_hitbox()
                for asteroid in asteroids:
                    asteroid_hitbox = asteroid.get_hitbox()
                    if player_hitbox.colliderect(asteroid_hitbox):
                        # Check for more precise collision using distance
                        distance = (player.position - asteroid.position).magnitude()
                        if distance < player.radius + asteroid.radius:
                            player.lives -= 1
                            if player.lives <= 0:
                                game_over = True
                            else:
                                player.reset_position()
                            break
            
            # Spawn new asteroids if needed
            spawn_timer += 1
            if spawn_timer >= ASTEROID_SPAWN_RATE and len(asteroids) < MAX_ASTEROIDS:
                # Only spawn if we're not in level complete state
                if not level_complete:
                    asteroids.append(Asteroid())
                spawn_timer = 0
            
            # Check if level complete (all asteroids destroyed)
            if level_complete:
                # Spawn more asteroids for next level
                for _ in range(min(ASTEROID_MIN_SPAWN + len(asteroids) // 2, MAX_ASTEROIDS)):
                    while True:
                        asteroid = Asteroid()
                        distance = (asteroid.position - player.position).magnitude()
                        if distance > 150:
                            asteroids.append(asteroid)
                            break
                level_complete = False
        
        # Draw everything
        screen.fill(BLACK)
        
        # Draw player
        player.draw()
        
        # Draw bullets
        for bullet in bullets:
            bullet.draw()
        
        # Draw asteroids
        for asteroid in asteroids:
            asteroid.draw()
        
        # Draw UI
        score_text = font.render(f"Score: {player.score}", True, WHITE)
        screen.blit(score_text, (10, 10))
        
        lives_text = font.render(f"Lives: {player.lives}", True, WHITE)
        screen.blit(lives_text, (10, 50))
        
        # Draw game over or level complete message
        if game_over:
            game_over_text = font.render("GAME OVER", True, RED)
            restart_text = font.render("Press R to Restart or ESC to Quit", True, WHITE)
            screen.blit(game_over_text, (SCREEN_WIDTH//2 - game_over_text.get_width()//2, SCREEN_HEIGHT//2 - 50))
            screen.blit(restart_text, (SCREEN_WIDTH//2 - restart_text.get_width()//2, SCREEN_HEIGHT//2))
        
        if level_complete:
            level_complete_text = font.render("LEVEL COMPLETE!", True, GREEN)
            screen.blit(level_complete_text, (SCREEN_WIDTH//2 - level_complete_text.get_width()//2, SCREEN_HEIGHT//2))
        
        if paused:
            paused_text = font.render("PAUSED - Press P to Resume", True, YELLOW)
            screen.blit(paused_text, (SCREEN_WIDTH//2 - paused_text.get_width()//2, SCREEN_HEIGHT//2))
        
        # Update display
        pygame.display.flip()
        clock.tick(FPS)
    
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()