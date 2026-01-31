import pygame
import math
import random
import sys

# Initialize Pygame
pygame.init()

# Constants
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
FPS = 60
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)

# Physics constants
SHIP_ACCELERATION = 10.3
SHIP_FRICTION = 0.98
SHIP_ROTATION_SPEED = 3.5
BULLET_SPEED = 470
ASTEROID_SIZES = [40, 25, 15]  # Large, Medium, Small
ASTEROID_SPEEDS = [100.0, 100.8, 200.5]
ASTEROID_POINTS = [20, 50, 100]
UFO_LARGE_SPEED = 10.0
UFO_SMALL_SPEED = 120.5
UFO_LARGE_SHOOT_CHANCE = 0.005
UFO_SMALL_SHOOT_CHANCE = 0.01

# Game state
class GameState:
    def __init__(self):
        self.score = 0
        self.lives = 3
        self.level = 1
        self.game_over = False
        self.paused = False
        self.next_level = False
        self.extra_life_score = 10000

# Vector utility class for 2D math
class Vec2:
    def __init__(self, x=0, y=0):
        self.x = x
        self.y = y
    
    def add(self, other):
        return Vec2(self.x + other.x, self.y + other.y)
    
    def sub(self, other):
        return Vec2(self.x - other.x, self.y - other.y)
    
    def mul(self, scalar):
        return Vec2(self.x * scalar, self.y * scalar)
    
    def dot(self, other):
        return self.x * other.x + self.y * other.y
    
    def length(self):
        return math.sqrt(self.x * self.x + self.y * self.y)
    
    def normalize(self):
        l = self.length()
        if l == 0:
            return Vec2(0, 0)
        return Vec2(self.x / l, self.y / l)
    
    def rotate(self, angle):
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        return Vec2(
            self.x * cos_a - self.y * sin_a,
            self.x * sin_a + self.y * cos_a
        )
    
    def copy(self):
        return Vec2(self.x, self.y)

# Game object base class
class GameObject:
    def __init__(self, position, velocity=None):
        self.position = position if position else Vec2(0, 0)
        self.velocity = velocity if velocity else Vec2(0, 0)
        self.rotation = 0
        self.alive = True
    
    def update(self, dt):
        # Apply physics: move with inertia
        self.position = self.position.add(self.velocity.mul(dt))
        
        # Screen wraparound
        if self.position.x < 0:
            self.position.x = SCREEN_WIDTH
        elif self.position.x > SCREEN_WIDTH:
            self.position.x = 0
        if self.position.y < 0:
            self.position.y = SCREEN_HEIGHT
        elif self.position.y > SCREEN_HEIGHT:
            self.position.y = 0
    
    def draw(self, screen):
        pass

# Player Ship
class Ship(GameObject):
    def __init__(self, position):
        super().__init__(position)
        self.rotation = 0
        self.velocity = Vec2(0, 0)
        self.thrusting = False
        self.thrust_frame = 0
        self.invulnerable = False
        self.invulnerable_timer = 0
        
    def update(self, dt):
        # Handle invulnerability flash
        if self.invulnerable:
            self.invulnerable_timer -= dt
            if self.invulnerable_timer <= 0:
                self.invulnerable = False
        
        # Update thrust animation
        if self.thrusting:
            self.thrust_frame = (self.thrust_frame + 1) % 8
        else:
            self.thrust_frame = 0
            
        # Apply physics
        super().update(dt)
        
        # Handle rotation and thrust
        keys = pygame.key.get_pressed()
        
        if keys[pygame.K_LEFT]:
            self.rotation -= SHIP_ROTATION_SPEED * dt
        if keys[pygame.K_RIGHT]:
            self.rotation += SHIP_ROTATION_SPEED * dt
            
        if keys[pygame.K_UP]:
            self.thrusting = True
            # Apply thrust in direction of ship's rotation
            thrust_vector = Vec2(math.cos(self.rotation), math.sin(self.rotation)).mul(SHIP_ACCELERATION)
            self.velocity = self.velocity.add(thrust_vector)
        else:
            self.thrusting = False
            
        # Apply friction to slow down over time
        self.velocity = self.velocity.mul(SHIP_FRICTION)
        
    def shoot(self):
        # Create bullet in direction of ship's rotation
        bullet_pos = Vec2(
            self.position.x + math.cos(self.rotation) * 15,
            self.position.y + math.sin(self.rotation) * 15
        )
        bullet_vel = Vec2(math.cos(self.rotation), math.sin(self.rotation)).mul(BULLET_SPEED)
        return Bullet(bullet_pos, bullet_vel.add(self.velocity))
    
    def draw(self, screen):
        if not self.alive:
            return
            
        # Flash effect when invulnerable
        if self.invulnerable and int(self.invulnerable_timer * 10) % 2 == 0:
            return
            
        # Draw ship as triangle (wireframe)
        points = []
        ship_length = 15
        ship_width = 8
        
        # Triangle points: nose, left tail, right tail
        nose = Vec2(ship_length, 0).rotate(self.rotation).add(self.position)
        left_tail = Vec2(-ship_length/2, ship_width/2).rotate(self.rotation).add(self.position)
        right_tail = Vec2(-ship_length/2, -ship_width/2).rotate(self.rotation).add(self.position)
        
        points = [nose, left_tail, right_tail]
        
        # Draw ship outline
        pygame.draw.lines(screen, WHITE, True, [(p.x, p.y) for p in points], 1)
        
        # Draw thrust flame if thrusting
        if self.thrusting:
            flame_length = 12
            flame_width = 4
            
            # Flame points - three points forming a jagged flame
            flame_base = Vec2(-ship_length/2, 0).rotate(self.rotation).add(self.position)
            
            # Create jagged flame effect based on frame
            flame_offset = (self.thrust_frame % 3) * 1.5
            
            flame_points = [
                flame_base,
                Vec2(-ship_length + flame_offset, -flame_width/2).rotate(self.rotation).add(self.position),
                Vec2(-ship_length, 0).rotate(self.rotation).add(self.position),
                Vec2(-ship_length + flame_offset, flame_width/2).rotate(self.rotation).add(self.position)
            ]
            
            # Draw flame as a polygon (wireframe)
            pygame.draw.lines(screen, WHITE, False, [(p.x, p.y) for p in flame_points], 1)

# Bullet
class Bullet(GameObject):
    def __init__(self, position, velocity):
        super().__init__(position, velocity)
        self.lifetime = 60  # frames
        
    def update(self, dt):
        super().update(dt)
        self.lifetime -= 1
        if self.lifetime <= 0:
            self.alive = False
            
    def draw(self, screen):
        if not self.alive:
            return
        pygame.draw.circle(screen, WHITE, (int(self.position.x), int(self.position.y)), 1)

# Asteroid
class Asteroid(GameObject):
    def __init__(self, position, size_index=0):
        # Generate irregular jagged shape with random points
        self.size_index = size_index
        self.size = ASTEROID_SIZES[size_index]
        
        # Create random jagged polygon (5-8 points)
        self.points = []
        num_points = random.randint(5, 8)
        for i in range(num_points):
            angle = (2 * math.pi / num_points) * i
            radius = self.size * (0.7 + random.random() * 0.6)  # Random variation
            x = math.cos(angle) * radius
            y = math.sin(angle) * radius
            self.points.append(Vec2(x, y))
            
        super().__init__(position)
        
        # Random velocity and rotation
        angle = random.uniform(0, 2 * math.pi)
        speed = ASTEROID_SPEEDS[size_index]
        self.velocity = Vec2(math.cos(angle), math.sin(angle)).mul(speed)
        self.rotation_speed = (random.random() - 0.5) * 3
        
    def update(self, dt):
        super().update(dt)
        self.rotation += self.rotation_speed * dt
        
    def draw(self, screen):
        if not self.alive:
            return
            
        # Transform points by position and rotation
        transformed_points = []
        for point in self.points:
            rotated_point = point.rotate(self.rotation).add(self.position)
            transformed_points.append(rotated_point)
            
        # Draw wireframe polygon
        points_list = [(p.x, p.y) for p in transformed_points]
        pygame.draw.lines(screen, WHITE, True, points_list, 1)

# UFO (Flying Saucer)
class UFO(GameObject):
    def __init__(self, ufo_type="large", ship_position=None):
        super().__init__(Vec2(0, 0))
        self.ufo_type = ufo_type
        self.size = 30 if ufo_type == "large" else 18
        self.shoot_timer = 0
        self.ship_position = ship_position  # Reference to player ship position
        
        # Spawn at edge of screen
        side = random.randint(0, 3)  # 0: top, 1: right, 2: bottom, 3: left
        if side == 0:  # Top
            self.position = Vec2(random.uniform(0, SCREEN_WIDTH), -self.size)
        elif side == 1:  # Right
            self.position = Vec2(SCREEN_WIDTH + self.size, random.uniform(0, SCREEN_HEIGHT))
        elif side == 2:  # Bottom
            self.position = Vec2(random.uniform(0, SCREEN_WIDTH), SCREEN_HEIGHT + self.size)
        else:  # Left
            self.position = Vec2(-self.size, random.uniform(0, SCREEN_HEIGHT))
            
        # Set speed and target direction
        self.speed = UFO_LARGE_SPEED if ufo_type == "large" else UFO_SMALL_SPEED
        
        # Calculate target point (random position on opposite side)
        target_side = (side + 2) % 4
        if target_side == 0:  # Top
            target_pos = Vec2(random.uniform(0, SCREEN_WIDTH), -self.size)
        elif target_side == 1:  # Right
            target_pos = Vec2(SCREEN_WIDTH + self.size, random.uniform(0, SCREEN_HEIGHT))
        elif target_side == 2:  # Bottom
            target_pos = Vec2(random.uniform(0, SCREEN_WIDTH), SCREEN_HEIGHT + self.size)
        else:  # Left
            target_pos = Vec2(-self.size, random.uniform(0, SCREEN_HEIGHT))
            
        direction = target_pos.sub(self.position).normalize()
        self.velocity = direction.mul(self.speed)
        
        # For small UFOs, make them more precise and faster to shoot
        self.shoot_chance = UFO_LARGE_SHOOT_CHANCE if ufo_type == "large" else UFO_SMALL_SHOOT_CHANCE
        
    def update(self, dt):
        super().update(dt)
        
        # Shoot randomly (smaller chance for large UFOs)
        self.shoot_timer += dt
        if random.random() < self.shoot_chance and self.shoot_timer > 1.0:
            self.shoot()
            self.shoot_timer = 0
            
    def shoot(self):
        # Shoot toward player position (fixed - now using passed ship_position)
        if self.ship_position:
            # Calculate direction to player
            direction = self.ship_position.sub(self.position).normalize()
            bullet_pos = Vec2(
                self.position.x + math.cos(self.rotation) * self.size,
                self.position.y + math.sin(self.rotation) * self.size
            )
            bullet_vel = direction.mul(BULLET_SPEED * 1.5)
            return Bullet(bullet_pos, bullet_vel)
        else:
            # Fallback: shoot in current direction if no ship position available
            bullet_pos = Vec2(
                self.position.x + math.cos(self.rotation) * self.size,
                self.position.y + math.sin(self.rotation) * self.size
            )
            bullet_vel = Vec2(math.cos(self.rotation), math.sin(self.rotation)).mul(BULLET_SPEED * 1.5)
            return Bullet(bullet_pos, bullet_vel)
        
    def draw(self, screen):
        if not self.alive:
            return
            
        # Draw UFO as a classic flying saucer wireframe
        points = []
        
        # Main body - oval shape
        for i in range(8):
            angle = (2 * math.pi / 8) * i
            x = math.cos(angle) * self.size
            y = math.sin(angle) * (self.size / 2)
            points.append(Vec2(x, y))
            
        # Rotate all points by UFO rotation
        rotated_points = []
        for point in points:
            rotated = point.rotate(self.rotation).add(self.position)
            rotated_points.append(rotated)
            
        # Draw main body as wireframe
        for i in range(len(rotated_points)):
            next_i = (i + 1) % len(rotated_points)
            pygame.draw.line(screen, WHITE, 
                           (rotated_points[i].x, rotated_points[i].y),
                           (rotated_points[next_i].x, rotated_points[next_i].y), 1)
        
        # Draw center detail
        center = Vec2(0, 0).rotate(self.rotation).add(self.position)
        pygame.draw.circle(screen, WHITE, (int(center.x), int(center.y)), 2, 1)

# Game class - main game logic
class AsteroidsGame:
    def __init__(self):
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Asteroids - 1979 Atari Original")
        self.clock = pygame.time.Clock()
        
        # Initialize game state
        self.state = GameState()
        
        # Game objects
        self.ship = None
        self.bullets = []
        self.asteroids = []
        self.ufos = []
        
        # Start the game
        self.start_new_game()
        
    def start_new_game(self):
        """Initialize a new game"""
        self.state.score = 0
        self.state.lives = 3
        self.state.level = 1
        self.state.game_over = False
        
        # Create ship in center
        self.ship = Ship(Vec2(SCREEN_WIDTH/2, SCREEN_HEIGHT/2))
        
        # Clear all other objects
        self.bullets = []
        self.ufos = []
        
        # Spawn initial asteroids (4 large)
        self.spawn_asteroids(4, 0)  # 4 large asteroids
        
    def start_new_level(self):
        """Start a new level with more asteroids"""
        self.state.level += 1
        self.ship = Ship(Vec2(SCREEN_WIDTH/2, SCREEN_HEIGHT/2))
        self.bullets = []
        self.ufos = []  # Clear UFOs on level start
        
        # Spawn more asteroids based on level
        asteroid_count = min(4 + self.state.level, 8)  # Max 8 asteroids
        self.spawn_asteroids(asteroid_count, 0)
        
    def spawn_asteroids(self, count, size_index):
        """Spawn specified number of asteroids at random positions"""
        for _ in range(count):
            while True:
                # Generate position away from ship (at least 150px distance)
                x = random.uniform(0, SCREEN_WIDTH)
                y = random.uniform(0, SCREEN_HEIGHT)
                ship_pos = self.ship.position
                dist = math.sqrt((x - ship_pos.x)**2 + (y - ship_pos.y)**2)
                
                if dist > 150:  # Safe distance from player
                    asteroid = Asteroid(Vec2(x, y), size_index)
                    self.asteroids.append(asteroid)
                    break
    
    def spawn_ufo(self):
        """Spawn a random UFO (large or small)"""
        ufo_type = "large" if random.random() > 0.5 else "small"
        # Pass ship position to UFO for accurate targeting
        ufo = UFO(ufo_type, self.ship.position)
        self.ufos.append(ufo)
    
    def check_collisions(self):
        """Check for collisions between game objects"""
        
        # Check bullet-asteroid collisions
        for bullet in self.bullets[:]:
            if not bullet.alive:
                continue
                
            for asteroid in self.asteroids[:]:
                if not asteroid.alive:
                    continue
                    
                # Simple distance check
                dist = math.sqrt((bullet.position.x - asteroid.position.x)**2 + 
                               (bullet.position.y - asteroid.position.y)**2)
                
                if dist < asteroid.size * 0.8:  # Collision detected
                    bullet.alive = False
                    asteroid.alive = False
                    
                    # Add points based on asteroid size
                    self.state.score += ASTEROID_POINTS[asteroid.size_index]
                    
                    # Split asteroid if not smallest size
                    if asteroid.size_index < len(ASTEROID_SIZES) - 1:
                        for _ in range(2):
                            new_asteroid = Asteroid(
                                Vec2(asteroid.position.x, asteroid.position.y),
                                asteroid.size_index + 1
                            )
                            # Give it a random velocity
                            angle = random.uniform(0, 2 * math.pi)
                            speed = ASTEROID_SPEEDS[asteroid.size_index + 1]
                            new_asteroid.velocity = Vec2(
                                math.cos(angle), math.sin(angle)
                            ).mul(speed)
                            self.asteroids.append(new_asteroid)
                    
                    # Check for extra life
                    if self.state.score >= self.state.extra_life_score:
                        self.state.lives += 1
                        self.state.extra_life_score += 10000
                    
                    break
        
        # Check bullet-UFO collisions
        for bullet in self.bullets[:]:
            if not bullet.alive:
                continue
                
            for ufo in self.ufos[:]:
                if not ufo.alive:
                    continue
                    
                dist = math.sqrt((bullet.position.x - ufo.position.x)**2 + 
                               (bullet.position.y - ufo.position.y)**2)
                
                if dist < ufo.size * 0.8:  # Collision detected
                    bullet.alive = False
                    ufo.alive = False
                    
                    # Award points for UFO
                    self.state.score += 1000 if ufo.ufo_type == "large" else 2000
                    
                    break
        
        # Check ship-asteroid collisions (only if not invulnerable)
        if self.ship.alive and not self.ship.invulnerable:
            for asteroid in self.asteroids[:]:
                if not asteroid.alive:
                    continue
                    
                dist = math.sqrt((self.ship.position.x - asteroid.position.x)**2 + 
                               (self.ship.position.y - asteroid.position.y)**2)
                
                if dist < asteroid.size * 0.8:  # Collision detected
                    self.ship.invulnerable = True
                    self.ship.invulnerable_timer = 3.0  # 3 seconds invulnerability
                    
                    # Remove ship and reduce life
                    self.state.lives -= 1
                    if self.state.lives <= 0:
                        self.state.game_over = True
                    else:
                        # Reset ship position
                        self.ship.position = Vec2(SCREEN_WIDTH/2, SCREEN_HEIGHT/2)
                        self.ship.velocity = Vec2(0, 0)
                        self.ship.rotation = 0
                        
                    break
        
        # Check ship-UFO collisions (only if not invulnerable)
        if self.ship.alive and not self.ship.invulnerable:
            for ufo in self.ufos[:]:
                if not ufo.alive:
                    continue
                    
                dist = math.sqrt((self.ship.position.x - ufo.position.x)**2 + 
                               (self.ship.position.y - ufo.position.y)**2)
                
                if dist < ufo.size * 0.8:  # Collision detected
                    self.ship.invulnerable = True
                    self.ship.invulnerable_timer = 3.0
                    
                    self.state.lives -= 1
                    if self.state.lives <= 0:
                        self.state.game_over = True
                    else:
                        self.ship.position = Vec2(SCREEN_WIDTH/2, SCREEN_HEIGHT/2)
                        self.ship.velocity = Vec2(0, 0)
                        self.ship.rotation = 0
                        
                    break
        
        # Clean up dead objects
        self.bullets = [b for b in self.bullets if b.alive]
        self.asteroids = [a for a in self.asteroids if a.alive]
        self.ufos = [u for u in self.ufos if u.alive]
        
        # Check if level complete (no asteroids left)
        if len(self.asteroids) == 0 and not self.state.game_over:
            self.start_new_level()
            
    def update(self, dt):
        """Update game state"""
        if self.state.game_over or self.state.paused:
            return
            
        # Update ship
        self.ship.update(dt)
        
        # Update bullets
        for bullet in self.bullets:
            bullet.update(dt)
            
        # Update asteroids
        for asteroid in self.asteroids:
            asteroid.update(dt)
            
        # Update UFOs - pass current ship position to each UFO
        for ufo in self.ufos:
            ufo.ship_position = self.ship.position  # Update reference
            ufo.update(dt)
            
        # Randomly spawn UFO (1% chance per frame)
        if random.random() < 0.005 and len(self.ufos) == 0:
            self.spawn_ufo()
            
        # Check collisions
        self.check_collisions()
        
        # Handle keyboard input for shooting
        keys = pygame.key.get_pressed()
        if keys[pygame.K_SPACE]:
            if len(self.bullets) < 10:  # Limit bullets on screen
                bullet = self.ship.shoot()
                self.bullets.append(bullet)
                
    def draw(self):
        """Draw everything to screen"""
        self.screen.fill(BLACK)
        
        # Draw ship
        self.ship.draw(self.screen)
        
        # Draw bullets
        for bullet in self.bullets:
            bullet.draw(self.screen)
            
        # Draw asteroids
        for asteroid in self.asteroids:
            asteroid.draw(self.screen)
            
        # Draw UFOs
        for ufo in self.ufos:
            ufo.draw(self.screen)
            
        # Draw UI
        font = pygame.font.SysFont('Courier', 20, bold=True)
        
        # Score
        score_text = font.render(f"SCORE: {self.state.score:06d}", True, WHITE)
        self.screen.blit(score_text, (10, 10))
        
        # Lives
        lives_text = font.render(f"LIVES: {self.state.lives}", True, WHITE)
        self.screen.blit(lives_text, (SCREEN_WIDTH - 150, 10))
        
        # Level
        level_text = font.render(f"LEVEL: {self.state.level}", True, WHITE)
        self.screen.blit(level_text, (SCREEN_WIDTH/2 - 40, 10))
        
        # Game over message
        if self.state.game_over:
            game_over_font = pygame.font.SysFont('Courier', 48, bold=True)
            game_over_text = game_over_font.render("GAME OVER", True, WHITE)
            text_rect = game_over_text.get_rect(center=(SCREEN_WIDTH/2, SCREEN_HEIGHT/2))
            self.screen.blit(game_over_text, text_rect)
            
            restart_font = pygame.font.SysFont('Courier', 24)
            restart_text = restart_font.render("Press R to Restart", True, WHITE)
            restart_rect = restart_text.get_rect(center=(SCREEN_WIDTH/2, SCREEN_HEIGHT/2 + 50))
            self.screen.blit(restart_text, restart_rect)
            
        # Pause message
        if self.state.paused:
            pause_font = pygame.font.SysFont('Courier', 48, bold=True)
            pause_text = pause_font.render("PAUSED", True, WHITE)
            text_rect = pause_text.get_rect(center=(SCREEN_WIDTH/2, SCREEN_HEIGHT/2))
            self.screen.blit(pause_text, text_rect)
            
        # Update display
        pygame.display.flip()
        
    def handle_events(self):
        """Handle user input"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
                
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return False
                    
                elif event.key == pygame.K_r and self.state.game_over:
                    self.start_new_game()
                    
                elif event.key == pygame.K_p:
                    self.state.paused = not self.state.paused
                    
        return True
        
    def run(self):
        """Main game loop"""
        running = True
        last_time = pygame.time.get_ticks() / 1000.0
        
        while running:
            # Calculate delta time for smooth physics
            current_time = pygame.time.get_ticks() / 1000.0
            dt = min(current_time - last_time, 0.1)  # Cap at 0.1s to prevent big jumps
            last_time = current_time
            
            running = self.handle_events()
            
            if not self.state.paused and not self.state.game_over:
                self.update(dt)
                
            self.draw()
            self.clock.tick(FPS)
            
        pygame.quit()
        sys.exit()

# Run the game
if __name__ == "__main__":
    game = AsteroidsGame()
    game.run()