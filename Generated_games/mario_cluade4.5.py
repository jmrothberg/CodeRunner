import pygame
import math
import random
from enum import Enum
from dataclasses import dataclass
from typing import List, Tuple, Optional

# Initialize Pygame
pygame.init()

# Constants
SCREEN_WIDTH = 1200
SCREEN_HEIGHT = 800
FPS = 60

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (128, 128, 128)
DARK_GRAY = (64, 64, 64)
GREEN = (34, 139, 34)
DARK_GREEN = (0, 100, 0)
LIGHT_GREEN = (144, 238, 144)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)
ORANGE = (255, 165, 0)
PURPLE = (128, 0, 128)
CYAN = (0, 255, 255)
PINK = (255, 192, 203)
BROWN = (139, 69, 19)
GOLD = (255, 215, 0)
SILVER = (192, 192, 192)

# Kart colors
KART_COLORS = [
    RED, BLUE, GREEN, YELLOW, ORANGE, PURPLE, CYAN, PINK
]

class PowerUpType(Enum):
    SHELL = 1
    BANANA = 2
    MUSHROOM = 3
    LIGHTNING = 4
    STAR = 5

@dataclass
class PowerUp:
    type: PowerUpType
    duration: float = 0
    active: bool = False

class Particle:
    """Visual effect particle"""
    def __init__(self, x: float, y: float, color: Tuple[int, int, int], 
                 velocity: Tuple[float, float], lifetime: float, size: int = 3):
        self.x = x
        self.y = y
        self.color = color
        self.vx, self.vy = velocity
        self.lifetime = lifetime
        self.max_lifetime = lifetime
        self.size = size
    
    def update(self, dt: float):
        self.x += self.vx * dt
        self.y += self.vy * dt
        self.lifetime -= dt
        self.vy += 200 * dt  # Gravity
    
    def draw(self, screen: pygame.Surface, camera_x: float, camera_y: float):
        alpha = self.lifetime / self.max_lifetime
        size = int(self.size * alpha)
        if size > 0:
            color = tuple(int(c * alpha) for c in self.color)
            pygame.draw.circle(screen, color, 
                             (int(self.x - camera_x), int(self.y - camera_y)), size)
    
    def is_alive(self) -> bool:
        return self.lifetime > 0

class ItemBox:
    """Collectible item box on track"""
    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y
        self.active = True
        self.respawn_timer = 0
        self.respawn_time = 5.0
        self.rotation = 0
        self.bob_offset = 0
        self.bob_speed = 3
        self.collected_animation = 0
        
    def update(self, dt: float):
        if self.active:
            self.rotation += 90 * dt
            self.bob_offset = math.sin(pygame.time.get_ticks() / 200) * 5
        else:
            self.respawn_timer += dt
            if self.respawn_timer >= self.respawn_time:
                self.active = True
                self.respawn_timer = 0
        
        if self.collected_animation > 0:
            self.collected_animation -= dt
    
    def collect(self) -> PowerUpType:
        if self.active:
            self.active = False
            self.collected_animation = 0.5
            return random.choice(list(PowerUpType))
        return None
    
    def draw(self, screen: pygame.Surface, camera_x: float, camera_y: float):
        screen_x = int(self.x - camera_x)
        screen_y = int(self.y - camera_y - self.bob_offset)
        
        if self.active:
            # Draw rotating box with question mark
            size = 20
            # Shadow
            pygame.draw.rect(screen, (0, 0, 0, 100), 
                           (screen_x - size//2 + 2, screen_y - size//2 + 2, size, size))
            # Box
            pygame.draw.rect(screen, GOLD, 
                           (screen_x - size//2, screen_y - size//2, size, size))
            pygame.draw.rect(screen, ORANGE, 
                           (screen_x - size//2, screen_y - size//2, size, size), 2)
            
            # Question mark
            font = pygame.font.Font(None, 24)
            text = font.render("?", True, WHITE)
            text_rect = text.get_rect(center=(screen_x, screen_y))
            screen.blit(text, text_rect)
        elif self.collected_animation > 0:
            # Explosion effect
            alpha = self.collected_animation / 0.5
            size = int(30 * (1 - alpha))
            pygame.draw.circle(screen, YELLOW, (screen_x, screen_y), size, 2)

class Banana:
    """Banana obstacle"""
    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y
        self.active = True
        self.rotation = random.uniform(0, 360)
    
    def draw(self, screen: pygame.Surface, camera_x: float, camera_y: float):
        if self.active:
            screen_x = int(self.x - camera_x)
            screen_y = int(self.y - camera_y)
            
            # Draw banana shape
            pygame.draw.ellipse(screen, YELLOW, 
                              (screen_x - 8, screen_y - 5, 16, 10))
            pygame.draw.ellipse(screen, (200, 200, 0), 
                              (screen_x - 8, screen_y - 5, 16, 10), 1)

class Shell:
    """Projectile shell"""
    def __init__(self, x: float, y: float, angle: float, speed: float = 400):
        self.x = x
        self.y = y
        self.angle = angle
        self.speed = speed
        self.vx = math.cos(angle) * speed
        self.vy = math.sin(angle) * speed
        self.active = True
        self.lifetime = 5.0
        self.rotation = 0
    
    def update(self, dt: float):
        self.x += self.vx * dt
        self.y += self.vy * dt
        self.lifetime -= dt
        self.rotation += 720 * dt
        
        if self.lifetime <= 0:
            self.active = False
    
    def draw(self, screen: pygame.Surface, camera_x: float, camera_y: float):
        if self.active:
            screen_x = int(self.x - camera_x)
            screen_y = int(self.y - camera_y)
            
            # Draw spinning shell
            pygame.draw.circle(screen, RED, (screen_x, screen_y), 8)
            pygame.draw.circle(screen, DARK_GRAY, (screen_x, screen_y), 8, 2)
            # Spikes
            for i in range(4):
                angle = math.radians(self.rotation + i * 90)
                end_x = screen_x + math.cos(angle) * 10
                end_y = screen_y + math.sin(angle) * 10
                pygame.draw.line(screen, WHITE, (screen_x, screen_y), 
                               (int(end_x), int(end_y)), 2)

class Kart:
    """Racing kart with physics"""
    def __init__(self, x: float, y: float, color: Tuple[int, int, int], 
                 is_player: bool = False, ai_difficulty: float = 1.0):
        self.x = x
        self.y = y
        self.color = color
        self.is_player = is_player
        self.ai_difficulty = ai_difficulty
        
        # Physics
        self.angle = 0
        self.speed = 0
        self.max_speed = 300
        self.acceleration = 200
        self.brake_power = 300
        self.turn_speed = 180
        self.friction = 0.95
        self.off_road_friction = 0.85
        
        # State
        self.lap = 0
        self.checkpoint_index = 0
        self.position = 1
        self.finished = False
        self.finish_time = 0
        
        # Power-ups
        self.power_up: Optional[PowerUp] = None
        self.star_active = False
        self.star_timer = 0
        self.stunned = False
        self.stun_timer = 0
        
        # AI
        self.ai_target_index = 0
        self.ai_stuck_timer = 0
        self.ai_last_x = x
        self.ai_last_y = y
        
        # Animation
        self.wheel_rotation = 0
        self.drift_particles_timer = 0
        
    def update(self, dt: float, track, karts: List['Kart'], 
               item_boxes: List[ItemBox], bananas: List[Banana], 
               shells: List[Shell], particles: List[Particle]):
        """Update kart physics and state"""
        
        # Update power-up timers
        if self.star_active:
            self.star_timer -= dt
            if self.star_timer <= 0:
                self.star_active = False
                print(f"Kart {self.color} star power-up ended")
        
        if self.stunned:
            self.stun_timer -= dt
            if self.stun_timer <= 0:
                self.stunned = False
                print(f"Kart {self.color} recovered from stun")
            else:
                self.speed *= 0.9
                return
        
        # Get input (player or AI)
        if self.is_player:
            accelerate, brake, turn_left, turn_right = self.get_player_input()
        else:
            accelerate, brake, turn_left, turn_right = self.get_ai_input(track, karts, bananas)
        
        # Apply acceleration
        if accelerate and not self.finished:
            self.speed += self.acceleration * dt
        elif brake:
            self.speed -= self.brake_power * dt
        
        # Speed boost from mushroom
        if self.power_up and self.power_up.type == PowerUpType.MUSHROOM and self.power_up.active:
            self.speed = self.max_speed * 1.5
            self.power_up.duration -= dt
            if self.power_up.duration <= 0:
                self.power_up = None
                print(f"Kart {self.color} mushroom boost ended")
        
        # Star speed boost
        if self.star_active:
            self.max_speed = 400
        else:
            self.max_speed = 300
        
        # Clamp speed
        self.speed = max(-self.max_speed * 0.5, min(self.speed, self.max_speed))
        
        # Apply turning
        if self.speed != 0:
            turn_factor = min(abs(self.speed) / self.max_speed, 1.0)
            if turn_left:
                self.angle -= self.turn_speed * turn_factor * dt
            if turn_right:
                self.angle += self.turn_speed * turn_factor * dt
        
        # Update wheel rotation for animation
        self.wheel_rotation += self.speed * dt * 0.5
        
        # Apply friction based on terrain
        on_road = track.is_on_road(self.x, self.y)
        if on_road:
            self.speed *= self.friction
        else:
            self.speed *= self.off_road_friction
            # Drift particles on grass
            self.drift_particles_timer += dt
            if self.drift_particles_timer > 0.05 and abs(self.speed) > 50:
                self.drift_particles_timer = 0
                for _ in range(2):
                    angle = self.angle + random.uniform(-30, 30)
                    vel = (math.cos(math.radians(angle)) * random.uniform(-50, -20),
                          math.sin(math.radians(angle)) * random.uniform(-50, -20))
                    particles.append(Particle(self.x, self.y, DARK_GREEN, vel, 0.5, 2))
        
        # Move kart
        rad_angle = math.radians(self.angle)
        self.x += math.cos(rad_angle) * self.speed * dt
        self.y += math.sin(rad_angle) * self.speed * dt
        
        # Check collisions with other karts
        for other in karts:
            if other != self and not self.star_active:
                dist = math.hypot(self.x - other.x, self.y - other.y)
                if dist < 30:
                    # Bounce
                    angle_to_other = math.atan2(other.y - self.y, other.x - self.x)
                    self.x -= math.cos(angle_to_other) * 2
                    self.y -= math.sin(angle_to_other) * 2
                    self.speed *= 0.7
        
        # Check item box collection
        for box in item_boxes:
            if box.active:
                dist = math.hypot(self.x - box.x, self.y - box.y)
                if dist < 25:
                    power_up_type = box.collect()
                    if power_up_type:
                        self.power_up = PowerUp(power_up_type)
                        print(f"Kart {self.color} collected {power_up_type.name}")
                        # Particle effect
                        for _ in range(20):
                            angle = random.uniform(0, 2 * math.pi)
                            speed = random.uniform(50, 150)
                            vel = (math.cos(angle) * speed, math.sin(angle) * speed)
                            particles.append(Particle(box.x, box.y, GOLD, vel, 0.8, 4))
        
        # Check banana collision
        for banana in bananas:
            if banana.active and not self.star_active:
                dist = math.hypot(self.x - banana.x, self.y - banana.y)
                if dist < 20:
                    banana.active = False
                    self.spin_out()
                    print(f"Kart {self.color} hit a banana!")
                    # Particle effect
                    for _ in range(15):
                        angle = random.uniform(0, 2 * math.pi)
                        speed = random.uniform(30, 100)
                        vel = (math.cos(angle) * speed, math.sin(angle) * speed)
                        particles.append(Particle(banana.x, banana.y, YELLOW, vel, 0.6, 3))
        
        # Check shell collision
        for shell in shells:
            if shell.active and not self.star_active:
                dist = math.hypot(self.x - shell.x, self.y - shell.y)
                if dist < 20:
                    shell.active = False
                    self.spin_out()
                    print(f"Kart {self.color} hit by shell!")
                    # Explosion particles
                    for _ in range(25):
                        angle = random.uniform(0, 2 * math.pi)
                        speed = random.uniform(50, 200)
                        vel = (math.cos(angle) * speed, math.sin(angle) * speed)
                        particles.append(Particle(shell.x, shell.y, RED, vel, 1.0, 5))
        
        # Check checkpoints
        track.check_checkpoint(self)
        
        # AI stuck detection
        if not self.is_player:
            dist_moved = math.hypot(self.x - self.ai_last_x, self.y - self.ai_last_y)
            if dist_moved < 10 * dt:
                self.ai_stuck_timer += dt
                if self.ai_stuck_timer > 2.0:
                    # Unstuck: reverse and turn
                    self.speed = -100
                    self.angle += 45
                    self.ai_stuck_timer = 0
                    print(f"AI Kart {self.color} unsticking")
            else:
                self.ai_stuck_timer = 0
            
            self.ai_last_x = self.x
            self.ai_last_y = self.y
    
    def get_player_input(self) -> Tuple[bool, bool, bool, bool]:
        """Get player keyboard input"""
        keys = pygame.key.get_pressed()
        accelerate = keys[pygame.K_UP]
        brake = keys[pygame.K_DOWN]
        turn_left = keys[pygame.K_LEFT]
        turn_right = keys[pygame.K_RIGHT]
        return accelerate, brake, turn_left, turn_right
    
    def get_ai_input(self, track, karts: List['Kart'], 
                     bananas: List[Banana]) -> Tuple[bool, bool, bool, bool]:
        """AI navigation logic"""
        # Get target waypoint
        if self.ai_target_index >= len(track.ai_waypoints):
            self.ai_target_index = 0
        
        target_x, target_y = track.ai_waypoints[self.ai_target_index]
        
        # Check if reached waypoint
        dist_to_target = math.hypot(target_x - self.x, target_y - self.y)
        if dist_to_target < 50:
            self.ai_target_index = (self.ai_target_index + 1) % len(track.ai_waypoints)
            target_x, target_y = track.ai_waypoints[self.ai_target_index]
        
        # Avoid bananas
        for banana in bananas:
            if banana.active:
                dist_to_banana = math.hypot(banana.x - self.x, banana.y - self.y)
                if dist_to_banana < 80:
                    # Steer away from banana
                    angle_to_banana = math.degrees(math.atan2(banana.y - self.y, 
                                                              banana.x - self.x))
                    angle_diff = (angle_to_banana - self.angle + 180) % 360 - 180
                    if abs(angle_diff) < 45:
                        target_x += (self.x - banana.x) * 2
                        target_y += (self.y - banana.y) * 2
        
        # Calculate angle to target
        angle_to_target = math.degrees(math.atan2(target_y - self.y, target_x - self.x))
        angle_diff = (angle_to_target - self.angle + 180) % 360 - 180
        
        # Determine turning
        turn_threshold = 10 * self.ai_difficulty
        turn_left = angle_diff < -turn_threshold
        turn_right = angle_diff > turn_threshold
        
        # Accelerate unless need to brake for turn
        accelerate = abs(angle_diff) < 45 or abs(self.speed) < 100
        brake = abs(angle_diff) > 60 and abs(self.speed) > 150
        
        # Use power-ups intelligently
        if self.power_up and not self.power_up.active:
            if self.power_up.type == PowerUpType.MUSHROOM:
                # Use on straightaways
                if abs(angle_diff) < 20 and track.is_on_road(self.x, self.y):
                    self.use_power_up(karts, [], [])
            elif self.power_up.type == PowerUpType.SHELL:
                # Use if kart ahead
                for kart in karts:
                    if kart != self and not kart.finished:
                        if kart.position < self.position:
                            dist = math.hypot(kart.x - self.x, kart.y - self.y)
                            if dist < 200:
                                self.use_power_up(karts, [], [])
                                break
            elif self.power_up.type == PowerUpType.BANANA:
                # Drop randomly
                if random.random() < 0.01:
                    self.use_power_up(karts, [], [])
        
        return accelerate, brake, turn_left, turn_right
    
    def use_power_up(self, karts: List['Kart'], bananas: List[Banana], 
                     shells: List[Shell]):
        """Activate current power-up"""
        if not self.power_up:
            return
        
        print(f"Kart {self.color} using {self.power_up.type.name}")
        
        if self.power_up.type == PowerUpType.MUSHROOM:
            self.power_up.active = True
            self.power_up.duration = 1.0
        
        elif self.power_up.type == PowerUpType.SHELL:
            # Fire shell forward
            rad_angle = math.radians(self.angle)
            shell_x = self.x + math.cos(rad_angle) * 30
            shell_y = self.y + math.sin(rad_angle) * 30
            shells.append(Shell(shell_x, shell_y, rad_angle))
            self.power_up = None
        
        elif self.power_up.type == PowerUpType.BANANA:
            # Drop banana behind
            rad_angle = math.radians(self.angle + 180)
            banana_x = self.x + math.cos(rad_angle) * 25
            banana_y = self.y + math.sin(rad_angle) * 25
            bananas.append(Banana(banana_x, banana_y))
            self.power_up = None
        
        elif self.power_up.type == PowerUpType.LIGHTNING:
            # Stun all other karts
            for kart in karts:
                if kart != self and not kart.star_active:
                    kart.stunned = True
                    kart.stun_timer = 2.0
            self.power_up = None
        
        elif self.power_up.type == PowerUpType.STAR:
            self.star_active = True
            self.star_timer = 5.0
            self.power_up = None
    
    def spin_out(self):
        """Kart spins out from collision"""
        self.stunned = True
        self.stun_timer = 1.5
        self.speed *= 0.3
        self.angle += random.uniform(-180, 180)
    
    def draw(self, screen: pygame.Surface, camera_x: float, camera_y: float):
        """Draw kart with details"""
        screen_x = int(self.x - camera_x)
        screen_y = int(self.y - camera_y)
        
        # Star effect
        if self.star_active:
            for i in range(5):
                angle = (pygame.time.get_ticks() / 200 + i * 72) % 360
                offset_x = math.cos(math.radians(angle)) * 25
                offset_y = math.sin(math.radians(angle)) * 25
                pygame.draw.circle(screen, GOLD, 
                                 (int(screen_x + offset_x), int(screen_y + offset_y)), 4)
        
        # Shadow
        shadow_offset = 3
        pygame.draw.ellipse(screen, (0, 0, 0, 100),
                          (screen_x - 12 + shadow_offset, screen_y - 8 + shadow_offset, 24, 16))
        
        # Kart body
        kart_surface = pygame.Surface((30, 20), pygame.SRCALPHA)
        
        # Main body
        body_color = self.color if not self.stunned else GRAY
        pygame.draw.ellipse(kart_surface, body_color, (5, 2, 20, 16))
        pygame.draw.ellipse(kart_surface, tuple(max(0, c - 50) for c in body_color), 
                          (5, 2, 20, 16), 2)
        
        # Wheels
        wheel_color = BLACK if not self.stunned else DARK_GRAY
        pygame.draw.circle(kart_surface, wheel_color, (8, 5), 4)
        pygame.draw.circle(kart_surface, wheel_color, (22, 5), 4)
        pygame.draw.circle(kart_surface, wheel_color, (8, 15), 4)
        pygame.draw.circle(kart_surface, wheel_color, (22, 15), 4)
        
        # Driver helmet
        helmet_color = tuple(min(255, c + 50) for c in body_color)
        pygame.draw.circle(kart_surface, helmet_color, (15, 10), 5)
        
        # Rotate kart
        rotated_kart = pygame.transform.rotate(kart_surface, -self.angle)
        rotated_rect = rotated_kart.get_rect(center=(screen_x, screen_y))
        screen.blit(rotated_kart, rotated_rect)
        
        # Stun stars
        if self.stunned:
            for i in range(3):
                angle = (pygame.time.get_ticks() / 100 + i * 120) % 360
                offset_x = math.cos(math.radians(angle)) * 20
                offset_y = math.sin(math.radians(angle)) * 20 - 15
                pygame.draw.circle(screen, YELLOW, 
                                 (int(screen_x + offset_x), int(screen_y + offset_y)), 3)

class Track:
    """Race track with checkpoints"""
    def __init__(self):
        # Track dimensions
        self.width = 2400
        self.height = 1800
        
        # Define track path (road segments)
        self.road_segments = [
            # Outer oval
            (300, 300, 2100, 300, 200),  # Top straight
            (2100, 300, 2100, 1500, 200),  # Right straight
            (2100, 1500, 300, 1500, 200),  # Bottom straight
            (300, 1500, 300, 300, 200),  # Left straight
        ]
        
        # Inner curves for visual detail
        self.inner_curves = [
            (300, 300, 150),  # Top-left
            (2100, 300, 150),  # Top-right
            (2100, 1500, 150),  # Bottom-right
            (300, 1500, 150),  # Bottom-left
        ]
        
        # Checkpoints (x, y, width, height, angle)
        self.checkpoints = [
            (1200, 250, 150, 50, 0),  # Start/Finish
            (2050, 900, 50, 150, 90),  # Right side
            (1200, 1450, 150, 50, 0),  # Bottom
            (350, 900, 50, 150, 90),  # Left side
        ]
        
        # AI waypoints for navigation
        self.ai_waypoints = [
            (1200, 400),
            (1900, 400),
            (2000, 900),
            (1900, 1400),
            (1200, 1400),
            (500, 1400),
            (400, 900),
            (500, 400),
        ]
        
        # Item box positions
        self.item_box_positions = [
            (800, 400), (1600, 400),
            (2000, 700), (2000, 1100),
            (1600, 1400), (800, 1400),
            (400, 1100), (400, 700),
        ]
        
        # Starting positions
        self.start_positions = [
            (1150, 350, 0),
            (1250, 350, 0),
            (1150, 400, 0),
            (1250, 400, 0),
            (1150, 450, 0),
            (1250, 450, 0),
            (1150, 500, 0),
            (1250, 500, 0),
        ]
    
    def is_on_road(self, x: float, y: float) -> bool:
        """Check if position is on road"""
        for seg in self.road_segments:
            x1, y1, x2, y2, width = seg
            # Distance to line segment
            dx = x2 - x1
            dy = y2 - y1
            if dx == 0 and dy == 0:
                dist = math.hypot(x - x1, y - y1)
            else:
                t = max(0, min(1, ((x - x1) * dx + (y - y1) * dy) / (dx * dx + dy * dy)))
                proj_x = x1 + t * dx
                proj_y = y1 + t * dy
                dist = math.hypot(x - proj_x, y - proj_y)
            
            if dist <= width / 2:
                return True
        
        # Check curves
        for cx, cy, radius in self.inner_curves:
            dist = math.hypot(x - cx, y - cy)
            if abs(dist - radius) < 100:
                return True
        
        return False
    
    def check_checkpoint(self, kart: Kart):
        """Check if kart passed through checkpoint"""
        if kart.finished:
            return
        
        next_checkpoint = self.checkpoints[kart.checkpoint_index]
        cx, cy, cw, ch, angle = next_checkpoint
        
        # Check if kart is in checkpoint area
        if (cx - cw/2 <= kart.x <= cx + cw/2 and 
            cy - ch/2 <= kart.y <= cy + ch/2):
            
            kart.checkpoint_index += 1
            
            # Check if completed lap
            if kart.checkpoint_index >= len(self.checkpoints):
                kart.checkpoint_index = 0
                kart.lap += 1
                print(f"Kart {kart.color} completed lap {kart.lap}")
                
                # Check if finished race (3 laps)
                if kart.lap >= 3:
                    kart.finished = True
                    kart.finish_time = pygame.time.get_ticks() / 1000
                    print(f"Kart {kart.color} finished in {kart.finish_time:.2f}s")
    
    def draw(self, screen: pygame.Surface, camera_x: float, camera_y: float):
        """Draw track with details"""
        # Draw grass background
        for y in range(0, self.height, 50):
            for x in range(0, self.width, 50):
                screen_x = int(x - camera_x)
                screen_y = int(y - camera_y)
                if -50 <= screen_x <= SCREEN_WIDTH and -50 <= screen_y <= SCREEN_HEIGHT:
                    color = DARK_GREEN if (x // 50 + y // 50) % 2 == 0 else GREEN
                    pygame.draw.rect(screen, color, (screen_x, screen_y, 50, 50))
        
        # Draw road segments
        for seg in self.road_segments:
            x1, y1, x2, y2, width = seg
            screen_x1 = int(x1 - camera_x)
            screen_y1 = int(y1 - camera_y)
            screen_x2 = int(x2 - camera_x)
            screen_y2 = int(y2 - camera_y)
            
            # Road
            pygame.draw.line(screen, DARK_GRAY, (screen_x1, screen_y1), 
                           (screen_x2, screen_y2), width)
            # Road edges
            pygame.draw.line(screen, WHITE, (screen_x1, screen_y1), 
                           (screen_x2, screen_y2), width + 10)
            pygame.draw.line(screen, DARK_GRAY, (screen_x1, screen_y1), 
                           (screen_x2, screen_y2), width)
            # Center line
            pygame.draw.line(screen, YELLOW, (screen_x1, screen_y1), 
                           (screen_x2, screen_y2), 3)
        
        # Draw curves
        for cx, cy, radius in self.inner_curves:
            screen_cx = int(cx - camera_x)
            screen_cy = int(cy - camera_y)
            pygame.draw.circle(screen, WHITE, (screen_cx, screen_cy), radius + 105, 10)
            pygame.draw.circle(screen, DARK_GRAY, (screen_cx, screen_cy), radius + 100, 200)
            pygame.draw.circle(screen, DARK_GREEN, (screen_cx, screen_cy), radius - 100)
        
        # Draw checkpoints (debug)
        for i, checkpoint in enumerate(self.checkpoints):
            cx, cy, cw, ch, angle = checkpoint
            screen_cx = int(cx - camera_x)
            screen_cy = int(cy - camera_y)
            
            if i == 0:
                # Start/finish line
                pygame.draw.rect(screen, WHITE, 
                               (screen_cx - cw//2, screen_cy - ch//2, cw, ch))
                # Checkered pattern
                for j in range(10):
                    color = BLACK if j % 2 == 0 else WHITE
                    pygame.draw.rect(screen, color,
                                   (screen_cx - cw//2 + j * (cw//10), 
                                    screen_cy - ch//2, cw//10, ch//2))
                    color = WHITE if j % 2 == 0 else BLACK
                    pygame.draw.rect(screen, color,
                                   (screen_cx - cw//2 + j * (cw//10), 
                                    screen_cy, cw//10, ch//2))

class MiniMap:
    """Mini-map display"""
    def __init__(self, x: int, y: int, width: int, height: int):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.scale = 0.1
    
    def draw(self, screen: pygame.Surface, track: Track, karts: List[Kart], 
             item_boxes: List[ItemBox]):
        """Draw mini-map with real-time updates"""
        # Background
        pygame.draw.rect(screen, (0, 0, 0, 180), (self.x, self.y, self.width, self.height))
        pygame.draw.rect(screen, WHITE, (self.x, self.y, self.width, self.height), 2)
        
        # Draw track outline
        for seg in track.road_segments:
            x1, y1, x2, y2, width = seg
            map_x1 = self.x + int(x1 * self.scale)
            map_y1 = self.y + int(y1 * self.scale)
            map_x2 = self.x + int(x2 * self.scale)
            map_y2 = self.y + int(y2 * self.scale)
            pygame.draw.line(screen, GRAY, (map_x1, map_y1), (map_x2, map_y2), 
                           max(2, int(width * self.scale)))
        
        # Draw item boxes
        for box in item_boxes:
            if box.active:
                map_x = self.x + int(box.x * self.scale)
                map_y = self.y + int(box.y * self.scale)
                pygame.draw.circle(screen, GOLD, (map_x, map_y), 2)
        
        # Draw karts
        for kart in karts:
            map_x = self.x + int(kart.x * self.scale)
            map_y = self.y + int(kart.y * self.scale)
            
            if kart.is_player:
                pygame.draw.circle(screen, WHITE, (map_x, map_y), 4)
                pygame.draw.circle(screen, kart.color, (map_x, map_y), 3)
            else:
                pygame.draw.circle(screen, kart.color, (map_x, map_y), 3)

class Game:
    """Main game class"""
    def __init__(self):
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Kart Racing Championship")
        self.clock = pygame.time.Clock()
        self.running = True
        
        # Game state
        self.track = Track()
        self.karts: List[Kart] = []
        self.item_boxes: List[ItemBox] = []
        self.bananas: List[Banana] = []
        self.shells: List[Shell] = []
        self.particles: List[Particle] = []
        
        # Camera
        self.camera_x = 0
        self.camera_y = 0
        
        # UI
        self.mini_map = MiniMap(SCREEN_WIDTH - 210, 10, 200, 150)
        self.font = pygame.font.Font(None, 36)
        self.small_font = pygame.font.Font(None, 24)
        
        # Race state
        self.race_started = False
        self.countdown = 3
        self.countdown_timer = 0
        self.race_time = 0
        
        self.setup_race()
    
    def setup_race(self):
        """Initialize race"""
        print("=== Setting up race ===")
        
        # Create karts
        num_ai_karts = 7
        for i in range(num_ai_karts + 1):
            start_x, start_y, start_angle = self.track.start_positions[i]
            color = KART_COLORS[i]
            is_player = (i == 0)
            ai_difficulty = random.uniform(0.8, 1.2)
            
            kart = Kart(start_x, start_y, color, is_player, ai_difficulty)
            kart.angle = start_angle
            self.karts.append(kart)
            
            print(f"Created {'PLAYER' if is_player else 'AI'} kart at ({start_x}, {start_y})")
        
        # Create item boxes
        for pos in self.track.item_box_positions:
            self.item_boxes.append(ItemBox(pos[0], pos[1]))
        
        print(f"Created {len(self.item_boxes)} item boxes")
        print("=== Race setup complete ===\n")
    
    def update_camera(self):
        """Update camera to follow player"""
        player = self.karts[0]
        
        # Smooth camera following
        target_x = player.x - SCREEN_WIDTH // 2
        target_y = player.y - SCREEN_HEIGHT // 2
        
        self.camera_x += (target_x - self.camera_x) * 0.1
        self.camera_y += (target_y - self.camera_y) * 0.1
        
        # Clamp camera to track bounds
        self.camera_x = max(0, min(self.camera_x, self.track.width - SCREEN_WIDTH))
        self.camera_y = max(0, min(self.camera_y, self.track.height - SCREEN_HEIGHT))
    
    def update_positions(self):
        """Calculate race positions"""
        # Sort by lap, then checkpoint, then distance to next checkpoint
        def get_progress(kart):
            if kart.finished:
                return 1000000 - kart.finish_time
            
            next_checkpoint = self.track.checkpoints[kart.checkpoint_index]
            cx, cy = next_checkpoint[0], next_checkpoint[1]
            dist_to_checkpoint = math.hypot(kart.x - cx, kart.y - cy)
            
            return kart.lap * 10000 + kart.checkpoint_index * 1000 - dist_to_checkpoint
        
        sorted_karts = sorted(self.karts, key=get_progress, reverse=True)
        
        for i, kart in enumerate(sorted_karts):
            kart.position = i + 1
    
    def handle_events(self):
        """Handle input events"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False
                
                # Use power-up
                if event.key == pygame.K_SPACE:
                    player = self.karts[0]
                    if player.power_up:
                        player.use_power_up(self.karts, self.bananas, self.shells)
    
    def update(self, dt: float):
        """Update game state"""
        # Countdown before race starts
        if not self.race_started:
            self.countdown_timer += dt
            if self.countdown_timer >= 1.0:
                self.countdown_timer = 0
                self.countdown -= 1
                if self.countdown < 0:
                    self.race_started = True
                    print("=== RACE STARTED ===")
            return
        
        # Update race time
        self.race_time += dt
        
        # Update all karts
        for kart in self.karts:
            kart.update(dt, self.track, self.karts, self.item_boxes, 
                       self.bananas, self.shells, self.particles)
        
        # Update positions
        self.update_positions()
        
        # Update item boxes
        for box in self.item_boxes:
            box.update(dt)
        
        # Update shells
        for shell in self.shells:
            shell.update(dt)
        self.shells = [s for s in self.shells if s.active]
        
        # Update particles
        for particle in self.particles:
            particle.update(dt)
        self.particles = [p for p in self.particles if p.is_alive()]
        
        # Update camera
        self.update_camera()
        
        # Check if race is over
        if self.karts[0].finished:
            print(f"\n=== RACE FINISHED ===")
            print(f"Your position: {self.karts[0].position}")
            print(f"Your time: {self.karts[0].finish_time:.2f}s")
    
    def draw(self):
        """Draw everything"""
        self.screen.fill(GREEN)
        
        # Draw track
        self.track.draw(self.screen, self.camera_x, self.camera_y)
        
        # Draw bananas
        for banana in self.bananas:
            banana.draw(self.screen, self.camera_x, self.camera_y)
        
        # Draw item boxes
        for box in self.item_boxes:
            box.draw(self.screen, self.camera_x, self.camera_y)
        
        # Draw shells
        for shell in self.shells:
            shell.draw(self.screen, self.camera_x, self.camera_y)
        
        # Draw particles
        for particle in self.particles:
            particle.draw(self.screen, self.camera_x, self.camera_y)
        
        # Draw karts (sorted by y position for proper overlap)
        sorted_karts = sorted(self.karts, key=lambda k: k.y)
        for kart in sorted_karts:
            kart.draw(self.screen, self.camera_x, self.camera_y)
        
        # Draw UI
        self.draw_ui()
        
        # Draw mini-map
        self.mini_map.draw(self.screen, self.track, self.karts, self.item_boxes)
        
        pygame.display.flip()
    
    def draw_ui(self):
        """Draw user interface"""
        player = self.karts[0]
        
        # Countdown
        if not self.race_started:
            if self.countdown > 0:
                text = self.font.render(str(self.countdown), True, RED)
            else:
                text = self.font.render("GO!", True, GREEN)
            text_rect = text.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2))
            
            # Shadow
            shadow = self.font.render(text.get_at((0, 0)) and str(self.countdown) or "GO!", 
                                     True, BLACK)
            shadow_rect = shadow.get_rect(center=(SCREEN_WIDTH // 2 + 2, 
                                                  SCREEN_HEIGHT // 2 + 2))
            self.screen.blit(shadow, shadow_rect)
            self.screen.blit(text, text_rect)
            return
        
        # Position
        position_text = self.font.render(f"Position: {player.position}/8", True, WHITE)
        self.screen.blit(position_text, (10, 10))
        
        # Lap
        lap_text = self.font.render(f"Lap: {player.lap + 1}/3", True, WHITE)
        self.screen.blit(lap_text, (10, 50))
        
        # Speed
        speed_text = self.small_font.render(f"Speed: {abs(player.speed):.0f}", True, WHITE)
        self.screen.blit(speed_text, (10, 90))
        
        # Power-up
        if player.power_up:
            powerup_name = player.power_up.type.name
            powerup_text = self.small_font.render(f"Item: {powerup_name}", True, GOLD)
            self.screen.blit(powerup_text, (10, 120))
            
            # Draw power-up icon
            icon_x, icon_y = 150, 125
            if player.power_up.type == PowerUpType.MUSHROOM:
                pygame.draw.circle(self.screen, RED, (icon_x, icon_y), 10)
                pygame.draw.circle(self.screen, WHITE, (icon_x - 3, icon_y - 3), 3)
                pygame.draw.circle(self.screen, WHITE, (icon_x + 3, icon_y + 3), 3)
            elif player.power_up.type == PowerUpType.SHELL:
                pygame.draw.circle(self.screen, RED, (icon_x, icon_y), 10)
            elif player.power_up.type == PowerUpType.BANANA:
                pygame.draw.ellipse(self.screen, YELLOW, (icon_x - 8, icon_y - 5, 16, 10))
            elif player.power_up.type == PowerUpType.LIGHTNING:
                pygame.draw.polygon(self.screen, YELLOW, 
                                  [(icon_x, icon_y - 10), (icon_x - 5, icon_y), 
                                   (icon_x, icon_y), (icon_x - 3, icon_y + 10)])
            elif player.power_up.type == PowerUpType.STAR:
                for i in range(5):
                    angle = math.radians(i * 72 - 90)
                    x = icon_x + math.cos(angle) * 10
                    y = icon_y + math.sin(angle) * 10
                    pygame.draw.circle(self.screen, GOLD, (int(x), int(y)), 3)
        
        # Star timer
        if player.star_active:
            star_text = self.small_font.render(f"STAR: {player.star_timer:.1f}s", 
                                              True, GOLD)
            self.screen.blit(star_text, (10, 150))
        
        # Race time
        time_text = self.small_font.render(f"Time: {self.race_time:.1f}s", True, WHITE)
        self.screen.blit(time_text, (SCREEN_WIDTH - 150, 10))
        
        # Finish message
        if player.finished:
            finish_text = self.font.render("FINISHED!", True, GOLD)
            finish_rect = finish_text.get_rect(center=(SCREEN_WIDTH // 2, 100))
            
            # Background
            pygame.draw.rect(self.screen, (0, 0, 0, 180), 
                           finish_rect.inflate(20, 10))
            self.screen.blit(finish_text, finish_rect)
            
            position_text = self.small_font.render(
                f"Final Position: {player.position}", True, WHITE)
            position_rect = position_text.get_rect(center=(SCREEN_WIDTH // 2, 140))
            self.screen.blit(position_text, position_rect)
    
    def run(self):
        """Main game loop"""
        print("=== GAME STARTED ===")
        print("Controls:")
        print("  Arrow Keys - Accelerate, Brake, Turn")
        print("  SPACE - Use Power-up")
        print("  ESC - Quit")
        print()
        
        while self.running:
            dt = self.clock.tick(FPS) / 1000.0
            
            self.handle_events()
            self.update(dt)
            self.draw()
        
        print("\n=== GAME ENDED ===")
        pygame.quit()

if __name__ == "__main__":
    game = Game()
    game.run()