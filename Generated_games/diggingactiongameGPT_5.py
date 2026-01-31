import math
import random
import sys
import time
from collections import deque
from enum import Enum, auto

import pygame
import pygame.freetype

# ------------- CONFIG / CONSTANTS -------------
SCREEN_W, SCREEN_H = 960, 720
FPS = 60
TILE = 32
GRID_W = SCREEN_W // TILE
GRID_H = SCREEN_H // TILE

DIRT_COLOR = (110, 70, 35)
TUNNEL_COLOR = (18, 10, 8)
WALL_COLOR = (30, 20, 15)
CLAMPED_FPS = 120

PLAYER_SPEED = 140.0
MONSTER_SPEED = 85.0
MONSTER_SPEED_INC = 5.0
APPLE_FALL_SPEED = 220.0
APPLE_ROLL_SPEED = 150.0
POWERBALL_SPEED = 400.0
POWERBALL_MAX_BOUNCES = 5
POWERBALL_RETURN_DELAY = 0.8

PARTICLE_COUNT_DIG = 12
PARTICLE_COUNT_CRUSH = 24

SCORE_CHERRY = 50
SCORE_MONSTER = 100
SCORE_ALPHA_BONUS = 100
SCORE_APPLE_CRUSH_CHAIN = 50

EXTRA_LETTERS = "EXTRA"
LIGHT_DARK_ALPHA = 220
LIGHT_PLAYER_RADIUS = 160
LIGHT_SOFT_STEPS = 5

# ------------- UTILS -------------
vec2 = pygame.math.Vector2

def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def tile_to_world(ix, iy):
    return ix * TILE + TILE // 2, iy * TILE + TILE // 2

def world_to_tile(px, py):
    return int(px // TILE), int(py // TILE)

def in_bounds(ix, iy):
    return 0 <= ix < GRID_W and 0 <= iy < GRID_H

def manhattan(a, b):
    return abs(a[0]-b[0]) + abs(a[1]-b[1])

# ------------- TILE MAP -------------
class TileType(Enum):
    WALL = 0
    DIRT = 1
    TUNNEL = 2

class Grid:
    def __init__(self, w, h):
        self.w, self.h = w, h
        self.tiles = [[TileType.DIRT for _ in range(h)] for _ in range(w)]
        for x in range(w):
            for y in range(h):
                if x == 0 or y == 0 or x == w-1 or y == h-1:
                    self.tiles[x][y] = TileType.WALL
        self.cherries = set()  # set of (x,y) tile positions
        self.occupied = {}     # maps (x,y) -> entity id (apples occupy)
    def set_tile(self, x, y, t):
        if in_bounds(x,y):
            self.tiles[x][y] = t
    def get_tile(self, x, y):
        if not in_bounds(x,y):
            return TileType.WALL
        return self.tiles[x][y]
    def is_passable(self, x, y):
        return self.get_tile(x,y) == TileType.TUNNEL
    def is_solid(self, x, y):
        tt = self.get_tile(x,y)
        return tt == TileType.WALL or tt == TileType.DIRT

# ------------- PARTICLES -------------
class Particle:
    def __init__(self, pos, vel, color, life, radius):
        self.pos = vec2(pos)
        self.vel = vec2(vel)
        self.color = color
        self.life = life
        self.max_life = life
        self.radius = radius
        self.grav = 300.0
    def update(self, dt):
        self.life -= dt
        self.vel.y += self.grav * dt
        self.pos += self.vel * dt
    def draw(self, surf):
        if self.life <= 0:
            return
        alpha = int(255 * (self.life / self.max_life))
        col = (*self.color, alpha)
        pygame.draw.circle(surf, col, self.pos, max(1, int(self.radius)))
    def alive(self):
        return self.life > 0.0

# ------------- ENTITIES -------------
_next_ent_id = 1
def _alloc_id():
    global _next_ent_id
    i = _next_ent_id
    _next_ent_id += 1
    return i

class Entity:
    def __init__(self, pos):
        self.id = _alloc_id()
        self.pos = vec2(pos)
        self.dead = False
    def update(self, dt, game):
        pass
    def draw(self, surf, game):
        pass

# ------------- APPLE -------------
class Apple(Entity):
    def __init__(self, tile_pos):
        cx, cy = tile_to_world(*tile_pos)
        super().__init__((cx, cy))
        self.grid = vec2(tile_pos)
        self.falling = False
        self.vy = 0.0
        self.roll_dir = 0  # -1 left, 1 right, 0 none
        self.squish = 0.0
        self.rot = random.uniform(0, math.tau)
        self.shadow_bob = random.uniform(0, math.tau)
        self.chain_crushes = 0
    def occupy(self, game):
        game.grid.occupied[(int(self.grid.x), int(self.grid.y))] = self.id
    def vacate(self, game):
        game.grid.occupied.pop((int(self.grid.x), int(self.grid.y)), None)
    def can_move_to(self, game, tx, ty):
        if not in_bounds(tx, ty): return False
        if game.grid.is_passable(tx, ty) and (tx, ty) not in game.grid.occupied:
            return True
        return False
    def push(self, game, dirx):
        # attempt to move one tile horizontally if free
        tx = int(self.grid.x + dirx)
        ty = int(self.grid.y)
        if self.can_move_to(game, tx, ty):
            self.vacate(game)
            self.grid.x = tx
            self.pos.x = tile_to_world(tx, ty)[0]
            self.occupy(game)
            print(f"[APPLE] Pushed to {(tx, ty)}")
            return True
        return False
    def do_gravity(self, game, dt):
        tx, ty = int(self.grid.x), int(self.grid.y)
        below = (tx, ty+1)
        if self.can_move_to(game, *below):
            self.falling = True
            self.vy += 900.0 * dt
            self.vy = clamp(self.vy, -1000, 1000)
            dy = self.vy * dt
            self.pos.y += dy
            self.squish = clamp(self.vy / 500.0, 0, 0.35)
            # crush detection while falling
            self.crush_check(game)
            # if crosses into next tile center, advance grid cell
            cx, cy = tile_to_world(tx, ty+1)
            if self.pos.y >= cy:
                self.vacate(game)
                self.grid.y += 1
                self.occupy(game)
                print(f"[APPLE] Fell to {(tx, ty+1)}")
        else:
            # maybe roll if supported diagonally
            self.vy = 0.0
            self.squish = 0.0
            self.falling = False
            # roll rules: if blocked below, and side+down side are empty, roll
            options = []
            left_free = self.can_move_to(game, tx-1, ty) and self.can_move_to(game, tx-1, ty+1)
            right_free = self.can_move_to(game, tx+1, ty) and self.can_move_to(game, tx+1, ty+1)
            if left_free: options.append(-1)
            if right_free: options.append(1)
            if options:
                # choose roll direction biased by slope of tunnel
                if self.roll_dir == 0: self.roll_dir = random.choice(options)
                elif self.roll_dir not in options: self.roll_dir = random.choice(options)
                # move slowly horizontally
                sign = self.roll_dir
                self.pos.x += sign * APPLE_ROLL_SPEED * dt
                # crush entities while rolling
                self.crush_check(game)
                cx, cy = tile_to_world(int(self.grid.x + sign), ty)
                if (sign > 0 and self.pos.x >= cx) or (sign < 0 and self.pos.x <= cx):
                    self.vacate(game)
                    self.grid.x += sign
                    self.occupy(game)
                    print(f"[APPLE] Rolled to {(int(self.grid.x), int(self.grid.y))}")
            else:
                self.roll_dir = 0
                # snap to center of current tile
                cx, cy = tile_to_world(tx, ty)
                self.pos.x = cx
                self.pos.y = cy
    def crush_check(self, game):
        # if intersects monster or player while falling or rolling
        rect = pygame.Rect(0, 0, TILE*0.8, TILE*0.8)
        rect.center = (self.pos.x, self.pos.y)
        # monsters
        for m in list(game.monsters):
            if m.dead: continue
            if rect.colliderect(m.aabb()):
                m.kill(game, cause="apple")
                self.chain_crushes += 1
                add = SCORE_MONSTER + self.chain_crushes * SCORE_APPLE_CRUSH_CHAIN
                game.add_score(add)
                print(f"[APPLE] Crushed monster {m.id}, chain={self.chain_crushes}, +{add}")
                game.spawn_crush_particles(m.pos)
        # player
        p = game.player
        if not p.invuln and rect.colliderect(p.aabb()):
            print("[APPLE] Crushed player")
            game.player_die()
    def update(self, dt, game):
        self.do_gravity(game, dt)
        self.rot += dt * (0.6 + 0.2 * math.sin(self.pos.y*0.01))
        self.shadow_bob += dt * 3.0
    def draw(self, surf, game):
        # apple body with slight shading and squish
        r = TILE*0.40
        squx = 1.0 + self.squish
        squy = 1.0 - self.squish
        base = vec2(self.pos)
        # shadow
        sh = pygame.Surface((TILE*2, TILE*2), pygame.SRCALPHA)
        sh_alpha = 50 + int(20*math.sin(self.shadow_bob))
        pygame.draw.ellipse(sh, (0,0,0,sh_alpha), (TILE*0.5, TILE*1.2, TILE*0.9, TILE*0.3))
        surf.blit(sh, (base.x - TILE, base.y - TILE))
        # body
        apple_surf = pygame.Surface((int(r*2)+6, int(r*2)+6), pygame.SRCALPHA)
        center = (apple_surf.get_width()//2, apple_surf.get_height()//2)
        # gradient-like fill
        for i in range(6, 0, -1):
            frac = i/6.0
            col = (220, 20 + int(60*frac), 20 + int(20*frac), 255)
            rx = int(r*squx*frac)
            ry = int(r*squy*frac)
            pygame.draw.ellipse(apple_surf, col, (center[0]-rx, center[1]-ry, rx*2, ry*2))
        # highlight
        pygame.draw.circle(apple_surf, (255,255,255,80), (center[0]-int(r*0.35), center[1]-int(r*0.35)), int(r*0.25))
        # stem
        pygame.draw.line(apple_surf, (60,40,10,255), (center[0], center[1]-int(r)), (center[0]+4, center[1]-int(r*1.3)), 5)
        # rotate sprite for subtle motion
        rotated = pygame.transform.rotozoom(apple_surf, math.degrees(self.rot)*0.2, 1.0)
        rect = rotated.get_rect(center=(base.x, base.y))
        surf.blit(rotated, rect)

# ------------- LETTER PICKUP -------------
class LetterPickup(Entity):
    def __init__(self, pos, ch):
        super().__init__(pos)
        self.ch = ch
        self.life = 8.0
        self.vel = vec2(random.uniform(-40,40), random.uniform(-140,-80))
        self.gravity = 350.0
        self.bounce = 0.35
        self.grounded = False
    def update(self, dt, game):
        self.life -= dt
        if self.life <= 0:
            self.dead = True
            return
        if not self.grounded:
            self.vel.y += self.gravity * dt
            self.pos += self.vel * dt
            # ground at tunnel floor
            ix, iy = world_to_tile(self.pos.x, self.pos.y)
            if game.grid.is_passable(ix, iy+1):
                pass
            else:
                # bounce on ground
                self.pos.y = tile_to_world(ix, iy)[1]
                self.vel.y *= -self.bounce
                if abs(self.vel.y) < 20:
                    self.vel.y = 0
                    self.grounded = True
        # collect with player
        if self.pos.distance_to(game.player.pos) < TILE*0.6:
            game.collect_letter(self.ch)
            self.dead = True
            print(f"[LETTER] Collected {self.ch}")
    def draw(self, surf, game):
        col = (255, 240, 120)
        sz = 20
        alpha = int(230 if self.life > 1.0 else 230 * self.life)
        pygame.draw.circle(surf, (255,255,0,alpha), self.pos, 16)
        font = game.hud.font_small
        font.render_to(surf, (self.pos.x-8, self.pos.y-12), self.ch, (0,0,0))

# ------------- POWERBALL -------------
class PowerBall(Entity):
    def __init__(self, pos):
        super().__init__(pos)
        self.vel = vec2(0,0)
        self.active = False
        self.returning = False
        self.bounces = 0
        self.timer = 0.0
        self.radius = 10
    def throw(self, origin, direction):
        self.pos = vec2(origin)
        self.vel = direction.normalize() * POWERBALL_SPEED
        self.active = True
        self.returning = False
        self.bounces = 0
        self.timer = 0.0
        print(f"[POWERBALL] Thrown from {tuple(map(int, origin))} dir={direction}")
    def update(self, dt, game):
        if not self.active: return
        self.timer += dt
        # check collisions with monsters
        for m in list(game.monsters):
            if m.dead: continue
            if self.collides_with(m.aabb()):
                m.kill(game, cause="powerball")
                game.add_score(SCORE_MONSTER + (SCORE_ALPHA_BONUS if m.is_alpha else 0))
                game.spawn_crush_particles(m.pos, color=(80,200,255))
                print(f"[POWERBALL] Hit monster {m.id}")
        # bounce on walls/ dirt / apples
        next_pos = self.pos + self.vel * dt
        collided = False
        # tile collision
        ix, iy = world_to_tile(next_pos.x, next_pos.y)
        if not in_bounds(ix,iy) or game.grid.is_solid(ix, iy) or (ix,iy) in game.grid.occupied:
            # reflect on the axis of collision, sample two positions to infer normal
            cx, cy = world_to_tile(self.pos.x + self.vel.x*dt, self.pos.y)
            if not in_bounds(cx,cy) or game.grid.is_solid(cx,cy) or (cx,cy) in game.grid.occupied:
                self.vel.x *= -1
                collided = True
            cy2 = world_to_tile(self.pos.x, self.pos.y + self.vel.y*dt)[1]
            if not in_bounds(ix,cy2) or game.grid.is_solid(ix,cy2) or (ix,cy2) in game.grid.occupied:
                self.vel.y *= -1
                collided = True
        if collided:
            self.bounces += 1
            print(f"[POWERBALL] Bounce #{self.bounces}")
        self.pos += self.vel * dt
        # return logic
        if (self.timer >= POWERBALL_RETURN_DELAY and not self.returning) or self.bounces >= POWERBALL_MAX_BOUNCES:
            self.returning = True
            print("[POWERBALL] Returning...")
        if self.returning:
            to_player = (game.player.pos - self.pos)
            if to_player.length() > 1:
                self.vel = to_player.normalize() * POWERBALL_SPEED
        # catch
        if self.pos.distance_to(game.player.pos) < TILE*0.6:
            self.active = False
            self.returning = False
            print("[POWERBALL] Caught")
    def collides_with(self, rect):
        # circle-rect collision
        circle = pygame.Rect(0,0, self.radius*2, self.radius*2)
        circle.center = (self.pos.x, self.pos.y)
        return rect.colliderect(circle)
    def draw(self, surf, game):
        if not self.active: return
        # glow
        pygame.draw.circle(surf, (30, 160, 255), self.pos, self.radius)
        pygame.draw.circle(surf, (180, 240, 255), self.pos, self.radius*0.6)
        # trail
        for i in range(4):
            p = self.pos - self.vel * (i*0.02)
            pygame.draw.circle(surf, (30, 160, 255, 100), p, max(1, int(self.radius*(1-i*0.18))))

# ------------- MONSTER -------------
class Monster(Entity):
    def __init__(self, tile_pos, is_alpha=False):
        cx, cy = tile_to_world(*tile_pos)
        super().__init__((cx, cy))
        self.grid = vec2(tile_pos)
        self.dir = vec2(0, 1)
        self.speed = MONSTER_SPEED
        self.is_alpha = is_alpha
        self.bob = random.uniform(0, math.tau)
        self.color = (200, 80, 100) if not is_alpha else (200, 200, 80)
        self.next_path_check = 0.0
        self.path = []
        self.radius = TILE*0.36
        self.eye_dir = vec2(1,0)
    def aabb(self):
        r = TILE*0.72
        rect = pygame.Rect(0,0,r,r)
        rect.center = (self.pos.x, self.pos.y)
        return rect
    def pick_next_dir(self, game):
        # BFS path to player across tunnels
        src = (int(self.grid.x), int(self.grid.y))
        dst = world_to_tile(game.player.pos.x, game.player.pos.y)
        if src == dst:
            return
        q = deque([src])
        prev = {src: None}
        while q:
            cur = q.popleft()
            if cur == dst: break
            for d in [(1,0),(-1,0),(0,1),(0,-1)]:
                nx, ny = cur[0]+d[0], cur[1]+d[1]
                if not in_bounds(nx, ny): continue
                if (nx, ny) in prev: continue
                if not game.grid.is_passable(nx, ny): continue
                prev[(nx,ny)] = cur
                q.append((nx,ny))
        if dst not in prev:
            # wander
            choices = []
            for d in [(1,0),(-1,0),(0,1),(0,-1)]:
                nx, ny = src[0]+d[0], src[1]+d[1]
                if game.grid.is_passable(nx, ny):
                    choices.append(vec2(d))
            if choices:
                self.dir = random.choice(choices)
            return
        # reconstruct path
        path = []
        cur = dst
        while cur != src:
            path.append(cur)
            cur = prev[cur]
        path.reverse()
        self.path = path
        if self.path:
            nx, ny = self.path[0]
            self.dir = vec2(nx - src[0], ny - src[1])
    def kill(self, game, cause="unknown"):
        if self.dead: return
        self.dead = True
        # drop letter if alpha
        if self.is_alpha:
            needed = game.letters_needed()
            if needed:
                ch = random.choice(list(needed))
            else:
                ch = random.choice(list(EXTRA_LETTERS))
            lp = LetterPickup(self.pos, ch)
            game.pickups.append(lp)
            print(f"[MONSTER] Alpha killed by {cause}, dropped letter {ch}")
        else:
            print(f"[MONSTER] Killed by {cause}")
    def update(self, dt, game):
        if self.dead: return
        self.next_path_check -= dt
        if self.next_path_check <= 0:
            self.pick_next_dir(game)
            self.next_path_check = 0.35 + random.uniform(-0.1, 0.1)
        # movement along tunnels; snap to centers
        tgt = self.grid + self.dir
        if game.grid.is_passable(int(tgt.x), int(tgt.y)):
            # move towards center of target tile
            cx, cy = tile_to_world(int(tgt.x), int(tgt.y))
            vec = vec2(cx, cy) - self.pos
            dist = vec.length()
            if dist < 1:
                self.grid = vec2(tgt)
            else:
                step = self.speed * dt
                if step >= dist:
                    self.pos = vec2(cx, cy)
                    self.grid = vec2(tgt)
                else:
                    self.pos += vec.normalize() * step
        else:
            # pick a different direction
            self.pick_next_dir(game)
        # player collision
        if self.aabb().colliderect(game.player.aabb()) and not game.player.invuln:
            print("[MONSTER] Hit player")
            game.player_die()
        # eye dir for rendering
        if self.dir.length_squared() > 0:
            self.eye_dir = self.dir.normalize()
        self.bob += dt * 3.0
    def draw(self, surf, game):
        if self.dead: return
        base = vec2(self.pos)
        bob = math.sin(self.bob) * 3.5
        # body
        body_col = self.color
        pygame.draw.circle(surf, body_col, (base.x, base.y-8+bob), self.radius)
        # feet
        for i in range(-2,3,2):
            pygame.draw.circle(surf, (max(0,body_col[0]-30),max(0,body_col[1]-30),max(0,body_col[2]-30)),
                               (base.x + i*10, base.y + 10 + math.sin(self.bob*2+i)*2), 6)
        # eyes
        eye_base = (base.x + self.eye_dir.x*6, base.y-12+bob)
        pygame.draw.circle(surf, (255,255,255), eye_base, 6)
        pupil = (eye_base[0] + self.eye_dir.x*3, eye_base[1] + self.eye_dir.y*3)
        pygame.draw.circle(surf, (0,0,0), pupil, 3)
        # alpha halo
        if self.is_alpha:
            pygame.draw.circle(surf, (255,230,80,60), (base.x, base.y), self.radius+6, 2)

# ------------- PLAYER -------------
class Player(Entity):
    def __init__(self, pos):
        super().__init__(pos)
        self.grid = vec2(world_to_tile(pos[0], pos[1]))
        self.dir = vec2(0,0)
        self.speed = PLAYER_SPEED
        self.throw_cool = 0.0
        self.powerball_ready = True
        self.invuln = False
        self.invuln_time = 0.0
        self.anim_t = 0.0
        self.face = 0.0
        self.pushing = False
    def aabb(self):
        r = TILE*0.65
        rect = pygame.Rect(0,0,r,r)
        rect.center = (self.pos.x, self.pos.y)
        return rect
    def handle_input(self, keys, game, dt):
        self.dir = vec2(0,0)
        if keys[pygame.K_LEFT] or keys[pygame.K_a]: self.dir.x = -1
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]: self.dir.x = 1
        if keys[pygame.K_UP] or keys[pygame.K_w]: self.dir.y = -1
        elif keys[pygame.K_DOWN] or keys[pygame.K_s]: self.dir.y = 1
        # restrict to cardinal dominant axis
        if abs(self.dir.x) > abs(self.dir.y): self.dir.y = 0
        else: self.dir.x = 0
        # throw powerball
        if keys[pygame.K_SPACE] and self.throw_cool <= 0.0 and game.powerball and not game.powerball.active:
            aim = self.dir if self.dir.length_squared()>0 else vec2(1,0)
            game.powerball.throw(self.pos, aim)
            self.powerball_ready = False
            self.throw_cool = 0.2
        elif keys[pygame.K_SPACE] and game.powerball and game.powerball.active and not game.powerball.returning:
            # force return
            game.powerball.returning = True
            print("[POWERBALL] Recall requested")
    def move_and_dig(self, game, dt):
        if self.dir.length_squared() == 0:
            return
        tgt = self.grid + self.dir
        tx, ty = int(tgt.x), int(tgt.y)
        # pushing apple?
        self.pushing = False
        if (tx, ty) in game.grid.occupied:
            # try pushing apple horizontally only
            if self.dir.y == 0:
                apple_id = game.grid.occupied[(tx,ty)]
                apple = next((a for a in game.apples if a.id == apple_id), None)
                if apple and apple.push(game, int(self.dir.x)):
                    # after push, we can step into that tile
                    pass
                self.pushing = True
        tt = game.grid.get_tile(tx, ty)
        if tt == TileType.WALL:
            return
        # Dig dirt
        if tt == TileType.DIRT:
            game.grid.set_tile(tx, ty, TileType.TUNNEL)
            game.spawn_dig_particles((tx,ty))
            print(f"[PLAYER] Dug at {(tx,ty)}")
        # step to target tile
        cx, cy = tile_to_world(tx, ty)
        vec = vec2(cx, cy) - self.pos
        dist = vec.length()
        if dist < 1:
            self.grid = vec2((tx,ty))
        else:
            step = self.speed * dt
            if step >= dist:
                self.pos = vec2(cx, cy)
                self.grid = vec2((tx,ty))
            else:
                self.pos += vec.normalize() * step
        # collect cherry if present
        if (tx, ty) in game.grid.cherries:
            game.grid.cherries.remove((tx, ty))
            game.add_score(SCORE_CHERRY)
            print(f"[CHERRY] Collected at {(tx,ty)} +{SCORE_CHERRY}")
    def update(self, dt, game):
        self.throw_cool -= dt
        keys = pygame.key.get_pressed()
        self.handle_input(keys, game, dt)
        self.move_and_dig(game, dt)
        if game.powerball and not game.powerball.active:
            self.powerball_ready = True
        # invuln timer
        if self.invuln:
            self.invuln_time -= dt
            if self.invuln_time <= 0:
                self.invuln = False
        # animation timer
        self.anim_t += dt * (2.0 if self.dir.length_squared()>0 else 1.0)
        self.face = math.atan2(self.dir.y, self.dir.x) if self.dir.length_squared()>0 else self.face
    def draw(self, surf, game):
        base = vec2(self.pos)
        # body bounce
        bob = math.sin(self.anim_t*8.0) * (3.0 if self.dir.length_squared()>0 else 1.0)
        # draw clown: head, hat, eyes, mouth, nose, body, limbs
        # body
        body_col = (240, 120, 240)
        alt_col = (120, 200, 255)
        pygame.draw.ellipse(surf, alt_col, (base.x - 16, base.y - 8 + bob, 32, 28))
        pygame.draw.ellipse(surf, body_col, (base.x - 18, base.y - 16 + bob, 36, 30))
        # legs
        leg_phase = math.sin(self.anim_t*10)
        pygame.draw.line(surf, (50, 20, 80), (base.x-8, base.y+8+bob), (base.x-8, base.y+16+bob + leg_phase*2), 5)
        pygame.draw.line(surf, (50, 20, 80), (base.x+8, base.y+8+bob), (base.x+8, base.y+16+bob - leg_phase*2), 5)
        # arms
        arm_phase = math.sin(self.anim_t*12)
        pygame.draw.line(surf, (50,20,80), (base.x-14, base.y-4+bob), (base.x-22, base.y-4+bob + arm_phase*3), 4)
        pygame.draw.line(surf, (50,20,80), (base.x+14, base.y-4+bob), (base.x+22, base.y-4+bob - arm_phase*3), 4)
        # head
        head_y = base.y - 28 + bob
        pygame.draw.circle(surf, (255, 230, 200), (base.x, head_y), 14)
        # hat
        pygame.draw.polygon(surf, (40,80,200), [(base.x-12, head_y-6),(base.x+12, head_y-6),(base.x, head_y-22)])
        pygame.draw.circle(surf, (255,255,255), (base.x, head_y-24), 3)
        # eyes
        ex = math.cos(self.face)*3
        ey = math.sin(self.face)*2
        pygame.draw.circle(surf, (255,255,255), (base.x-5, head_y-3), 4)
        pygame.draw.circle(surf, (255,255,255), (base.x+5, head_y-3), 4)
        pygame.draw.circle(surf, (0,0,0), (base.x-5+ex, head_y-3+ey), 2)
        pygame.draw.circle(surf, (0,0,0), (base.x+5+ex, head_y-3+ey), 2)
        # mouth (changes when pushing)
        mood = -2 if self.pushing else 2
        pygame.draw.arc(surf, (150,0,0), (base.x-6, head_y+2, 12, 8), math.pi*0.1, math.pi*0.9, 2)
        # nose
        pygame.draw.circle(surf, (240,40,40), (base.x, head_y), 3)
        # outline blink when invuln
        if self.invuln and int(time.time()*10)%2==0:
            pygame.draw.circle(surf, (255,255,0), (base.x, head_y), 16, 2)

# ------------- HUD -------------
class HUD:
    def __init__(self):
        pygame.freetype.init()
        # Attempt retro fonts; fallback to default
        candidates = ["Press Start 2P", "VT323", "Pixel Operator", "Consolas", "Courier New", None]
        self.font = None
        for name in candidates:
            try:
                self.font = pygame.freetype.SysFont(name, 22)
                self.font_small = pygame.freetype.SysFont(name, 16)
                break
            except Exception:
                continue
        if self.font is None:
            self.font = pygame.freetype.Font(None, 22)
            self.font_small = pygame.freetype.Font(None, 16)
        self.flash_t = 0.0
        self.flash_text = ""
    def update(self, dt):
        self.flash_t = max(0.0, self.flash_t - dt)
    def flash(self, text):
        self.flash_text = text
        self.flash_t = 1.0
        print(f"[HUD] Flash: {text}")
    def draw(self, surf, score, lives, level, letters_collected, cherries_left):
        # top bar
        pygame.draw.rect(surf, (0,0,0,140), (0,0,SCREEN_W, 36))
        self.font.render_to(surf, (12, 8), f"SCORE {score:06d}", (255, 255, 180))
        self.font.render_to(surf, (250, 8), f"LIVES {lives}", (180, 255, 180))
        self.font.render_to(surf, (400, 8), f"LEVEL {level}", (180, 200, 255))
        self.font.render_to(surf, (540, 8), f"CHERRIES {cherries_left}", (255, 180, 180))
        # letters
        s = "EXTRA: "
        for ch in EXTRA_LETTERS:
            col = (255,255,120) if ch in letters_collected else (80,80,50)
            self.font.render_to(surf, (740 + EXTRA_LETTERS.index(ch)*20, 8), ch, col)
        # center message
        if self.flash_t > 0:
            alpha = int(255 * min(1, self.flash_t))
            text_surf, rect = self.font.render(self.flash_text, (255,255,255))
            rect.center = (SCREEN_W//2, 60)
            surf.blit(text_surf, rect)

# ------------- LEVEL GENERATION -------------
class Level:
    def __init__(self, idx):
        self.idx = idx
        self.monster_count = 6 + (idx-1)*1
        self.alpha_count = 2 + (idx//2)
        self.apple_count = 28 + (idx-1)*4
        self.cherry_clusters = 12
        self.seed = random.randint(0, 999999)
    def build(self, game):
        random.seed(self.seed)
        g = game.grid
        # reset tiles
        for x in range(GRID_W):
            for y in range(GRID_H):
                if x == 0 or y == 0 or x == GRID_W-1 or y == GRID_H-1:
                    g.tiles[x][y] = TileType.WALL
                else:
                    g.tiles[x][y] = TileType.DIRT
        g.cherries.clear()
        g.occupied.clear()
        # carve main tunnels via random walkers
        for _ in range(6):
            x = random.randint(2, GRID_W-3)
            y = random.randint(2, GRID_H-3)
            length = random.randint(50, 110)
            for __ in range(length):
                g.set_tile(x,y, TileType.TUNNEL)
                if random.random() < 0.5: x += random.choice([-1,1])
                else: y += random.choice([-1,1])
                x = clamp(x, 2, GRID_W-3)
                y = clamp(y, 2, GRID_H-3)
        # ensure a spawn room
        for x in range(3, GRID_W-3):
            for y in range(3, 6):
                g.set_tile(x,y, TileType.TUNNEL)
        # place cherries embedded in dirt clusters
        attempts = 0
        placed = 0
        while placed < self.cherry_clusters and attempts < 1000:
            attempts += 1
            x = random.randint(2, GRID_W-3)
            y = random.randint(5, GRID_H-3)
            # need 2x2 dirt block
            if all(g.get_tile(x+dx, y+dy) == TileType.DIRT for dx in (0,1) for dy in (0,1)):
                # plant cherries in 2x2 area
                for dx in (0,1):
                    for dy in (0,1):
                        g.cherries.add((x+dx, y+dy))
                placed += 1
        # apples inside dirt near tunnels
        game.apples.clear()
        tries = 0
        while len(game.apples) < self.apple_count and tries < 5000:
            tries += 1
            x = random.randint(2, GRID_W-3)
            y = random.randint(4, GRID_H-3)
            if g.get_tile(x,y) == TileType.DIRT and (x,y) not in g.occupied:
                # require a tunnel adjacent so it can fall/roll into it
                neigh = [(x+1,y),(x-1,y),(x,y+1),(x,y-1)]
                if any(g.is_passable(nx,ny) for nx,ny in neigh):
                    a = Apple((x,y))
                    game.apples.append(a)
                    a.occupy(game)
        print(f"[LEVEL] Placed {len(game.apples)} apples")
        # monsters
        game.monsters.clear()
        spawn_area = [(x,y) for x in range(4, GRID_W-4) for y in range(3,7) if g.is_passable(x,y)]
        random.shuffle(spawn_area)
        mcount = self.monster_count
        acount = min(self.alpha_count, mcount)
        for i in range(mcount):
            is_alpha = i < acount
            tp = spawn_area[i % len(spawn_area)]
            m = Monster(tp, is_alpha=is_alpha)
            m.speed = MONSTER_SPEED + (game.level-1)*MONSTER_SPEED_INC + random.uniform(-5,5)
            game.monsters.append(m)
        print(f"[LEVEL] Spawned monsters: {mcount} (alpha: {acount})")
        # player
        game.player.grid = vec2((GRID_W//2, 4))
        game.player.pos = vec2(tile_to_world(int(game.player.grid.x), int(game.player.grid.y)))
        game.player.invuln = True
        game.player.invuln_time = 2.0
        # powerball
        game.powerball.active = False
        game.powerball.returning = False

# ------------- GAME -------------
class Game:
    def __init__(self):
        pygame.init()
        flags = pygame.SRCALPHA
        self.screen = pygame.display.set_mode((SCREEN_W, SCREEN_H), flags)
        pygame.display.set_caption("Digging Action Game")
        self.clock = pygame.time.Clock()
        self.grid = Grid(GRID_W, GRID_H)
        self.hud = HUD()
        # state
        self.score = 0
        self.level = 1
        self.lives = 3
        self.paused = False
        # entities
        self.player = Player(tile_to_world(GRID_W//2, GRID_H//2))
        self.apples = []
        self.monsters = []
        self.pickups = []
        self.powerball = PowerBall(self.player.pos)
        self.particles = []
        # letters
        self.letters = set()
        # lighting
        self.dark_surf = pygame.Surface((SCREEN_W, SCREEN_H), pygame.SRCALPHA)
        # build first level
        self.level_def = Level(self.level)
        self.level_def.build(self)
        print("[GAME] Initialized")
        self.state = "intro"  # intro, playing, life_lost, level_cleared, game_over
        self.hud.flash("PRESS ENTER TO START")
    def letters_needed(self):
        return set(EXTRA_LETTERS) - self.letters
    def add_score(self, pts):
        self.score += pts
        print(f"[SCORE] +{pts} => {self.score}")
    def collect_letter(self, ch):
        self.letters.add(ch)
        self.hud.flash(f"LETTER {ch}!")
        if self.letters_needed() == set():
            self.lives += 1
            self.letters.clear()
            self.hud.flash("EXTRA! +1 LIFE")
            print("[EXTRA] Extra life awarded")
    def spawn_dig_particles(self, tile):
        cx, cy = tile_to_world(*tile)
        for _ in range(PARTICLE_COUNT_DIG):
            ang = random.uniform(0, math.tau)
            spd = random.uniform(40, 150)
            vel = vec2(math.cos(ang)*spd, math.sin(ang)*spd)
            p = Particle((cx, cy), vel, (140,90,50), random.uniform(0.3,0.7), random.uniform(1,2))
            self.particles.append(p)
    def spawn_crush_particles(self, pos, color=(200,80,100)):
        for _ in range(PARTICLE_COUNT_CRUSH):
            ang = random.uniform(0, math.tau)
            spd = random.uniform(80, 240)
            vel = vec2(math.cos(ang)*spd, math.sin(ang)*spd)
            p = Particle(pos, vel, color, random.uniform(0.4,0.8), random.uniform(2,3))
            self.particles.append(p)
    def player_die(self):
        if self.state != "playing":
            return
        self.lives -= 1
        self.spawn_crush_particles(self.player.pos, color=(255,200,80))
        print(f"[GAME] Player died. Lives left: {self.lives}")
        if self.lives <= 0:
            self.state = "game_over"
            self.hud.flash("GAME OVER! PRESS ENTER")
        else:
            self.state = "life_lost"
            self.hud.flash("PRESS ENTER")
        # stop powerball
        self.powerball.active = False
        # reset monsters apple chains
        for a in self.apples:
            a.chain_crushes = 0
    def level_clear(self):
        self.state = "level_cleared"
        self.hud.flash("LEVEL CLEARED! PRESS ENTER")
        print(f"[LEVEL] Cleared {self.level}")
    def next_level(self):
        self.level += 1
        self.level_def = Level(self.level)
        self.level_def.build(self)
        self.state = "playing"
        self.hud.flash(f"LEVEL {self.level}")
    def update(self, dt):
        self.hud.update(dt)
        if self.state == "intro":
            return
        if self.state == "life_lost" or self.state == "game_over" or self.state == "level_cleared":
            return
        # playing
        self.player.update(dt, self)
        for a in self.apples:
            a.update(dt, self)
        for m in self.monsters:
            m.update(dt, self)
        for p in self.pickups:
            p.update(dt, self)
        self.pickups = [p for p in self.pickups if not p.dead]
        self.powerball.update(dt, self)
        # particles
        for pr in self.particles:
            pr.update(dt)
        self.particles = [pr for pr in self.particles if pr.alive()]
        # win condition
        if len(self.grid.cherries) == 0 or all(m.dead for m in self.monsters):
            self.level_clear()
        # catch powerball to rearm
        if not self.powerball.active:
            self.player.powerball_ready = True
    def draw_grid(self, surf):
        # background dirt texture
        surf.fill((0, 0, 0))
        # dirt cells: draw as solid fill; tunnels as darker with subtle texture
        for x in range(GRID_W):
            for y in range(GRID_H):
                rect = pygame.Rect(x*TILE, y*TILE, TILE, TILE)
                tt = self.grid.get_tile(x,y)
                if tt == TileType.WALL:
                    pygame.draw.rect(surf, WALL_COLOR, rect)
                elif tt == TileType.DIRT:
                    # noisy dirt
                    base = pygame.Surface((TILE, TILE))
                    base.fill(DIRT_COLOR)
                    for i in range(4):
                        c = (DIRT_COLOR[0]+random.randint(-5,5),
                             DIRT_COLOR[1]+random.randint(-5,5),
                             DIRT_COLOR[2]+random.randint(-5,5))
                        pygame.draw.circle(base, c, (random.randint(0,TILE), random.randint(0,TILE)), random.randint(1,3))
                    surf.blit(base, rect)
                elif tt == TileType.TUNNEL:
                    pygame.draw.rect(surf, TUNNEL_COLOR, rect)
                    # tunnel walls highlight
                    pygame.draw.rect(surf, (30,16,12), rect, 1)
        # cherries
        for (x,y) in self.grid.cherries:
            cx, cy = tile_to_world(x,y)
            pygame.draw.circle(surf, (220,0,0), (cx-6, cy), 6)
            pygame.draw.circle(surf, (220,0,0), (cx+6, cy), 6)
            pygame.draw.line(surf, (40,200,40), (cx, cy-8), (cx, cy-18), 2)
            pygame.draw.circle(surf, (255,180,180), (cx-8, cy-2), 2)
            pygame.draw.circle(surf, (255,180,180), (cx+4, cy-2), 2)
    def draw_lighting(self, surf):
        self.dark_surf.fill((0,0,0, LIGHT_DARK_ALPHA))
        # player light
        self.draw_radial_light(self.dark_surf, self.player.pos, LIGHT_PLAYER_RADIUS, (255, 180, 80))
        # subtle glows on powerball
        if self.powerball.active:
            self.draw_radial_light(self.dark_surf, self.powerball.pos, 70, (120, 200, 255))
        # apply lighting overlay with multiplicative feel (approx using alpha clear)
        surf.blit(self.dark_surf, (0,0))
    def draw_radial_light(self, surface, pos, radius, color):
        # simple soft-edged radial gradient
        for i in range(LIGHT_SOFT_STEPS):
            r = int(radius * (1 - i / LIGHT_SOFT_STEPS))
            alpha = int(160 * (1 - i / (LIGHT_SOFT_STEPS+1)))
            c = (color[0], color[1], color[2], alpha)
            pygame.draw.circle(surface, c, (int(pos.x), int(pos.y)), r)
        # punch-through center
        hole = pygame.Surface((radius*2, radius*2), pygame.SRCALPHA)
        pygame.draw.circle(hole, (0,0,0,0), (radius, radius), int(radius*0.5))
        surface.blit(hole, (pos.x-radius, pos.y-radius), special_flags=0)
    def draw(self):
        self.draw_grid(self.screen)
        # entities
        for a in self.apples:
            a.draw(self.screen, self)
        for m in self.monsters:
            m.draw(self.screen, self)
        for p in self.pickups:
            p.draw(self.screen, self)
        self.player.draw(self.screen, self)
        self.powerball.draw(self.screen, self)
        # particles on top
        for pr in self.particles:
            pr.draw(self.screen)
        # lighting last
        self.draw_lighting(self.screen)
        # HUD
        self.hud.draw(self.screen, self.score, self.lives, self.level, self.letters, len(self.grid.cherries))
        # overlay prompts
        if self.state == "intro":
            self.center_text("DIGGING ACTION GAME", (255,255,255), y=SCREEN_H//2 - 20, size=36)
            self.center_text("Arrows: Move/ Dig/ Push | Space: Throw Powerball | Enter: Start", (200,200,200), y=SCREEN_H//2 + 20)
        elif self.state == "life_lost":
            self.center_text("YOU DIED - PRESS ENTER", (255,160,160), y=SCREEN_H//2)
        elif self.state == "level_cleared":
            self.center_text("LEVEL CLEARED! PRESS ENTER", (160,255,160), y=SCREEN_H//2)
        elif self.state == "game_over":
            self.center_text("GAME OVER - PRESS ENTER", (255,120,120), y=SCREEN_H//2)
        pygame.display.flip()
    def center_text(self, text, color, y=None, size=24):
        font = pygame.freetype.SysFont(self.hud.font.name, size) if hasattr(self.hud.font, "name") else self.hud.font
        text_surf, rect = font.render(text, color)
        if y is None: y = SCREEN_H//2
        rect.center = (SCREEN_W//2, y)
        self.screen.blit(text_surf, rect)
    def handle_events(self):
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                pygame.quit()
                sys.exit(0)
            elif e.type == pygame.KEYDOWN:
                if e.key == pygame.K_ESCAPE:
                    pygame.quit()
                    sys.exit(0)
                if e.key == pygame.K_RETURN:
                    if self.state == "intro":
                        self.state = "playing"
                        self.hud.flash(f"LEVEL {self.level}")
                    elif self.state == "life_lost":
                        # rebuild level but keep score and lives
                        self.level_def.build(self)
                        self.state = "playing"
                        self.hud.flash(f"LEVEL {self.level}")
                    elif self.state == "level_cleared":
                        self.next_level()
                    elif self.state == "game_over":
                        # reset everything
                        self.score = 0
                        self.lives = 3
                        self.level = 1
                        self.letters.clear()
                        self.level_def = Level(self.level)
                        self.level_def.build(self)
                        self.state = "intro"
                        self.hud.flash("PRESS ENTER TO START")
    def run(self):
        acc = 0.0
        dt_fixed = 1.0 / FPS
        while True:
            self.handle_events()
            dt = self.clock.tick(CLAMPED_FPS) / 1000.0
            acc += dt
            # fixed update
            while acc >= dt_fixed:
                self.update(dt_fixed)
                acc -= dt_fixed
            self.draw()

# ------------- MAIN -------------
if __name__ == "__main__":
    Game().run()