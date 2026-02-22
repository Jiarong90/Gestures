import time
import threading
import pygame
import socket

# CONFIG

# PLAYER
# SPRITE_PATH = "spritesheets/sephiroth_sprite.png" 
SPRITE_PATH = "spritesheets/soldier.png" 
# FRAME = 64
# ROWS = 4
# COLS = 21
# SCALE = 2
# DRAW_FRAME = FRAME * SCALE
FRAME = 32
ROWS = 4
COLS = 31
SCALE = 2
DRAW_FRAME = FRAME * SCALE

# PLAYER SPRITESHEET CONFIG
IDLE   = (1, 2)
WALK   = (3, 10)
ATTACK = (11, 21)

# ENEMY
# ENEMY_SPRITE_PATH = "spritesheets/OrcWarrior.png"
ENEMY_SPRITE_PATH = "spritesheets/soldier.png" 
ENEMY_FRAME = 32
ENEMY_ROWS = 4
ENEMY_COLS = 31
ENEMY_SCALE = 2
ENEMY_DRAW = ENEMY_FRAME * ENEMY_SCALE

# ENEMY SPRITESHEET CONFIG
ENEMY_ROW = 1
ENEMY_IDLE = (11, 22)    
ENEMY_IDLE_SL = slice(ENEMY_IDLE[0]-1, ENEMY_IDLE[1])

ENEMY_IDLE_FPS = 10

# Receive UDP label signal from live_prediction_keys.py
def udp_label_receiver():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.bind(("127.0.0.1", 5055))
    while True:
        data, _ = s.recvfrom(1024)
        set_label(data.decode("utf-8"))

# convert to python slices 
def to_slice(rng_1b):
    a, b = rng_1b
    return slice(a-1, b) 

IDLE_SL   = to_slice(IDLE)    # frames[0:2]
WALK_SL   = to_slice(WALK)    # frames[2:10]
ATTACK_SL = to_slice(ATTACK)  # frames[10:21]

# Basically take the correct row from my spritesheet
ROW_LEFT  = 1
ROW_RIGHT = 2

# window
W, H = 960, 540
GROUND_Y = 420

# movement/physics
MOVE_SPEED = 220.0      
JUMP_V = -520.0          
GRAVITY = 1500.0         

# animation speeds (frames per second)
IDLE_FPS = 10
WALK_FPS = 12
ATTACK_FPS = 18

# action cooldowns
JUMP_COOLDOWN = 0.40
ATTACK_COOLDOWN = 0.30
PAUSE_COOLDOWN = 0.50

latest_label = "none"
lock = threading.Lock()

def set_label(lbl: str):
    global latest_label
    with lock:
        latest_label = lbl

def get_label():
    with lock:
        return latest_label
    
# Sprite loading / slicing

def load_rows(path):
    sheet = pygame.image.load(path).convert_alpha()
    sw, sh = sheet.get_size()
    assert sh == FRAME * ROWS, f"Expected height {FRAME*ROWS}, got {sh}"
    assert sw == FRAME * COLS, f"Expected width {FRAME*COLS}, got {sw}"

    rows = []
    for r in range(ROWS):
        frames = []
        for c in range(COLS):
            rect = pygame.Rect(c*FRAME, r*FRAME, FRAME, FRAME)
            frames.append(sheet.subsurface(rect).copy())
        rows.append(frames)
    return rows

def frames_from_row(row_frames):
    idle   = row_frames[IDLE_SL]
    walk   = row_frames[WALK_SL]
    attack = row_frames[ATTACK_SL]
    return idle, walk, attack

# LOAD ENEMY
def load_rows_generic(path, frame, rows, cols):
    sheet = pygame.image.load(path).convert_alpha()
    sw, sh = sheet.get_size()
    assert sh == frame * rows, f"Expected height {frame*rows}, got {sh}"
    assert sw == frame * cols, f"Expected width {frame*cols}, got {sw}"

    out = []
    for r in range(rows):
        frames = []
        for c in range(cols):
            rect = pygame.Rect(c*frame, r*frame, frame, frame)
            frames.append(sheet.subsurface(rect).copy())
        out.append(frames)
    return out

# Fill in placeholder background
def draw_background(screen):
    # sky
    screen.fill((135, 206, 235))
    # random circles to represent.. hills
    pygame.draw.circle(screen, (120, 190, 120), (200, 380), 180)
    pygame.draw.circle(screen, (110, 180, 110), (520, 400), 220)
    # ground
    pygame.draw.rect(screen, (60, 160, 75),
                 pygame.Rect(0, GROUND_Y, W, H - GROUND_Y))
    pygame.draw.rect(screen, (110, 80, 50),
                    pygame.Rect(0, GROUND_Y, W, 18))

# Simple animation helper
class AnimPlayer:
    def __init__(self):
        self.t = 0.0
        self.idx = 0
        self._last_len = None

    def step(self, frames, fps, dt, loop=True):
        if not frames:
            return None

        # reset index/timer
        if self._last_len != len(frames):
            self._last_len = len(frames)
            self.t = 0.0
            self.idx = 0

        if self.idx >= len(frames):
            self.idx = 0 if loop else (len(frames) - 1)

        self.t += dt
        frame_time = 1.0 / max(1, fps)

        while self.t >= frame_time:
            self.t -= frame_time
            self.idx += 1
            if self.idx >= len(frames):
                self.idx = 0 if loop else (len(frames) - 1)

        return frames[self.idx]

    def reset(self):
        self.t = 0.0
        self.idx = 0
        self._last_len = None

def main():
    threading.Thread(target=udp_label_receiver, daemon=True).start()

    pygame.init()
    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption("Gesture Sprite Demo")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 28)
    big_font = pygame.font.SysFont(None, 90)


    # ENEMY
    enemy_hp = 10
    enemy_hit_flash = 0.0

    victory = False

    enemy_rows = load_rows_generic(ENEMY_SPRITE_PATH, ENEMY_FRAME, ENEMY_ROWS, ENEMY_COLS)
    enemy_idle_frames = enemy_rows[ENEMY_ROW][ENEMY_IDLE_SL]

    enemy_anim = AnimPlayer()
    enemy_x = 700
    enemy_y = float(GROUND_Y)

    # Collision rectangle for hits
    enemy_rect = pygame.Rect(enemy_x, int(GROUND_Y) - 60, 60, 60)

    ENEMY_DRAW_I = int(ENEMY_DRAW)   

    # LOAD PLAYER
    rows = load_rows(SPRITE_PATH)
    left_idle, left_walk, left_attack   = frames_from_row(rows[ROW_LEFT])
    right_idle, right_walk, right_attack = frames_from_row(rows[ROW_RIGHT])

    # player state
    x, y = 140.0, float(GROUND_Y)
    vy = 0.0
    on_ground = True
    facing = "right"
    state = "idle" # idle/walk/attack/jump
    paused = False

    anim = AnimPlayer()

    last_jump = 0.0
    last_attack = 0.0
    last_pause = 0.0


    running = True
    held_key = None  # "a" or "d" or None
    while running:
        dt = clock.tick(60) / 1000.0
        now = time.time()

        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                running = False

        # keyboard fallback 
        keys = pygame.key.get_pressed()
        kb_left  = keys[pygame.K_a]
        kb_right = keys[pygame.K_d]
        kb_jump  = keys[pygame.K_w]
        kb_atk   = keys[pygame.K_SPACE]
        kb_pause = keys[pygame.K_p]

        # gesture label (your mapping)
        lbl = get_label()

        enemy_draw_x = enemy_x
        enemy_draw_y = int(enemy_y) - (ENEMY_DRAW_I - ENEMY_FRAME) - 12

        enemy_rect.x = enemy_draw_x
        enemy_rect.y = enemy_draw_y
        enemy_rect.width = ENEMY_DRAW_I
        enemy_rect.height = ENEMY_DRAW_I

        # Victory screen controls
        if victory:
            keys = pygame.key.get_pressed()
            if keys[pygame.K_ESCAPE]:
                running = False
                continue
            if keys[pygame.K_r]:
                # quick restart 
                enemy_hp = 10
                victory = False
                enemy_hit_flash = 0.0
                x, y = 140.0, float(GROUND_Y)
                vy = 0.0
                on_ground = True
                state = "idle"
                anim.reset()
                enemy_anim.reset()
                continue

            # draw frozen scene + overlay
            draw_background(screen)

            # draw dead enemy 
            dead_txt = font.render("Enemy defeated!", True, (240,240,240))
            screen.blit(dead_txt, (enemy_x - 20, GROUND_Y - 28))

            # draw player current frame 
            frame = right_idle[0] if facing == "right" else left_idle[0]
            frame_big = pygame.transform.scale(frame, (DRAW_FRAME, DRAW_FRAME))
            screen.blit(frame_big, (int(x), int(y) - (DRAW_FRAME - FRAME) - 12))

            # overlay text
            msg = big_font.render("VICTORY!", True, (255,255,255))
            sub = font.render("Press R to restart, ESC to quit", True, (230,230,230))
            screen.blit(msg, (W//2 - msg.get_width()//2, 160))
            screen.blit(sub, (W//2 - sub.get_width()//2, 240))

            pygame.display.flip()
            continue


        # pause toggle (trigger + cooldown)
        pause_trigger = (lbl == "palm") or kb_pause
        if pause_trigger and (now - last_pause) > PAUSE_COOLDOWN:
            paused = not paused
            last_pause = now

        if paused:
            # draw paused frame
            screen.fill((25, 25, 30))
            txt = font.render("PAUSED (palm / P)", True, (240,240,240))
            screen.blit(txt, (W//2 - 90, 20))
            pygame.display.flip()
            continue

        # hold movement
        move_dir = 0
        if lbl == "point_left" or kb_left:
            move_dir = -1
            facing = "left"
        elif lbl == "point_right" or kb_right:
            move_dir = 1
            facing = "right"

        # jump trigger
        jump_trigger = (lbl == "point_up") or kb_jump
        if jump_trigger and on_ground and (now - last_jump) > JUMP_COOLDOWN:
            vy = JUMP_V
            on_ground = False
            last_jump = now
            if state != "attack":
                state = "jump"
                anim.reset()

        # attack trigger
        atk_trigger = (lbl == "fist") or kb_atk
        if atk_trigger and (now - last_attack) > ATTACK_COOLDOWN:
            state = "attack"
            anim.reset()
            last_attack = now

        # apply movement 
        x += move_dir * MOVE_SPEED * dt
        x = max(0, min(W - DRAW_FRAME, x))

        # physics
        if not on_ground:
            vy += GRAVITY * dt
            y += vy * dt
            if y >= GROUND_Y:
                y = float(GROUND_Y)
                vy = 0.0
                on_ground = True
                if state == "jump":
                    state = "idle"
                    anim.reset()

        # decide state if not attacking/jumping
        if state not in ("attack", "jump"):
            state = "walk" if move_dir != 0 else "idle"

        # choose frames set
        if facing == "left":
            idle_frames, walk_frames, attack_frames = left_idle, left_walk, left_attack
        else:
            idle_frames, walk_frames, attack_frames = right_idle, right_walk, right_attack

        # animate + handle attack end + hit detection
        if state == "idle":
            frame = anim.step(idle_frames, IDLE_FPS, dt, loop=True)
        elif state == "walk":
            frame = anim.step(walk_frames, WALK_FPS, dt, loop=True)
        elif state == "jump":
            # show idle frame while in air
            frame = idle_frames[0]
        else:  # attack
            frame = anim.step(attack_frames, ATTACK_FPS, dt, loop=False)

            # very simple hitbox
            hitbox = pygame.Rect(int(x), int(y) - (DRAW_FRAME - FRAME), DRAW_FRAME, DRAW_FRAME)
            if facing == "right":
                hitbox.x += DRAW_FRAME // 2
            else:
                hitbox.x -= DRAW_FRAME // 2

            # only count a hit during early-middle frames
            atk_progress = anim.idx / max(1, len(attack_frames)-1)
            if 0.25 <= atk_progress <= 0.55 and enemy_hp > 0:
                if hitbox.colliderect(enemy_rect):
                    enemy_hp -= 1
                    enemy_hit_flash = 0.15
                    if enemy_hp <= 0:
                        victory = True
            # end attack after last frame + go idle
            if anim.idx >= len(attack_frames) - 1 and anim.t < (1.0 / ATTACK_FPS):
                state = "idle"
                anim.reset()

        # enemy flash timer
        if enemy_hit_flash > 0:
            enemy_hit_flash -= dt
            if enemy_hit_flash < 0:
                enemy_hit_flash = 0

        # draw
        draw_background(screen)

 
        # DRAW ENEMY
        if enemy_hp > 0:
            eframe = enemy_anim.step(enemy_idle_frames, ENEMY_IDLE_FPS, dt, loop=True)

            # draw a tinted rect behind enemy
            if enemy_hit_flash > 0:
                pygame.draw.rect(screen, (255, 140, 140), enemy_rect)

            eframe_big = pygame.transform.scale(eframe, (ENEMY_DRAW_I, ENEMY_DRAW_I))
            screen.blit(eframe_big, (enemy_x, int(enemy_y) - (ENEMY_DRAW_I - ENEMY_FRAME) - 12))

            hp_txt = font.render(f"Enemy HP: {enemy_hp}", True, (240,240,240))
            screen.blit(hp_txt, (enemy_draw_x + (ENEMY_DRAW_I - hp_txt.get_width()) // 2,
                     enemy_draw_y - 22))
        else:
            dead_txt = font.render("Enemy defeated!", True, (240,240,240))
            screen.blit(dead_txt, (enemy_x - 20, GROUND_Y - 28))
 
        # player sprite
        frame_big = pygame.transform.scale(frame, (DRAW_FRAME, DRAW_FRAME))
        screen.blit(frame_big, (int(x), int(y) - (DRAW_FRAME - FRAME) - 12))

        # HUD
        hud = font.render(f"gesture:{lbl}  state:{state}  facing:{facing}", True, (240,240,240))
        screen.blit(hud, (12, 12))
        hint = font.render("Keyboard fallback: A/D move, W jump, SPACE attack, P pause", True, (200,200,200))
        screen.blit(hint, (12, 40))

        pygame.display.flip()

    pygame.quit()

if __name__ == "__main__":
    main()
