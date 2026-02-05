"""
Blade Arena — Refactored single-file game
- Fixed player select (click works)
- Add delete player option
- Robust button handling (debounced clicks)
- Clean structure for UI and game flow

Requirements:
pip install pygame opencv-python numpy
"""

import pygame
import cv2
import os
import json
import random
import math
import time
from pathlib import Path
from datetime import datetime

# ---------------- Config ----------------
W, H = 1000, 660
ASSETS = "assets"
DB = "players.json"
FPS = 60
AVATAR_SIZE = 96
Path(ASSETS).mkdir(parents=True, exist_ok=True)

pygame.init()
screen = pygame.display.set_mode((W, H))
pygame.display.set_caption("Blade Arena")
clock = pygame.time.Clock()

FONT = pygame.font.SysFont("Segoe UI", 18)
BIG = pygame.font.SysFont("Segoe UI", 30, bold=True)
SMALL = pygame.font.SysFont("Segoe UI", 14)

# optional music (place assets/bgm.mp3)
try:
    pygame.mixer.init()
    MUSIC_PATH = os.path.join(ASSETS, "bgm.mp3")
    if os.path.exists(MUSIC_PATH):
        pygame.mixer.music.load(MUSIC_PATH)
        pygame.mixer.music.set_volume(0.4)
        pygame.mixer.music.play(-1)
except Exception:
    pass

# ---------------- Storage ----------------
def init_db():
    if not os.path.exists(DB):
        with open(DB, "w") as f:
            json.dump({"players": []}, f)

def load_db():
    init_db()
    with open(DB, "r") as f:
        return json.load(f)["players"]

def save_db(players_list):
    with open(DB, "w") as f:
        json.dump({"players": players_list}, f, indent=2)

def add_player_record(name, img_path):
    players = load_db()
    players.append({"name": name, "photo": img_path, "created": datetime.now().isoformat()})
    save_db(players)

def delete_player_record_by_photo(photo_path):
    players = load_db()
    players = [p for p in players if p.get("photo") != photo_path]
    save_db(players)

# ---------------- Face capture (passport style) ----------------
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
def capture_and_crop(name, out_path, preview_window_title="Capture Face (SPACE to take, ESC to cancel)"):
    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        raise RuntimeError("Could not open webcam.")
    face_img = None
    start = time.time()
    while True:
        ret, frame = cam.read()
        if not ret:
            continue
        h, w, _ = frame.shape
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.15, 5)
        display = frame.copy()
        faces = sorted(list(faces), key=lambda f: f[2]*f[3], reverse=True) if len(faces) else []
        for (x,y,fw,fh) in faces[:1]:
            pad_w = int(fw * 0.7)
            pad_h = int(fh * 1.0)
            x1 = max(0, x - pad_w); y1 = max(0, y - pad_h)
            x2 = min(w, x + fw + pad_w); y2 = min(h, y + fh + pad_h)
            cv2.rectangle(display, (x1,y1), (x2,y2), (0,200,0), 2)
            face_img = frame[y1:y2, x1:x2]
        cv2.putText(display, preview_window_title, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (220,220,220), 1, cv2.LINE_AA)
        cv2.imshow(preview_window_title, display)
        key = cv2.waitKey(1)
        if key == 27:  # ESC
            face_img = None
            break
        if key == 32 and face_img is not None:  # SPACE
            face_img = cv2.resize(face_img, (256,256))
            cv2.imwrite(out_path, face_img)
            break
        # safety fallback (if camera freezes) after 90s
        if time.time() - start > 90:
            break
    cam.release()
    cv2.destroyAllWindows()
    return face_img is not None

# ---------------- Avatar helper ----------------
def make_circular_avatar(path, size=AVATAR_SIZE):
    surf = pygame.image.load(path).convert_alpha()
    surf = pygame.transform.smoothscale(surf, (size, size))
    mask = pygame.Surface((size, size), pygame.SRCALPHA)
    pygame.draw.circle(mask, (255,255,255,255), (size//2, size//2), size//2)
    out = pygame.Surface((size, size), pygame.SRCALPHA)
    out.blit(surf, (0,0))
    out.blit(mask, (0,0), special_flags=pygame.BLEND_RGBA_MIN)
    return out

# ---------------- UI primitives ----------------
def rounded_rect(surface, rect, color, radius=12, border=0, border_color=(0,0,0)):
    pygame.draw.rect(surface, color, rect, border_radius=radius)
    if border:
        pygame.draw.rect(surface, border_color, rect, width=border, border_radius=radius)

def draw_text(surface, text, pos, font=FONT, color=(245,245,245)):
    surface.blit(font.render(text, True, color), pos)

def draw_text_center(surface, text, y, font=BIG, color=(245,245,245)):
    txt = font.render(text, True, color)
    surface.blit(txt, (W//2 - txt.get_width()//2, y))

def draw_triangle(surface, x, y, size, color):
    h = size * 0.866
    points = [(x, y - 2*h/3), (x - size/2, y + h/3), (x + size/2, y + h/3)]
    pygame.draw.polygon(surface, color, points)

def draw_heart(surface, x, y, size, color):
    r = int(size * 0.45)
    left = (x - r//2, y - r//2)
    right = (x + r//2, y - r//2)
    pygame.draw.circle(surface, color, left, r)
    pygame.draw.circle(surface, color, right, r)
    pts = [(x - size, y), (x + size, y), (x, y + int(size*1.15))]
    pygame.draw.polygon(surface, color, pts)

# ---------------- UI components ----------------
class Button:
    def __init__(self, rect, text, primary=False):
        self.rect = pygame.Rect(rect)
        self.text = text
        self.primary = primary
        self.hover = False
        self.disabled = False
        self._last_click = 0.0  # debounce

    def draw(self, surf):
        if self.disabled:
            bg = (40,40,45)
        else:
            bg = (30,160,120) if self.primary else ((58,62,74) if self.hover else (40,44,54))
        rounded_rect(surf, self.rect, bg, radius=10)
        draw_text(surf, self.text, (self.rect.x + (self.rect.width//2 - FONT.size(self.text)[0]//2),
                                    self.rect.y + (self.rect.height//2 - FONT.size(self.text)[1]//2)))

    def handle_event(self, event):
        if self.disabled:
            return False
        if event.type == pygame.MOUSEMOTION:
            self.hover = self.rect.collidepoint(event.pos)
        if event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            if self.rect.collidepoint(event.pos):
                # debounce 200ms
                now = time.time()
                if now - self._last_click > 0.18:
                    self._last_click = now
                    return True
        return False

class Card:
    def __init__(self, rect, player_record):
        self.rect = pygame.Rect(rect)
        self.data = player_record
        self.hover = False
        try:
            self.thumb = make_circular_avatar(self.data["photo"], size=80)
        except Exception:
            self.thumb = None
        # delete button inside card
        self.del_btn = pygame.Rect(self.rect.right - 36, self.rect.top + 8, 28, 28)
        self.del_hover = False

    def draw(self, surf):
        rounded_rect(surf, self.rect, (28,30,36), radius=12)
        if self.hover:
            pygame.draw.rect(surf, (80,90,110), self.rect, 2, border_radius=12)
        # avatar
        if self.thumb:
            surf.blit(self.thumb, (self.rect.x + 12, self.rect.y + (self.rect.height - 80)//2))
        else:
            pygame.draw.circle(surf, (90,90,100), (self.rect.x + 52, self.rect.y + self.rect.height//2), 40)
        # name
        name_surf = BIG.render(self.data["name"], True, (230,230,230))
        surf.blit(name_surf, (self.rect.x + 110, self.rect.y + 28))
        created = self.data.get("created", "")[:10]
        meta = SMALL.render(f"Added: {created}", True, (170,170,170))
        surf.blit(meta, (self.rect.x + 110, self.rect.y + 64))
        # delete button (icon)
        pygame.draw.rect(surf, (200,50,50) if self.del_hover else (160,40,40), self.del_btn, border_radius=6)
        draw_text(surf, "Del", (self.del_btn.x + 6, self.del_btn.y + 4), font=SMALL)

    def handle_event(self, event):
        mx,my = pygame.mouse.get_pos()
        self.hover = self.rect.collidepoint((mx,my))
        self.del_hover = self.del_btn.collidepoint((mx,my))
        if event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            if self.del_btn.collidepoint(event.pos):
                return "delete"
            if self.rect.collidepoint(event.pos):
                return "select"
        return None

# ---------------- Player Select Screen (robust) ----------------
def player_select_screen(slot):
    """
    Shows saved players, allow selecting, deleting, or creating new.
    Returns (name, img_path) or (None, None) for back.
    """
    # load DB fresh each entry
    while True:
        players_db = load_db()
        per_row = 3
        card_w, card_h = 320, 120
        padding = 22
        x0 = (W - (card_w * per_row + padding * (per_row - 1))) // 2
        margin_top = 120
        # create cards
        cards = []
        for idx, p in enumerate(players_db):
            col = idx % per_row
            row = idx // per_row
            x = x0 + col * (card_w + padding)
            y = margin_top + row * (card_h + padding)
            cards.append(Card((x, y, card_w, card_h), p))
        # header buttons
        back_btn = Button((18, 18, 120, 44), "Back")
        create_btn = Button((W - 170, 18, 150, 44), "Create New", primary=True)
        # scroll handling
        scroll_y = 0
        scrolling = False
        dragging = False
        drag_last_y = 0
        running_select = True
        # loop for this screen
        while running_select:
            clock.tick(FPS)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit(); exit()
                # button events
                if back_btn.handle_event(event):
                    return None, None
                if create_btn.handle_event(event):
                    # create new flow
                    name = text_input_modal(f"Enter name for Player {slot}")
                    if not name:
                        break
                    fname = f"{ASSETS}/{name}_{random.randint(1000,9999)}.jpg"
                    try:
                        ok = capture_and_crop(name, fname)
                        if not ok:
                            # cancelled
                            break
                        add_player_record(name, fname)
                        # immediately return the created player
                        return name, fname
                    except Exception as e:
                        print("Camera error:", e)
                        break
                # card events
                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    dragging = True
                    drag_last_y = event.pos[1]
                if event.type == pygame.MOUSEBUTTONUP:
                    dragging = False
                    # propagate click to cards
                    # compute positions with scroll_y
                    for c in cards:
                        res = c.handle_event(event)
                        if res == "delete":
                            # confirm delete modal
                            confirm = confirm_modal(f"Delete {c.data['name']}? This will remove their photo.")
                            if confirm:
                                # remove file if exists
                                try:
                                    if os.path.exists(c.data.get("photo", "")):
                                        os.remove(c.data["photo"])
                                except Exception:
                                    pass
                                delete_player_record_by_photo(c.data.get("photo"))
                                # break to reload DB
                                running_select = False
                                break
                        elif res == "select":
                            return c.data["name"], c.data["photo"]
                if event.type == pygame.MOUSEMOTION and dragging:
                    dy = event.pos[1] - drag_last_y
                    drag_last_y = event.pos[1]
                    scroll_y += dy
                if event.type == pygame.MOUSEBUTTONDOWN and event.button in (4,5):
                    # wheel scroll
                    if event.button == 4:
                        scroll_y = min(scroll_y + 40, 0)
                    else:
                        scroll_y = max(scroll_y - 40, -10000)  # will be clamped below
            # draw
            screen.fill((12,14,20))
            draw_text_center(screen, f"Select Player {slot}", 38)
            back_btn.hover = back_btn.rect.collidepoint(pygame.mouse.get_pos())
            create_btn.hover = create_btn.rect.collidepoint(pygame.mouse.get_pos())
            back_btn.draw(screen); create_btn.draw(screen)
            # draw cards with scroll_y
            rows = (len(cards) + per_row - 1) // per_row
            content_h = rows * (card_h + padding)
            min_scroll = min(0, H - margin_top - content_h - 40)
            scroll_y = max(min_scroll, min(0, scroll_y))
            for c in cards:
                c.rect.y += scroll_y  # temporary adjust
            # re-draw cards (we will revert y after drawing)
            for c in cards:
                c.del_btn = pygame.Rect(c.rect.right - 36, c.rect.top + 8, 28, 28)
                c.draw(screen)
            # revert y changes
            for c in cards:
                c.rect.y -= scroll_y
            # footer hint
            hint = SMALL.render("Click card to select. 'Del' to remove. Drag/scroll to view.", True, (160,160,160))
            screen.blit(hint, (W//2 - hint.get_width()//2, H-28))
            pygame.display.update()
        # end internal loop; if delete happened, top-level will reload DB and rebuild cards
    # end outer while (should be unreachable)

# ---------------- Modal input & confirm ----------------
def text_input_modal(prompt):
    name = ""
    active = True
    while True:
        clock.tick(FPS)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit(); exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RETURN:
                    if name.strip():
                        return name.strip()
                elif event.key == pygame.K_ESCAPE:
                    return None
                elif event.key == pygame.K_BACKSPACE:
                    name = name[:-1]
                else:
                    if event.unicode.isprintable() and len(name) < 18:
                        name += event.unicode
        # draw modal
        overlay = pygame.Surface((W,H), pygame.SRCALPHA)
        overlay.fill((0,0,0,160))
        screen.blit(overlay, (0,0))
        box = pygame.Rect(W//2 - 320, H//2 - 80, 640, 160)
        rounded_rect(screen, box, (26,28,34), radius=12)
        draw_text_center(screen, prompt, H//2 - 40)
        txt = FONT.render(name + ("|" if pygame.time.get_ticks() % 1000 < 500 else ""), True, (230,230,230))
        screen.blit(txt, (box.x + 40, H//2 - 5))
        hint = SMALL.render("Enter = OK. Esc = Cancel", True, (150,150,150))
        screen.blit(hint, (box.x + 40, box.y + box.height - 36))
        pygame.display.update()

def confirm_modal(prompt):
    global btn_ok, btn_cancel
    while True:
        clock.tick(FPS)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit(); exit()
            if event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                mx,my = event.pos
                if btn_ok.rect.collidepoint((mx,my)):
                    return True
                if btn_cancel.rect.collidepoint((mx,my)):
                    return False
        # draw
        overlay = pygame.Surface((W,H), pygame.SRCALPHA); overlay.fill((0,0,0,160))
        screen.blit(overlay,(0,0))
        box = pygame.Rect(W//2 - 320, H//2 - 80, 640, 160)
        rounded_rect(screen, box, (26,28,34), radius=12)
        draw_text_center(screen, prompt, H//2 - 30, font=BIG)
        # create buttons local
        btn_ok = Button((W//2 - 140, H//2 + 10, 120, 42), "Delete", primary=True)
        btn_cancel = Button((W//2 + 20, H//2 + 10, 120, 42), "Cancel")
        btn_ok.hover = btn_ok.rect.collidepoint(pygame.mouse.get_pos())
        btn_cancel.hover = btn_cancel.rect.collidepoint(pygame.mouse.get_pos())
        btn_ok.draw(screen); btn_cancel.draw(screen)
        pygame.display.update()

# ---------------- Main Menu (uses Buttons properly) ----------------
def main_menu():
    start_btn = Button((W//2 - 150, 200, 300, 64), "Quick Start", primary=True)
    players_btn = Button((W//2 - 150, 290, 300, 54), "Player Select")
    settings_btn = Button((W//2 - 150, 354, 300, 54), "Settings")
    credits_btn = Button((W//2 - 150, 418, 300, 54), "Credits")
    quit_btn = Button((W//2 - 150, 482, 300, 54), "Quit")

    while True:
        clock.tick(FPS)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit(); exit()
            if start_btn.handle_event(event):
                return "quick_start"
            if players_btn.handle_event(event):
                return "player_select"
            if settings_btn.handle_event(event):
                settings_screen()
            if credits_btn.handle_event(event):
                credits_screen()
            if quit_btn.handle_event(event):
                pygame.quit(); exit()

        # draw
        screen.fill((10,12,18))
        draw_text_center(screen, "Blade Arena", 72)
        mouse_pos = pygame.mouse.get_pos()
        for b in (start_btn, players_btn, settings_btn, credits_btn, quit_btn):
            b.hover = b.rect.collidepoint(mouse_pos)
            b.draw(screen)
        pygame.display.update()

def settings_screen():
    back = Button((W//2 - 70, H - 90, 140, 44), "Back")
    sound_on = True
    while True:
        clock.tick(FPS)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit(); exit()
            if back.handle_event(event):
                return
            if event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                # toggle sound by clicking the text area
                if pygame.Rect(W//2 - 120, 150, 240, 50).collidepoint(event.pos):
                    sound_on = not sound_on
                    if pygame.mixer.get_init():
                        if sound_on:
                            pygame.mixer.music.unpause()
                        else:
                            pygame.mixer.music.pause()
        screen.fill((10,10,14))
        draw_text_center(screen, "Settings", 60)
        draw_button(pygame.Rect(W//2 - 120, 150, 240, 50), f"Music: {'ON' if sound_on else 'OFF'}", hover=False)
        back.hover = back.rect.collidepoint(pygame.mouse.get_pos())
        back.draw(screen)
        pygame.display.update()

def credits_screen():
    back = Button((W//2 - 70, H - 90, 140, 44), "Back")
    while True:
        clock.tick(FPS)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit(); exit()
            if back.handle_event(event):
                return
        screen.fill((10,10,14))
        draw_text_center(screen, "Credits & Info", 60)
        lines = [
            "Made with Python, Pygame and OpenCV",
            "Face capture + local player history",
            "Controls: Player1 - WASD | Player2 - Arrow keys",
            "ESC - Pause | R - Restart"
        ]
        for i,l in enumerate(lines):
            draw_text(screen, l, (W//2 - FONT.size(l)[0]//2, 140 + i*28), font=FONT)
        back.hover = back.rect.collidepoint(pygame.mouse.get_pos())
        back.draw(screen)
        pygame.display.update()

# ---------------- Game helpers & flow ----------------
def respawn_blade():
    return {"x": random.randint(120, W-120), "y": random.randint(160, H-120), "size": 34}

def respawn_health():
    return {"x": random.randint(120, W-120), "y": random.randint(160, H-120), "size": 22}

def create_players(p1_name, p1_img, p2_name, p2_img):
    try:
        a1 = make_circular_avatar(p1_img) if p1_img else None
    except:
        a1 = None
    try:
        a2 = make_circular_avatar(p2_img) if p2_img else None
    except:
        a2 = None
    if a1 is None:
        a1 = pygame.Surface((AVATAR_SIZE, AVATAR_SIZE), pygame.SRCALPHA); pygame.draw.circle(a1, (130,140,150), (AVATAR_SIZE//2, AVATAR_SIZE//2), AVATAR_SIZE//2)
    if a2 is None:
        a2 = pygame.Surface((AVATAR_SIZE, AVATAR_SIZE), pygame.SRCALPHA); pygame.draw.circle(a2, (150,120,120), (AVATAR_SIZE//2, AVATAR_SIZE//2), AVATAR_SIZE//2)
    return [
        {"name": p1_name, "x": 180, "y": H//2, "hp": 5, "blade": False, "img": a1, "color": (70,150,230)},
        {"name": p2_name, "x": W-180, "y": H//2, "hp": 5, "blade": False, "img": a2, "color": (235,80,80)}
    ]

def dist(a,b):
    return math.hypot(a["x"]-b["x"], a["y"]-b["y"])

# ---------------- Startup flow ----------------
init_db()
menu_choice = main_menu()

if menu_choice == "player_select":
    res1 = player_select_screen(1)
    if res1 == (None, None):
        p1_name, p1_img = "Player1", None
    else:
        p1_name, p1_img = res1
    res2 = player_select_screen(2)
    if res2 == (None, None):
        p2_name, p2_img = "Player2", None
    else:
        p2_name, p2_img = res2
else:
    # quick start -> prompt names and capture
    n1 = text_input_modal("Enter Player 1 Name")
    p1_name = n1 if n1 else "Player1"
    n2 = text_input_modal("Enter Player 2 Name")
    p2_name = n2 if n2 else "Player2"
    p1_img = f"{ASSETS}/{p1_name}_{random.randint(1000,9999)}.jpg"
    p2_img = f"{ASSETS}/{p2_name}_{random.randint(1000,9999)}.jpg"
    try:
        ok1 = capture_and_crop(p1_name, p1_img)
        if ok1:
            add_player_record(p1_name, p1_img)
        else:
            p1_img = None
        ok2 = capture_and_crop(p2_name, p2_img)
        if ok2:
            add_player_record(p2_name, p2_img)
        else:
            p2_img = None
    except Exception as e:
        print("Camera error:", e)
        p1_img = None; p2_img = None

players = create_players(p1_name, p1_img, p2_name, p2_img)
blade = respawn_blade()
health = respawn_health()
blade_angle = 0
paused = False

# ---------------- Main game loop ----------------
running = True
while running:
    dt = clock.tick(FPS) / 1000.0
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                paused = not paused
            if event.key == pygame.K_r:
                players = create_players(p1_name, p1_img, p2_name, p2_img)
                blade = respawn_blade(); health = respawn_health(); blade_angle = 0

    if paused:
        # simple paused overlay
        overlay = pygame.Surface((W,H), pygame.SRCALPHA); overlay.fill((0,0,0,160))
        screen.blit(overlay, (0,0))
        draw_text_center(screen, "Paused", 140)
        resume_btn = Button((W//2 - 120, 240, 240, 52), "Resume", primary=True)
        restart_btn = Button((W//2 - 120, 310, 240, 44), "Restart")
        menu_btn = Button((W//2 - 120, 370, 240, 44), "Main Menu")
        mouse_pos = pygame.mouse.get_pos()
        for b in (resume_btn, restart_btn, menu_btn):
            b.hover = b.rect.collidepoint(mouse_pos)
            b.draw(screen)
        pygame.display.update()
        # wait for clicks inside paused overlay
        paused_event = pygame.event.wait()
        if paused_event.type == pygame.MOUSEBUTTONUP and paused_event.button == 1:
            if resume_btn.rect.collidepoint(paused_event.pos):
                paused = False
            elif restart_btn.rect.collidepoint(paused_event.pos):
                players = create_players(p1_name, p1_img, p2_name, p2_img)
                blade = respawn_blade(); health = respawn_health(); blade_angle = 0; paused = False
            elif menu_btn.rect.collidepoint(paused_event.pos):
                # go to main menu then reselect players
                menu_choice = main_menu()
                if menu_choice == "player_select":
                    r1 = player_select_screen(1)
                    if r1 != (None,None): p1_name, p1_img = r1
                    r2 = player_select_screen(2)
                    if r2 != (None,None): p2_name, p2_img = r2
                    players = create_players(p1_name, p1_img, p2_name, p2_img)
                    blade = respawn_blade(); health = respawn_health(); blade_angle = 0; paused = False
        continue

    # input movement
    keys = pygame.key.get_pressed()
    speed = 240 * dt
    if keys[pygame.K_w]: players[0]["y"] -= speed
    if keys[pygame.K_s]: players[0]["y"] += speed
    if keys[pygame.K_a]: players[0]["x"] -= speed
    if keys[pygame.K_d]: players[0]["x"] += speed
    if keys[pygame.K_UP]: players[1]["y"] -= speed
    if keys[pygame.K_DOWN]: players[1]["y"] += speed
    if keys[pygame.K_LEFT]: players[1]["x"] -= speed
    if keys[pygame.K_RIGHT]: players[1]["x"] += speed

    # clamp
    for p in players:
        p["x"] = max(90, min(W-90, p["x"]))
        p["y"] = max(160, min(H-90, p["y"]))

    # pickups & logic
    for i, p in enumerate(players):
        if dist(p, blade) < (blade["size"]/1.2 + 40):
            p["blade"] = True
            players[1-i]["blade"] = False
            blade = respawn_blade()
    for p in players:
        if dist(p, health) < (health["size"]/1.2 + 40):
            p["hp"] = min(5, p["hp"] + 1)
            health = respawn_health()
    # combat
    if dist(players[0], players[1]) < 82:
        if players[0]["blade"] and not players[1]["blade"]:
            players[1]["hp"] -= 0.09
        elif players[1]["blade"] and not players[0]["blade"]:
            players[0]["hp"] -= 0.09

    # draw frame
    screen.fill((14,16,22))
    rounded_rect(screen, pygame.Rect(0,0,W,110), (20,22,28), radius=0)
    # top UI
    screen.blit(BIG.render(players[0]["name"], True, players[0]["color"]), (24, 18))
    for i in range(int(players[0]["hp"])): draw_heart(screen, 24 + i*34 + 2, 70, 14, (235,60,60))
    name_r = BIG.render(players[1]["name"], True, players[1]["color"])
    screen.blit(name_r, (W - name_r.get_width() - 24, 18))
    for i in range(int(players[1]["hp"])):
        draw_heart(screen, W - (i + 1) * 34 - 24, 70, 14, (235, 60, 60))

    # powerups
    draw_heart(screen, int(health["x"]), int(health["y"]), int(health["size"]), (220,40,40))
    draw_triangle(screen, int(blade["x"]), int(blade["y"]), int(blade["size"]), (245,245,245))

    # aura behind (draw first)
    blade_angle += 6
    for p in players:
        if p.get("blade"):
            AURA_SIZE = 180
            aura = pygame.Surface((AURA_SIZE, AURA_SIZE), pygame.SRCALPHA)
            cx, cy = AURA_SIZE//2, AURA_SIZE//2
            inner_r, outer_r = 36, 72
            for k in range(6):
                ang = math.radians(blade_angle + k*60)
                dx1 = int(inner_r * math.cos(ang)); dy1 = int(inner_r * math.sin(ang))
                dx2 = int(outer_r * math.cos(ang)); dy2 = int(outer_r * math.sin(ang))
                pygame.draw.polygon(aura, (200,24,24,140), [(cx,cy),(cx+dx1,cy+dy1),(cx+dx2,cy+dy2)])
            pygame.draw.circle(aura, (255,120,120,120), (cx,cy), 12)
            aura_pos = (int(p["x"]) - cx, int(p["y"]) - cy)
            screen.blit(aura, aura_pos)

    # draw avatars on top
    for p in players:
        rect = p["img"].get_rect(center=(int(p["x"]), int(p["y"])))
        screen.blit(p["img"], rect)

    # win check
    if players[0]["hp"] <= 0 or players[1]["hp"] <= 0:
        winner = players[1]["name"] if players[0]["hp"] <= 0 else players[0]["name"]
        overlay = pygame.Surface((W,H), pygame.SRCALPHA); overlay.fill((6,8,10,210))
        screen.blit(overlay,(0,0))
        draw_text_center(screen, f"{winner} Wins!", H//2 - 40)
        btn = Button((W//2 - 120, H//2 + 20, 240, 56), "Restart", primary=True)
        btn.hover = True; btn.draw(screen)
        pygame.display.update()
        pygame.time.delay(1200)
        players = create_players(p1_name, p1_img, p2_name, p2_img)
        blade = respawn_blade(); health = respawn_health(); blade_angle = 0
        continue

    # bottom HUD
    hint = SMALL.render("WASD | Arrows — ESC: Pause  ·  R: Restart", True, (170,170,170))
    screen.blit(hint, (20, H-34))

    pygame.display.update()

pygame.quit()
