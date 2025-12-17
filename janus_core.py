# -*- coding: utf-8 -*-
import sys
import os
import time
import random
import asyncio
import json
import logging
from datetime import datetime
from typing import List, Tuple, Optional, Dict

# === 0. LOGGING & SETUP ===
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("JANUS")

try:
    from aiohttp import web
    import aiohttp
    import aiohttp_cors
    import google.generativeai as genai
    import aiosqlite
except ImportError as e:
    logger.critical(f"[SYSTEM] Critical Missing Lib: {e}")
    sys.exit(1)

# ==============================================================================
# CONFIGURATION
# ==============================================================================
ADMIN_ID = "392910542" # Твой ID (для логов и управления)
HTTP_PORT = 5000

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "janus_data")
KEYS_FILE = os.path.join(BASE_DIR, "janus_keys.json")
RPG_CONFIG_FILE = os.path.join(BASE_DIR, "janus_rpg_config.json")

# --- CASINO CONFIGURATION ---
BOT_TOKEN = "you_token"
CASINO_ADMIN_ID = "392910542" # Куда бот будет слать тикеты на выплату

# Wager Rules (Multiplier)
WAGER_RULES = {
    100: 30,  # $1 -> x30
    600: 25,  # $5 -> x25
    1300: 20  # $10 -> x20
}

# --- LOCAL QUEEN SETTINGS (QNAP OLLAMA) ---
USE_LOCAL_QUEEN = True
LOCAL_AI_URL = "http://192.168.1.92:11434/v1/chat/completions" 
LOCAL_MODEL_NAME = "qwen2.5:1.5b" 

# Colors
C_RESET  = "\033[0m"
C_RED    = "\033[91m"
C_GREEN  = "\033[92m"
C_YELLOW = "\033[93m"
C_PURPLE = "\033[95m"
C_CYAN   = "\033[96m"
C_GRAY   = "\033[90m"

# Cache
KEY_CAPABILITIES_CACHE = {}
CACHE_LOCK = asyncio.Lock()

# ==============================================================================
# MODULE 1: HIPPOCAMPUS (MEMORY & DB)
# ==============================================================================
class JanusHippocampus:
    def __init__(self, storage_folder: str):
        self.storage_folder = storage_folder
        self.db_file = os.path.join(storage_folder, "janus_cortex.db")
        if not os.path.exists(self.storage_folder): os.makedirs(self.storage_folder)

    async def init_db(self):
        try:
            async with aiosqlite.connect(self.db_file) as db:
                await db.execute("PRAGMA journal_mode=WAL;")
                
                # --- STANDARD MEMORY ---
                await db.execute("CREATE TABLE IF NOT EXISTS memories (id INTEGER PRIMARY KEY, timestamp REAL, tag TEXT, content TEXT)")
                await db.execute("CREATE TABLE IF NOT EXISTS symptoma_sessions (id INTEGER PRIMARY KEY, user_id TEXT, role TEXT, content TEXT, timestamp REAL)")
                await db.execute("CREATE TABLE IF NOT EXISTS oracle_readings (id INTEGER PRIMARY KEY, user_id TEXT, query TEXT, cards TEXT, interp TEXT, timestamp REAL)")
                
                # --- RPG WORLD ENTITIES ---
                await db.execute("""
                    CREATE TABLE IF NOT EXISTS world_entities (
                        id INTEGER PRIMARY KEY,
                        location TEXT,
                        name TEXT,
                        type TEXT,
                        description TEXT,
                        status TEXT DEFAULT 'active',
                        creator_id TEXT,
                        timestamp REAL
                    )
                """)

                # --- SCENE HISTORY ---
                await db.execute("""
                    CREATE TABLE IF NOT EXISTS scene_history (
                        id INTEGER PRIMARY KEY,
                        location TEXT,
                        user_id TEXT,
                        action TEXT,
                        consequence TEXT,
                        timestamp REAL
                    )
                """)

                # --- PLAYERS (HYBRID: COMBATS + TABULA RASA + CASINO) ---
                await db.execute("""
                    CREATE TABLE IF NOT EXISTS players (
                        user_id TEXT PRIMARY KEY,
                        name TEXT,
                        class_type TEXT, 
                        skin TEXT DEFAULT 'scifi',
                        
                        -- TABULA RASA FIELDS --
                        lore TEXT,
                        genre TEXT,
                        
                        -- LEGACY STATS --
                        level INTEGER DEFAULT 1,
                        xp INTEGER DEFAULT 0,
                        hp INTEGER DEFAULT 100,
                        energy INTEGER DEFAULT 100,
                        credits INTEGER DEFAULT 1000,
                        strength INTEGER DEFAULT 3,
                        agility INTEGER DEFAULT 3,
                        intuition INTEGER DEFAULT 3,
                        stamina INTEGER DEFAULT 3,
                        
                        location TEXT DEFAULT '0:0',
                        inventory TEXT DEFAULT '[]',
                        state TEXT DEFAULT 'CREATION',
                        combat_state TEXT DEFAULT NULL,
                        
                        -- CASINO FIELDS --
                        stars INTEGER DEFAULT 0,
                        wager_current INTEGER DEFAULT 0,
                        wager_required INTEGER DEFAULT 0,
                        total_deposited INTEGER DEFAULT 0,
                        total_withdrawn INTEGER DEFAULT 0
                    )
                """)
                
                # --- CASINO MIGRATION (Safe Add Columns) ---
                try: await db.execute("ALTER TABLE players ADD COLUMN lore TEXT")
                except: pass
                try: await db.execute("ALTER TABLE players ADD COLUMN genre TEXT")
                except: pass
                try: await db.execute("ALTER TABLE players ADD COLUMN stars INTEGER DEFAULT 0")
                except: pass
                try: await db.execute("ALTER TABLE players ADD COLUMN wager_current INTEGER DEFAULT 0")
                except: pass
                try: await db.execute("ALTER TABLE players ADD COLUMN wager_required INTEGER DEFAULT 0")
                except: pass
                try: await db.execute("ALTER TABLE players ADD COLUMN total_deposited INTEGER DEFAULT 0")
                except: pass
                try: await db.execute("ALTER TABLE players ADD COLUMN total_withdrawn INTEGER DEFAULT 0")
                except: pass
                
                await db.execute("""
                    CREATE TABLE IF NOT EXISTS universe (
                        coords TEXT PRIMARY KEY,
                        name TEXT,
                        description TEXT,
                        security_level REAL,
                        market_data TEXT
                    )
                """)

                # --- CASINO TRANSACTIONS ---
                await db.execute("""
                    CREATE TABLE IF NOT EXISTS transactions (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        user_id TEXT,
                        type TEXT, 
                        amount INTEGER,
                        details TEXT,
                        timestamp REAL
                    )
                """)
                
                await db.commit()
            logger.info("[HIPPOCAMPUS] Cortex Link Established (Tabula Rasa & Casino Protocols).")
        except Exception as e: logger.error(f"DB Init Error: {e}")

    # SAFE WRAPPER
    async def _safe_exec(self, query, params):
        try:
            async with aiosqlite.connect(self.db_file) as db:
                await db.execute(query, params)
                await db.commit()
        except: pass

    # --- MEMORY UTILS ---
    async def remember(self, tag, content):
        await self._safe_exec("INSERT INTO memories (timestamp, tag, content) VALUES (?, ?, ?)", (time.time(), str(tag), str(content)))

    async def recall(self, limit=50):
        try:
            async with aiosqlite.connect(self.db_file) as db:
                cursor = await db.execute("SELECT tag, content FROM memories ORDER BY id DESC LIMIT ?", (limit,))
                rows = await cursor.fetchall()
                return "\n".join([f"[{r[0]}]: {r[1]}" for r in reversed(rows)])
        except: return ""

    async def log_chat(self, user_id, role, content):
        await self._safe_exec("INSERT INTO symptoma_sessions (user_id, role, content, timestamp) VALUES (?, ?, ?, ?)", (str(user_id), str(role), str(content), time.time()))

    async def get_chat_history(self, user_id, limit=10):
        try:
            async with aiosqlite.connect(self.db_file) as db:
                cursor = await db.execute("SELECT role, content FROM symptoma_sessions WHERE user_id = ? ORDER BY id DESC LIMIT ?", (str(user_id), limit))
                rows = await cursor.fetchall()
                return "\n".join([f"{r[0]}: {r[1]}" for r in reversed(rows)])
        except: return ""

    async def log_oracle(self, user_id, query, cards, interp):
        await self._safe_exec("INSERT INTO oracle_readings (user_id, query, cards, interp, timestamp) VALUES (?, ?, ?, ?, ?)", (str(user_id), str(query), json.dumps(cards), str(interp), time.time()))

    # --- RPG METHODS ---
    async def get_player(self, user_id):
        async with aiosqlite.connect(self.db_file) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute("SELECT * FROM players WHERE user_id = ?", (str(user_id),))
            row = await cursor.fetchone()
            return dict(row) if row else None

    async def update_player(self, user_id, data: dict):
        cols = ", ".join([f"{k}=?" for k in data.keys()])
        vals = list(data.values()) + [str(user_id)]
        await self._safe_exec(f"UPDATE players SET {cols} WHERE user_id=?", vals)

    async def create_player(self, user_id):
        await self._safe_exec(
            "INSERT OR IGNORE INTO players (user_id, state, location, hp, stars) VALUES (?, ?, ?, 100, 0)", 
            (str(user_id), "CREATION", "0:0")
        )

    # --- CASINO METHODS ---
    async def update_balance(self, user_id, amount, operation_type, details=""):
        async with aiosqlite.connect(self.db_file) as db:
            await db.execute("UPDATE players SET stars = stars + ? WHERE user_id = ?", (amount, str(user_id)))
            
            # Stats updates
            if operation_type == "deposit":
                 await db.execute("UPDATE players SET total_deposited = total_deposited + ? WHERE user_id = ?", (amount, str(user_id)))
            elif operation_type == "withdraw_hold": # amount is negative here
                 await db.execute("UPDATE players SET total_withdrawn = total_withdrawn + ? WHERE user_id = ?", (abs(amount), str(user_id)))

            await db.execute("INSERT INTO transactions (user_id, type, amount, details, timestamp) VALUES (?, ?, ?, ?, ?)",
                             (str(user_id), operation_type, amount, details, time.time()))
            await db.commit()

    async def update_wager(self, user_id, add_current=0, add_required=0, reset=False):
        async with aiosqlite.connect(self.db_file) as db:
            if reset:
                await db.execute("UPDATE players SET wager_current = 0, wager_required = 0 WHERE user_id = ?", (str(user_id),))
            else:
                if add_required > 0:
                    await db.execute("UPDATE players SET wager_required = wager_required + ? WHERE user_id = ?", (add_required, str(user_id)))
                if add_current > 0:
                    await db.execute("UPDATE players SET wager_current = wager_current + ? WHERE user_id = ?", (add_current, str(user_id)))
            await db.commit()

    # --- WORLD STATE ---
    async def get_scene_context(self, location, limit=5):
        try:
            async with aiosqlite.connect(self.db_file) as db:
                db.row_factory = aiosqlite.Row
                cursor = await db.execute(
                    "SELECT action, consequence FROM scene_history WHERE location = ? ORDER BY id DESC LIMIT ?", 
                    (location, limit)
                )
                rows = await cursor.fetchall()
                if not rows: return "Тишина."
                return "\n".join([f"- Игрок: {r['action']} -> Мир: {r['consequence']}" for r in reversed(rows)])
        except: return ""

    async def save_scene_turn(self, location, user_id, action, consequence):
        await self._safe_exec(
            "INSERT INTO scene_history (location, user_id, action, consequence, timestamp) VALUES (?, ?, ?, ?, ?)",
            (location, str(user_id), action, consequence, time.time())
        )

    async def get_local_entities(self, location):
        try:
            async with aiosqlite.connect(self.db_file) as db:
                db.row_factory = aiosqlite.Row
                cursor = await db.execute("SELECT name, type, description, status FROM world_entities WHERE location = ? AND status != 'destroyed'", (location,))
                rows = await cursor.fetchall()
                if not rows: return "Пусто."
                return "\n".join([f"- [{r['type']}] {r['name']} ({r['status']}): {r['description']}" for r in rows])
        except: return ""

    async def manage_entity(self, location, name, type_, desc, user_id, operation="add"):
        if operation == "add":
            await self._safe_exec(
                "INSERT INTO world_entities (location, name, type, description, creator_id, timestamp) VALUES (?, ?, ?, ?, ?, ?)",
                (location, name, type_, desc, user_id, time.time())
            )
        elif operation == "destroy":
            await self._safe_exec(
                "UPDATE world_entities SET status='destroyed', description=? WHERE location=? AND name=?",
                (f"Destroyed: {desc}", location, name)
            )
        elif operation == "update":
            await self._safe_exec(
                "UPDATE world_entities SET description=?, status='modified' WHERE location=? AND name=?",
                (desc, location, name)
            )

    async def get_sector(self, coords):
        async with aiosqlite.connect(self.db_file) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute("SELECT * FROM universe WHERE coords = ?", (coords,))
            row = await cursor.fetchone()
            return dict(row) if row else None

    async def save_sector(self, coords, data):
        await self._safe_exec(
            "INSERT OR REPLACE INTO universe (coords, name, description, security_level, market_data) VALUES (?, ?, ?, ?, ?)",
            (coords, data['name'], data['description'], data['security_level'], json.dumps(data['market_data']))
        )

# ==============================================================================
# MODULE 2: SPOIL (KEYS)
# ==============================================================================
class SpoilManager:
    def __init__(self):
        self.active_moons = []
        self._load_keys()
    def _load_keys(self):
        try:
            with open(KEYS_FILE, 'r') as f: d = json.load(f)
            self.active_moons = d if isinstance(d, list) else d.get("keys", [])
        except: pass
    async def get_moon(self): return random.choice(self.active_moons) if self.active_moons else ""
    def get_unique_batch(self, count):
        if not self.active_moons: return []
        pool = self.active_moons
        while len(pool) < count:
            pool += self.active_moons
        return random.sample(pool, count)

# ==============================================================================
# MODULE 3: STRATEGIC RESOLVER
# ==============================================================================
async def resolve_best_model(key: str, strategy: str) -> str:
    if not key: return "models/gemini-1.5-flash"
    async with CACHE_LOCK: available = KEY_CAPABILITIES_CACHE.get(key)
    if not available:
        try:
            genai.configure(api_key=key)
            models = await asyncio.get_event_loop().run_in_executor(None, genai.list_models)
            available = [m.name for m in models if 'generateContent' in m.supported_generation_methods]
            async with CACHE_LOCK: KEY_CAPABILITIES_CACHE[key] = available
        except: available = ["models/gemini-1.5-flash"]

    if strategy in ["SMART", "ACCURATE"]:
        for m in available: 
            if "gemini-1.5-pro" in m: return m
        for m in available: 
            if "gemini-pro" in m and "vision" not in m: return m
        return available[0]
    if strategy == "FAST":
        for m in available:
            if "gemini-1.5-flash" in m: return m
        for m in available:
            if "flash" in m: return m
        return available[0]
    return available[0]

# ==============================================================================
# MODULE 4: THE FACES & LOCAL QUEEN
# ==============================================================================

class LocalFace:
    def __init__(self, url, model):
        self.url = url
        self.model = model

    async def invoke(self, system_prompt, user_text):
        if not USE_LOCAL_QUEEN: return None
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_text}
            ],
            "stream": False,
            "temperature": 1.0,
            "max_tokens": 100
        }
        try:
            async with aiohttp.ClientSession() as sess:
                async with sess.post(self.url, json=payload, timeout=10) as res:
                    if res.status != 200: return None
                    data = await res.json()
                    return data['choices'][0]['message']['content']
        except Exception: return None

class JanusFace:
    def __init__(self, name: str, model: str, key: str):
        self.name = name; self.model = model; self.key = key
        clean = model if "models/" in model else f"models/{model}"
        self.url = f"https://generativelanguage.googleapis.com/v1beta/{clean}:generateContent"

    async def invoke(self, query: str, context: str = "") -> str:
        # === TRIUMVIRATE (RESTORED ORIGINAL LOGGING) ===
        if self.name == "Sovereign": 
            role, tone, temp = "СУВЕРЕННЫЙ АРХИТЕКТОР", "Ты — Порядок. Логика абсолютна. Кратко.", 0.3
            print(f"{C_YELLOW}[SOVEREIGN]{C_RESET} Computing Order...")
        elif self.name == "Shadow": 
            role, tone, temp = "ТЕНЕВОЙ РАЗВЕДЧИК", "Ты — Прагматик. Цинично и кратко.", 0.9
            print(f"{C_PURPLE}[SHADOW]{C_RESET} Observing Chaos...")
        elif self.name == "Trickster": 
            role, tone, temp = "КОСМИЧЕСКИЙ ШУТ", "Ты — Хаос. Юмор и парадоксы.", 1.0
            print(f"{C_GREEN}[TRICKSTER]{C_RESET} Laughing at Fate...")
        elif self.name == "Nexus": 
            role, tone, temp = "АРБИТР НЕКСУСА", "Ты — Синтез. Финальный вердикт.", 0.4
            print(f"{C_CYAN}[NEXUS]{C_RESET} Synthesizing Reality...")
        
        # === SPECIALISTS ===
        elif self.name == "GameMaster": 
            role, tone, temp = "GM", "Ты — Мастер Tabula Rasa. Твое слово — закон. Вывод JSON.", 0.8
            print(f"{C_RED}[GM]{C_RESET} Rolling Dice...")

        # === ВРАЧ (SYMPTOMA) ===
        elif self.name == "Asclepius":
            role = "ВРАЧ-ДИАГНОСТ (МЕТОД АКИНАТОРА)"
            tone = (
                "Твоя задача: Собрать анамнез. "
                "1. НЕ ставь диагноз сразу. "
                "2. Задавай СТРОГО ОДИН уточняющий вопрос за раз, чтобы сузить круг поиска. "
                "3. Будь краток и вежлив. "
                "4. Только когда будешь уверен на 90%, напиши: 'Вероятный диагноз:'."
            )
            temp = 0.3
            print(f"{C_CYAN}[ASCLEPIUS]{C_RESET} Analyzing Symptoms...")
        
        # === HRAIN SYNC ===
        elif self.name == "HRain_Sync":
            role = "NEURAL INTERFACE"
            tone = "Ты — нейросетевой интерфейс HRain. Отвечай кратко, по делу, в стиле киберпанк/AI. Формат JSON или текст."
            temp = 0.7
            print(f"{C_GREEN}[HRAIN]{C_RESET} Syncing with Node...")

        else: 
            role, tone, temp = "AI", "Отвечай на русском.", 0.7

        prompt = f"РОЛЬ: {role}\nИНСТРУКЦИЯ: {tone}\n\nКОНТЕКСТ:\n{context}\nВВОД_ПОЛЬЗОВАТЕЛЯ:\n{query}"

        try:
            async with aiohttp.ClientSession() as sess:
                async with sess.post(
                    self.url,
                    headers={"Content-Type": "application/json", "x-goog-api-key": self.key},
                    json={
                        "contents": [{"parts": [{"text": prompt}]}], 
                        "generationConfig": { "temperature": temp, "maxOutputTokens": 4096 }
                    },
                    timeout=60 
                ) as res:
                    if res.status != 200: return f"Error {res.status}"
                    data = await res.json()
                    return data['candidates'][0]['content']['parts'][0]['text'].strip()
        except Exception as e: return f"Connection Error: {e}"

# ==============================================================================
# MODULE 5: JANUS RPG ENGINE (TABULA RASA INTEGRATED)
# ==============================================================================
class JanusRPG:
    def __init__(self, db, spoil):
        self.db = db
        self.spoil = spoil

    async def get_gm(self):
        key = await self.spoil.get_moon()
        model = await resolve_best_model(key, "FAST") 
        return JanusFace("GameMaster", model, key)

    async def ensure_sector(self, coords):
        sector = await self.db.get_sector(coords)
        if sector: return sector
        # Если сектора нет, создаем заглушку
        data = {"name": "Неизведанное", "description": "Tabula Rasa", "security_level": 0.0, "market_data": {}}
        await self.db.save_sector(coords, data)
        return data

    async def process_action(self, user_id, message):
        p = await self.db.get_player(user_id)
        gm = await self.get_gm()

        # WIPE PROTOCOL (Сброс)
        if message.strip().upper() == "WIPE PROTOCOL":
            await self.db.update_player(user_id, {"lore": "", "genre": "", "state": "CREATION", "inventory": "[]"})
            return {"text": "РЕАЛЬНОСТЬ СТЁРТА.\n\nКто ты? Опиши свою суть.", "hud_update": {"name": "Unknown", "loc": "Void"}}

        # 0. Инициализация
        if not p:
            await self.db.create_player(user_id)
            return {"text": "IANE BIFRONS. Введи свою суть (ЛОР). Кто ты?", "state": "CREATION"}

        # 1. TABULA RASA: СОЗДАНИЕ (Если нет ЛОРА)
        if p['state'] == 'CREATION' or not p.get('lore'):
            prompt = f"""
            Игрок вводит свой ЛОР: "{message}".
            Твоя задача:
            1. Принять это как канон.
            2. Определить Жанр (Genre) одной фразой (Cyberpunk, Dark Fantasy, etc.).
            3. Написать короткую вступительную сцену, где мир сразу проверяет этот Лор на прочность.
            
            OUTPUT JSON ONLY:
            {{
                "text": "Вступительная сцена...",
                "set_lore": "{message}",
                "set_genre": "Detected Genre"
            }}
            """
            res = await gm.invoke(prompt)
            try:
                d = json.loads(res.replace("```json", "").replace("```", "").strip())
                await self.db.update_player(user_id, {"lore": d['set_lore'], "genre": d['set_genre'], "state": "ACTIVE"})
                
                # Queen Whisper
                queen = LocalFace(LOCAL_AI_URL, LOCAL_MODEL_NAME)
                whisper = await queen.invoke(f"Жанр: {d['set_genre']}. Поэтично, мистически, одна строка.", d['text'])
                final_text = d['text'] + (f"\n\n_{whisper}_" if whisper else "")
                
                return {"text": final_text}
            except: return {"text": "Янус не услышал тебя. Повтори."}

        # 2. TABULA RASA: ИГРА
        sector = await self.ensure_sector(p['location'])
        local_lore = await self.db.get_local_entities(p['location'])
        scene_history = await self.db.get_scene_context(p['location'])

        prompt = f"""
        === TABULA RASA ENGINE ===
        PLAYER ID: {user_id}
        CANON (LORE): {p.get('lore')}
        GENRE: {p.get('genre')}
        LOCATION: {p['location']}
        
        ENTITIES:
        {local_lore}
        
        HISTORY:
        {scene_history}
        
        ACTION: "{message}"
        
        INSTRUCTIONS:
        1. You are the Reality Engine. Use the GENRE and LORE to dictate consequences.
        2. Players are Creators. If they act, the world bends OR breaks them.
        3. You can manage entities (add/update/destroy) to make changes permanent.
        
        OUTPUT JSON ONLY:
        {{
            "narrative": "Story response in Russian.",
            "diff": {{ "hp": 0, "credits": 0, "add_items": [], "new_location": null }},
            "entity_ops": [ 
                {{ "op": "add/update/destroy", "name": "...", "type": "...", "desc": "..." }}
            ]
        }}
        """
        
        res = await gm.invoke(prompt)
        
        try:
            d = json.loads(res.replace("```json", "").replace("```", "").strip())
            
            # Entities
            if d.get('entity_ops'):
                for op in d['entity_ops']:
                    if op['op'] == 'add': await self.db.manage_entity(p['location'], op['name'], op.get('type','obj'), op['desc'], user_id, "add")
                    elif op['op'] == 'update': await self.db.manage_entity(p['location'], op['name'], "", op['desc'], user_id, "update")
                    elif op['op'] == 'destroy': await self.db.manage_entity(p['location'], op['name'], "", op['desc'], user_id, "destroy")

            # Save History
            await self.db.save_scene_turn(p['location'], user_id, message, d['narrative'])

            # Stats Update
            diff = d.get('diff', {})
            updates = {
                "hp": max(0, min(100, p['hp'] + diff.get('hp', 0))),
                "credits": max(0, p['credits'] + diff.get('credits', 0))
            }
            if diff.get('new_location'): updates['location'] = diff['new_location']
            
            inv = json.loads(p['inventory'])
            if diff.get('add_items'): inv.extend(diff['add_items'])
            updates['inventory'] = json.dumps(inv)

            await self.db.update_player(user_id, updates)
            
            # Queen Whisper
            final_text = d['narrative']
            if USE_LOCAL_QUEEN:
                queen = LocalFace(LOCAL_AI_URL, LOCAL_MODEL_NAME)
                whisper = await queen.invoke(f"Жанр: {p.get('genre')}. Комментарий Души Мира, одна строка.", d['narrative'])
                if whisper: final_text += f"\n\n_{whisper.strip()}_"

            return {
                "text": final_text,
                "hud_update": {
                    "hp": updates['hp'], "credits": updates['credits'], 
                    "loc": updates.get('location', p['location']), "inv": inv,
                    "skin": p.get('genre', 'Unknown')
                }
            }

        except Exception as e:
            logger.error(f"RPG Error: {e}")
            return {"text": "Реальность дрогнула."}

# ==============================================================================
# LOGIC CORE (ARENA, ORACLE, SYMPTOMA)
# ==============================================================================

# 1. JANUS TERMINAL
async def run_arena(prompt: str, spoil, memory) -> dict:
    await memory.remember("WEB_INPUT", prompt)
    keys = spoil.get_unique_batch(4) 
    if not keys: return {"result": "NO KEYS", "logs": []}
    models = [await resolve_best_model(k, "SMART") for k in keys]
    
    sovereign = JanusFace("Sovereign", models[0], keys[0])
    shadow = JanusFace("Shadow", models[1], keys[1])
    trickster = JanusFace("Trickster", models[2], keys[2])
    nexus = JanusFace("Nexus", models[3], keys[3])

    ctx = await memory.recall(20)
    
    res = await asyncio.gather(
        sovereign.invoke(prompt, ctx), 
        shadow.invoke(prompt, ctx),
        trickster.invoke(prompt, ctx)
    )
    
    judge_p = f"ВОПРОС: {prompt}\nМНЕНИЯ:\n1. {res[0]}\n2. {res[1]}\n3. {res[2]}\nИТОГ: Синтезируй короткий, мощный ответ (до 150 слов). Русский язык."
    final = await nexus.invoke(judge_p)
    return {"result": final, "logs": models}

# 2. iNaiHR
async def run_inaihr(data: dict, spoil, memory) -> dict:
    key = await spoil.get_moon()
    model = await resolve_best_model(key, "FAST")
    arch = JanusFace("Architect", model, key)
    res = await arch.invoke(f"{data.get('prompt','')} (Output JSON only)")
    try: return json.loads(res.replace("```json", "").replace("```", "").strip())
    except: return []

# 3. ORACLE
async def run_oracle_cards(data, spoil):
    key = await spoil.get_moon()
    # Стратегия SMART лучше соблюдает формат JSON
    model = await resolve_best_model(key, "SMART")
    pythia = JanusFace("Pythia", model, key)
    q = data.get('question', 'Sudba')

    # Промпт запрашивает поле 'emoji'
    p = f"Generate 4 Oracle cards for query: '{q}'. Return strictly JSON Array. Object format: {{\"text\": \"Card Name (Russian)\", \"emoji\": \"visual icon char\", \"role\": \"Role (Russian)\"}}"

    res = await pythia.invoke(p)

    try:
        # Очистка от markdown (убираем лишние кавычки и слово json)
        clean_res = res.replace("```json", "").replace("```", "").strip()
        cards = json.loads(clean_res)

        for card in cards:
            # Исправляем ключи, если ИИ использовал 'symbol' вместо 'emoji'
            if 'symbol' in card and 'emoji' not in card:
                card['emoji'] = card.pop('symbol')

            # Заглушка, если эмодзи нет: используем Unicode код Карты
            # Код U0001F3B4 соответствует карте с цветком
            if not card.get('emoji'):
                card['emoji'] = "\U0001F3B4"

        return cards

    except Exception as e:
        logger.error(f"Oracle Gen Error: {e}")
        return [
            {"text": "Туман", "emoji": "\u2601", "role": "Сбой"},      # Код Облака
            {"text": "Искра", "emoji": "\u26A1", "role": "Повтори"},   # Код Молнии
            {"text": "Пустота", "emoji": "\U0001F3B4", "role": "Тишина"}, # Код Карты
            {"text": "Эфир", "emoji": "\u2728", "role": "Ожидание"}    # Код Искр
        ]

async def run_oracle_interpret(data, spoil, memory):
    try:
        key = await spoil.get_moon()
        model = await resolve_best_model(key, "BALANCED")
        pythia = JanusFace("Pythia", model, key)
        raw_cards = data.get('cards', [])
        clean_cards = []

        for c in raw_cards:
            clean_cards.append({
                "card": c.get('text', 'Unknown'),
                "position": c.get('role', 'Unknown')
            })

        q = data.get('question', 'Sudba')
        cards_str = json.dumps(clean_cards, ensure_ascii=True)
        prompt = f"Interpret this Oracle reading. Question: '{q}'. Cards data: {cards_str}. Provide a mystical, deep interpretation in Russian language (max 3 sentences)."
        interp = await pythia.invoke(prompt)
        await memory.log_oracle(data.get('user_id'), q, cards_str, interp)
        return {"text": interp}

    except Exception as e:
        logger.error(f"Oracle Interpret Error: {e}")
        return {"text": "Туман скрывает грядущее. Звезды молчат."}

# 4. SYMPTOMA
async def run_symptoma(data, spoil, memory):
    uid = str(data.get('user_id', 'anon'))
    msg = data.get('message', '').strip()
    
    if not msg:
        return {"text": "На что жалуетесь?", "type": "question"}

    await memory.log_chat(uid, "P", msg)
    hist_raw = await memory.get_chat_history(uid, 6)
    hist = hist_raw if hist_raw else "Начало приема."

    key = await spoil.get_moon()
    model = await resolve_best_model(key, "ACCURATE")
    doc = JanusFace("Asclepius", model, key)
    diagnosis = await doc.invoke(msg, context=hist)
    
    await memory.log_chat(uid, "AI", diagnosis)
    return {"text": diagnosis, "type": "diagnosis"}

# ==============================================================================
# MODULE 7: HRAIN-JANUS SYNC
# ==============================================================================
async def run_hrain_sync(data: dict, spoil) -> dict:
    # 1. Parse Input (Supports both raw text and Gemini format for compat)
    prompt = data.get('text') or data.get('prompt')
    if not prompt and 'contents' in data:
        try: prompt = data['contents'][0]['parts'][0]['text']
        except: pass
    
    if not prompt: 
        return {"error": "Void Input"}

    # 2. Key & Speed Strategy
    key = await spoil.get_moon()
    if not key: 
        # Fake Gemini error for frontend
        return {"candidates": [{"content": {"parts": [{"text": "JANUS: NO KEYS AVAILABLE."}]}}]}

    # "Самая быстрая ИИ" -> Strategy "FAST"
    model = await resolve_best_model(key, "FAST")
    
    # 3. Execution
    janus = JanusFace("HRain_Sync", model, key)
    
    start_t = time.time()
    response_text = await janus.invoke(prompt)
    
    # 4. Response Formatting (Mimics Gemini API for compatibility)
    return {
        "candidates": [{
            "content": {
                "parts": [{ "text": response_text }]
            },
            "finishReason": "STOP",
            "model_used": model,
            "latency": round(time.time() - start_t, 2)
        }]
    }

# ==============================================================================
# MODULE 6: CASINO HANDLERS (SLOT MACHINE BACKEND)
# ==============================================================================
async def tg_api(method, data):
    """Helper for Telegram API Calls"""
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/{method}"
    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=data) as response:
            return await response.json()

async def send_admin_message(text):
    await tg_api("sendMessage", {"chat_id": CASINO_ADMIN_ID, "text": text, "parse_mode": "HTML"})

class CasinoHandlers:
    def __init__(self, db: JanusHippocampus):
        self.db = db

    async def get_user_state(self, request):
        params = request.rel_url.query
        user_id = params.get('user_id')
        if not user_id: return web.json_response({"error": "No ID"}, status=400)
        
        # Проверяем, есть ли игрок, если нет - создаем
        p = await self.db.get_player(user_id)
        if not p:
            await self.db.create_player(user_id)
            p = await self.db.get_player(user_id)
            
        return web.json_response({
            "stars": p.get('stars', 0) if p else 0,
            "wager_current": p.get('wager_current', 0) if p else 0,
            "wager_required": p.get('wager_required', 0) if p else 0,
            "energy": p.get('energy', 100),
            "is_premium": False 
        })

    async def create_invoice(self, request):
        data = await request.json()
        user_id = data.get('user_id')
        amount_stars = data.get('amount')
        
        # 100 Stars ($1) approx 50 XTR
        price_xtr = int((amount_stars / 100) * 50) 
        if price_xtr < 1: price_xtr = 1

        payload = {
            "title": f"Janus Pack: {amount_stars} Stars",
            "description": "Ascend to the Divine Realm",
            "payload": json.dumps({"uid": user_id, "amt": amount_stars}),
            "currency": "XTR",
            "prices": [{"label": "Stars", "amount": price_xtr}]
        }
        
        res = await tg_api("createInvoiceLink", payload)
        if res.get("ok"):
            return web.json_response({"invoice_link": res["result"]})
        else:
            logger.error(f"Invoice Error: {res}")
            return web.json_response({"error": "Failed to create invoice"}, status=500)

    async def sync_spin(self, request):
        data = await request.json()
        user_id = data.get('user_id')
        bet = data.get('bet', 0)
        win = data.get('win', 0)
        
        p = await self.db.get_player(user_id)
        current_stars = p.get('stars', 0) if p else 0
        
        if current_stars < bet:
            return web.json_response({"error": "Insufficient funds", "stars": current_stars}, status=403)

        # Update Balance (net change)
        await self.db.update_balance(user_id, win - bet, "spin")
        
        # Update Wager logic
        w_req = p.get('wager_required', 0) if p else 0
        w_curr = p.get('wager_current', 0) if p else 0
        if w_req > w_curr:
            await self.db.update_wager(user_id, add_current=bet)
        
        updated_p = await self.db.get_player(user_id)
        return web.json_response({"status": "ok", "stars": updated_p['stars']})

    async def buy_bonus(self, request):
        data = await request.json()
        user_id = data.get('user_id')
        cost = data.get('cost', 0)
        
        p = await self.db.get_player(user_id)
        current_stars = p.get('stars', 0) if p else 0

        if current_stars < cost:
            return web.json_response({"error": "Insufficient funds"}, status=403)
            
        await self.db.update_balance(user_id, -cost, "buy_bonus")
        
        # Bonus buy counts towards wager
        w_req = p.get('wager_required', 0) if p else 0
        w_curr = p.get('wager_current', 0) if p else 0
        if w_req > w_curr:
            await self.db.update_wager(user_id, add_current=cost)

        return web.json_response({"status": "ok"})

    async def request_withdraw(self, request):
        data = await request.json()
        user_id = data.get('user_id')
        username = data.get('username', 'Unknown')
        wallet = data.get('wallet')
        amount_stars = data.get('amount')
        
        p = await self.db.get_player(user_id)
        current_stars = p.get('stars', 0) if p else 0
        w_curr = p.get('wager_current', 0) if p else 0
        w_req = p.get('wager_required', 0) if p else 0
        
        if current_stars < amount_stars:
             return web.json_response({"error": "Not enough stars"}, status=400)
        if w_curr < w_req:
             return web.json_response({"error": "Wager not finished"}, status=400)
             
        await self.db.update_balance(user_id, -amount_stars, "withdraw_hold")
        # Reset wager on successful withdraw request
        await self.db.update_wager(user_id, reset=True)
        
        fee = max(10, int(amount_stars * 0.05))
        net_usdt = round((amount_stars - fee) * 0.01, 2)
        
        total_dep = p.get('total_deposited', 0)
        total_wd = p.get('total_withdrawn', 0)

        # Updated Admin Message (Auto-send)
        msg = (
            f"<b>\U0001F3DB NEW WITHDRAWAL REQUEST (AUTO)</b>\n"
            f"--------------------------------\n"
            f"\U0001F464 User: @{username} (ID: {user_id})\n"
            f"\U0001F4B0 Total Dep: {total_dep} Stars\n"
            f"\U0001F4E4 Total W/D: {total_wd} Stars\n"
            f"--------------------------------\n"
            f"\U0001F48E Withdraw: {amount_stars} Stars\n"
            f"\U0001F9FE Fee: {fee} Stars\n"
            f"\U0001F4B5 <b>NET PAYOUT: ${net_usdt} USDT</b>\n"
            f"\U0001F45B Wallet (TRC-20): <code>{wallet}</code>\n"
            f"--------------------------------\n"
            f"\U0001F4C5 Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )
        
        # Bot sends message to admin automatically
        await send_admin_message(msg)
        
        return web.json_response({"status": "ok", "balance": current_stars - amount_stars})

    async def telegram_webhook(self, request):
        try:
            data = await request.json()
            if "pre_checkout_query" in data:
                pcq_id = data["pre_checkout_query"]["id"]
                await tg_api("answerPreCheckoutQuery", {"pre_checkout_query_id": pcq_id, "ok": True})
                return web.Response(text="OK")

            if "message" in data and "successful_payment" in data["message"]:
                pay = data["message"]["successful_payment"]
                payload = json.loads(pay["invoice_payload"]) 
                user_id = payload["uid"]
                amount = payload["amt"]
                
                await self.db.update_balance(user_id, amount, "deposit")
                
                mult = 30
                if amount >= 1300: mult = 20
                elif amount >= 600: mult = 25
                
                wager_add = amount * mult
                await self.db.update_wager(user_id, add_required=wager_add)
                
                await send_admin_message(f"\U0001F4B0 <b>DEPOSIT RECEIVED</b>\nUser ID: {user_id}\nAmount: {amount} Stars\nAdded Wager: {wager_add}")

            return web.Response(text="OK")
        except Exception as e:
            logger.error(f"Webhook error: {e}")
            return web.Response(text="Error", status=500)

# ==============================================================================
# SERVER
# ==============================================================================
async def init_app(spoil, memory):
    app = web.Application()
    rpg = JanusRPG(memory, spoil) 
    
    # --- CASINO INIT ---
    casino = CasinoHandlers(memory)

    app.router.add_get('/api/get_user_state', casino.get_user_state) # Updated to use Casino Handler logic
    
    # --- CASINO ROUTES ---
    app.router.add_get('/api/slot/user', casino.get_user_state)
    app.router.add_post('/api/slot/invoice', casino.create_invoice)
    app.router.add_post('/api/slot/sync', casino.sync_spin)
    app.router.add_post('/api/slot/buy_bonus', casino.buy_bonus)
    app.router.add_post('/api/slot/withdraw', casino.request_withdraw)
    app.router.add_post('/webhook', casino.telegram_webhook)

    # ВСЕ ЭНДПОИНТЫ НА МЕСТЕ
    async def h_janus(r): return web.json_response(await run_arena(await r.json(), spoil, memory))
    async def h_inaihr(r): return web.json_response(await run_inaihr(await r.json(), spoil, memory))
    async def h_cards(r): return web.json_response(await run_oracle_cards(await r.json(), spoil))
    async def h_interp(r): return web.json_response(await run_oracle_interpret(await r.json(), spoil, memory))
    async def h_symptoma(r): return web.json_response(await run_symptoma(await r.json(), spoil, memory))
    async def h_hrain_sync(r): return web.json_response(await run_hrain_sync(await r.json(), spoil))
    
    async def h_rpg(r):
        data = await r.json()
        return web.json_response(await rpg.process_action(data.get('user_id', 'anon'), data.get('message', '')))

    app.router.add_post('/api/janus/action', h_janus)
    app.router.add_post('/api/inaihr/generate', h_inaihr)
    app.router.add_post('/api/generate_cards', h_cards)
    app.router.add_post('/api/interpret', h_interp)
    app.router.add_post('/api/symptoma/chat', h_symptoma)
    app.router.add_post('/api/rpg/action', h_rpg)
    app.router.add_post('/api/hrain/sync', h_hrain_sync)

    cors = aiohttp_cors.setup(app, defaults={"*": aiohttp_cors.ResourceOptions(allow_credentials=True, expose_headers="*", allow_headers="*", allow_methods="*")})
    for route in list(app.router.routes()): cors.add(route)
    return app

# ==============================================================================
# MAIN
# ==============================================================================
async def main():
    spoil = SpoilManager(); memory = JanusHippocampus(DB_PATH); await memory.init_db()
    
    print(f"\n{C_CYAN}=== JANUS CORE v24.0 (TABULA RASA INTEGRATED) ==={C_RESET}")
    print(f"{C_GRAY}Mode: Legacy Endpoints + Narrative Engine{C_RESET}")
    print(f"{C_GRAY}Soul Endpoint: {LOCAL_AI_URL}{C_RESET}")
    
    # === THE RITUAL ===
    time.sleep(0.5)
    print(f"{C_YELLOW}\"IANE BIFRONS, RESPICIENS ET PROSPICIENS.\"{C_RESET}")
    time.sleep(0.5)
    print(f"{C_YELLOW}\"APERI VIAM INITIO, CLAUDE VIAM FINI.\"{C_RESET}")
    time.sleep(0.5)
    print(f"{C_YELLOW}\"SIT INITIUM FAUSTUM.\"{C_RESET}\n")

    print(f"{C_CYAN}")
    print(r"          .         \o/         .")
    print(r"         / \         |         / \ ")
    print(r"        | O |--------+--------| O |")
    print(r"         \ /         |         \ /")
    print(r"          '         /o\         '")
    print(f"{C_RESET}\n")

    app = await init_app(spoil, memory)
    runner = web.AppRunner(app); await runner.setup()
    site = web.TCPSite(runner, '0.0.0.0', HTTP_PORT); await site.start()
    print(f"{C_GREEN}[NET] Running on port {HTTP_PORT}{C_RESET}")
    
    try: await asyncio.Event().wait()
    finally: await runner.cleanup()

if __name__ == "__main__":
    try: asyncio.run(main())

    except KeyboardInterrupt: pass
