"""Servidor en vivo: el visor + un chat, sobre la mente real (SPEC §7, Fase 4).

Una interfaz más, como el REPL — orquesta, no implementa: la conversación es
ChatSession, la red se accede por la fachada Mind, el HTML lo genera viz. El
servidor solo sirve ficheros estáticos (el visor lee el diario por HTTP) y
expone POST /chat. Es el embrión del futuro backend FastAPI, sin serlo aún.

Despliegue público: una capa de ACCESO (usuario/contraseña por ENTORNO, nunca en
el código → no se filtra al repo) y un RATE-LIMIT por IP (horario y diario)
protegen la demo y el gasto del LLM. Sin esas variables, la demo va abierta (local).
"""

import hashlib
import http.cookies
import http.server
import json
import os
import threading
import time
import urllib.parse
from collections import defaultdict
from functools import partial
from pathlib import Path

from ..providers.settings import ProviderSettings
from ..viz import graph_view
from .loop import IdleSleeper, build_chat, build_mind
from .session import ChatSession

DEMO_USER = os.environ.get("DEMO_USER")            # acceso: credenciales por ENV (no en el código)
DEMO_PASS = os.environ.get("DEMO_PASS")
AUTH_COOKIE = "oroi_auth"
RATE_HOURLY = int(os.environ.get("RATE_HOURLY", "20"))   # respuestas del LLM por IP y hora
RATE_DAILY = int(os.environ.get("RATE_DAILY", "100"))    # ... y por IP y día
LOGIN_MAX = int(os.environ.get("LOGIN_MAX", "8"))        # intentos de login FALLIDOS por IP y hora


def _auth_secret() -> str:  # valor opaco del cookie de sesión, derivado de las credenciales
    return hashlib.sha256(f"{DEMO_USER}:{DEMO_PASS}".encode()).hexdigest()


LOGIN_PAGE = """<!doctype html><html lang="es"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>oroi — acceso</title><style>
 *{box-sizing:border-box} body{margin:0;font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;
  min-height:100vh;display:flex;align-items:center;justify-content:center;
  background:radial-gradient(circle at 50% 38%,#1f2937,#0b0f17);padding:16px}
 .card{width:100%;max-width:380px;background:#fff;border-radius:18px;box-shadow:0 24px 60px rgba(0,0,0,.45);padding:34px 30px}
 .brand{font-size:22px;font-weight:700;color:#111827;letter-spacing:.3px}.brand b{color:#DC2626}
 h1{font-size:18px;color:#111827;margin:18px 0 4px}.sub{color:#6b7280;font-size:13px;margin:0 0 8px}
 label{display:block;font-size:12px;font-weight:600;color:#374151;margin:14px 0 6px}
 input{width:100%;padding:11px 13px;border:1px solid #d1d5db;border-radius:10px;font-size:14px;color:#111827;outline:none}
 input:focus{border-color:#DC2626;box-shadow:0 0 0 3px rgba(220,38,38,.15)}
 button{width:100%;margin-top:22px;padding:12px;border:0;border-radius:10px;background:#DC2626;color:#fff;
  font-size:15px;font-weight:600;cursor:pointer;transition:background .15s}button:hover{background:#b91c1c}
 .err{margin-top:16px;background:#fef2f2;border:1px solid #fecaca;color:#b91c1c;font-size:13px;
  text-align:center;padding:9px;border-radius:9px;font-weight:500}
</style></head><body>
 <form class="card" method="POST" action="/login">
  <div class="brand">llm·<b>mind</b></div>
  <h1>Acceso a la demo</h1><p class="sub">Memoria asociativa para chatbots</p>
  <label for="user">Email</label>
  <input id="user" name="user" type="email" required placeholder="you@example.com" autocomplete="username">
  <label for="pass">Contraseña</label>
  <input id="pass" name="pass" type="password" required placeholder="••••••••" autocomplete="current-password">
  __ERROR__
  <button type="submit">Entrar</button>
 </form></body></html>"""


def serve(db_path: str, port: int, host: str = "127.0.0.1") -> None:
    mind = build_mind(db_path)
    mind.wake()
    sleeper = IdleSleeper(mind, mind.config.idle_sleep_seconds)
    sleeper.start()
    journal = graph_view.timeline_path(db_path)
    session = ChatSession(mind, build_chat(ProviderSettings()),
                          on_turn=lambda snap: graph_view.record(snap, journal))
    page = graph_view.export_html([], f"{db_path}.live.html",
                                  live_journal=journal.name, chat_endpoint="/chat")
    directory = Path(db_path).resolve().parent
    limiter = _RateLimiter(RATE_HOURLY, RATE_DAILY)
    guard = _LoginGuard(LOGIN_MAX)
    handler = partial(_Handler, session=session, sleeper=sleeper, limiter=limiter,
                      guard=guard, live_page=page.name, directory=str(directory))
    access = "con acceso" if (DEMO_USER and DEMO_PASS) else "ABIERTA (sin DEMO_USER/DEMO_PASS)"
    print(f"visor + chat en vivo [{access}]: http://{host}:{port}/  (Ctrl+C para parar)")
    with http.server.ThreadingHTTPServer((host, port), handler) as server:
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
    if sleeper.pending:  # al cerrar, consolida lo que quedara fresco
        print("\n(consolidando recuerdos antes de salir…)")
        mind.sleep()


class _RateLimiter:
    """Tope de respuestas del LLM por IP: por hora y por día. En memoria, con lock (servidor multihilo)."""

    def __init__(self, hourly: int, daily: int):
        self.hourly, self.daily = hourly, daily
        self.hits: dict[str, list[float]] = defaultdict(list)
        self.lock = threading.Lock()

    def allow(self, ip: str) -> tuple[bool, str]:
        now = time.time()
        with self.lock:
            recent = [t for t in self.hits[ip] if now - t < 86400]
            self.hits[ip] = recent
            if sum(1 for t in recent if now - t < 3600) >= self.hourly:
                return False, f"límite de {self.hourly} mensajes/hora alcanzado; prueba más tarde"
            if len(recent) >= self.daily:
                return False, f"límite de {self.daily} mensajes/día alcanzado; vuelve mañana"
            recent.append(now)
            return True, ""


class _LoginGuard:
    """Anti fuerza-bruta de /login: cuenta FALLOS por IP en una ventana; al exceder, bloquea un rato."""

    def __init__(self, max_fails: int, window: int = 3600):
        self.max_fails, self.window = max_fails, window
        self.fails: dict[str, list[float]] = defaultdict(list)
        self.lock = threading.Lock()

    def blocked(self, ip: str) -> bool:
        now = time.time()
        with self.lock:
            self.fails[ip] = [t for t in self.fails[ip] if now - t < self.window]
            return len(self.fails[ip]) >= self.max_fails

    def fail(self, ip: str) -> None:
        with self.lock:
            self.fails[ip].append(time.time())


class _Handler(http.server.SimpleHTTPRequestHandler):
    """Estáticos (visor + diario) y POST /chat, tras la capa de acceso y el rate-limit."""

    def __init__(self, *args, session: ChatSession, sleeper: IdleSleeper,
                 limiter: _RateLimiter, guard: _LoginGuard, live_page: str, **kwargs):
        self.session, self.sleeper, self.limiter = session, sleeper, limiter
        self.guard, self.live_page = guard, live_page
        super().__init__(*args, **kwargs)

    @staticmethod
    def _gated() -> bool:
        return bool(DEMO_USER and DEMO_PASS)

    def _authed(self) -> bool:
        morsel = http.cookies.SimpleCookie(self.headers.get("Cookie", "")).get(AUTH_COOKIE)
        return bool(morsel and morsel.value == _auth_secret())

    def _client_ip(self) -> str:  # tras el proxy de EasyPanel, la IP real va en X-Forwarded-For
        xff = self.headers.get("X-Forwarded-For")
        return xff.split(",")[0].strip() if xff else self.client_address[0]

    def _login_page(self, error: str = "") -> None:
        body = LOGIN_PAGE.replace("__ERROR__", f'<div class="err">{error}</div>' if error else "").encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _redirect(self, location: str, cookie: str | None = None) -> None:
        self.send_response(302)
        if cookie:
            self.send_header("Set-Cookie", cookie)
        self.send_header("Location", location)
        self.end_headers()

    def _do_login(self) -> None:
        if not self._gated():
            return self._redirect("/")
        if self.guard.blocked(self._client_ip()):  # anti fuerza-bruta
            return self._login_page("Demasiados intentos. Espera un rato e inténtalo de nuevo.")
        length = int(self.headers.get("Content-Length", 0))
        form = urllib.parse.parse_qs(self.rfile.read(length).decode("utf-8"))
        user = form.get("user", [""])[0].strip().lower()
        password = form.get("pass", [""])[0]
        if user == DEMO_USER.strip().lower() and password == DEMO_PASS:
            self._redirect("/", f"{AUTH_COOKIE}={_auth_secret()}; HttpOnly; Path=/; Max-Age=86400; SameSite=Lax")
        else:
            self.guard.fail(self._client_ip())
            self._login_page("Credenciales no válidas.")

    def do_GET(self) -> None:
        if self._gated():
            if self.path == "/login":
                return self._login_page()
            if not self._authed():
                return self._redirect("/login")
        if self.path == "/":  # raíz → el visor (URL de demo limpia, sin el nombre del fichero)
            return self._redirect(f"/{self.live_page}")
        super().do_GET()

    def do_POST(self) -> None:
        if self.path == "/login":
            return self._do_login()
        if self.path != "/chat":
            return self.send_error(404)
        if self._gated() and not self._authed():
            return self._json({"error": "no autorizado"}, status=401)
        allowed, msg = self.limiter.allow(self._client_ip())
        if not allowed:
            return self._json({"error": msg}, status=429)
        try:
            length = int(self.headers.get("Content-Length", 0))
            user_text = json.loads(self.rfile.read(length))["text"]
            reply = self.session.turn(user_text)
            self.sleeper.touch()
            self._json({"reply": reply, "turn": self.session.mind.turn})
        except Exception as error:
            self._json({"error": str(error)}, status=500)

    def _json(self, payload: dict, status: int = 200) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args) -> None:
        pass  # sin ruido de peticiones en consola
