"""Точка входа: FastAPI + WebSocket."""

import asyncio
from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse
from core.cognitive_cycle import extract_facts, monologue, generate_response, critic

app = FastAPI()

HTML_PAGE = """
<!DOCTYPE html>
<html>
<head><title>Sofia</title></head>
<body>
<h2>Sofia — SOAR + LLM</h2>
<div id="chat"></div>
<input id="msg" type="text" placeholder="Скажи что-нибудь...">
<button onclick="send()">Отправить</button>
<script>
const ws = new WebSocket("ws://localhost:8000/ws/chat");
ws.onmessage = e => {
    const d = document.getElementById("chat");
    d.innerText += e.data;
    if (e.data.includes("[END]")) d.innerText += "\\n\\n";
};
function send() {
    const i = document.getElementById("msg");
    ws.send(i.value); i.value = "";
}
</script>
</body>
</html>
"""


@app.get("/")
async def root():
    return HTMLResponse(HTML_PAGE)


@app.websocket("/ws/chat")
async def chat(ws: WebSocket):
    await ws.accept()
    while True:
        user_msg = await ws.receive_text()

        # 1. Экстрактор
        facts = await extract_facts(user_msg)

        # 2. Монолог (внутренний)
        mono = await monologue(facts, [], user_msg)

        # 3. Ответ со стримингом
        async for token in generate_response(mono, user_msg):
            await ws.send_text(token)
        await ws.send_text("[END]")

        # 4. Критик (фоном, не блокирует)
        asyncio.create_task(run_critic(user_msg, facts))


async def run_critic(response: str, facts: str):
    critique = await critic(response, facts)
    # TODO: обновить память, XP, PAD
    print("CRITIC:", critique)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
