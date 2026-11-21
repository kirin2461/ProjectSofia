param(
    [string]$Model = "$env:APPDATA\Jan\data\llamacpp\models\Huihui-Qwythos-9B-Claude-Mythos-5-1M-abliterated-Q8_0\model.gguf",
    [string]$Alias = "qwythos-9b",
    [int]$Port = 8080,
    [int]$Ctx = 8192,
    [int]$GpuLayers = 30,
    [int]$Threads = 8,
    [string]$Server = "$env:LOCALAPPDATA\llama.cpp\b12606\llama-server.exe"
)
if (-not (Test-Path $Server)) { throw "llama-server не найден: $Server" }
if (-not (Test-Path $Model))  { throw "модель не найдена: $Model" }

# Qwen3: без "размышлений" — быстрее и прямее ответы.
# через env, иначе PowerShell ломает JSON-кавычки
$env:LLAMA_ARG_CHAT_TEMPLATE_KWARGS = '{"enable_thinking": false}'

$args = @(
    "--model", $Model,
    "--alias", $Alias,
    "--host", "127.0.0.1",
    "--port", "$Port",
    "--ctx-size", "$Ctx",
    "--n-gpu-layers", "$GpuLayers",
    "--cache-type-k", "q8_0",
    "--cache-type-v", "q8_0",
    # cache-reuse ускоряет повторный префикс, но при ngl>=30 llama.cpp его
    # отключает (пишет безобидный warning) — оставляем для меньших offload.
    "--cache-reuse", "256",
    "--threads", "$Threads",
    "--batch-size", "1024",
    "--ubatch-size", "512",
    "--parallel", "1",
    "--jinja",
    "--no-webui",
    "--temp", "0.7",
    "--top-p", "0.9",
    "--top-k", "40",
    "--min-p", "0.05",
    "--repeat-penalty", "1.08"
)
Write-Host "llama-server: $Alias (ngl=$GpuLayers ctx=$Ctx port=$Port)"
& $Server @args
