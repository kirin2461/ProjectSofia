# Сборка автономной Софии: dist\Sofia\Sofia.exe
#
#   powershell -ExecutionPolicy Bypass -File build.ps1
#
# 1. кладёт в build\ llama.cpp (CUDA), кэш XTTS v2 и ONNX-модель Chroma —
#    чтобы готовая программа работала без интернета;
# 2. прогоняет юнит-тесты;
# 3. собирает PyInstaller по sofia.spec.
# GGUF-модель в сборку не входит (5–10 ГБ): лаунчер находит её на диске
# или скачивает Qwen3-4B при первом запуске.

param(
    [switch]$SkipTests,
    [string]$LlamaDir = ""
)

$ErrorActionPreference = "Stop"
$Root = $PSScriptRoot
$Py = Join-Path $Root ".venv\Scripts\python.exe"
$Build = Join-Path $Root "build"
New-Item -ItemType Directory -Force $Build | Out-Null

function Step($text) { Write-Host "`n=== $text ===" -ForegroundColor Cyan }

# ── llama.cpp ──────────────────────────────────────────────────────────────
Step "llama.cpp"
$llamaOut = Join-Path $Build "llama"
if (-not (Test-Path (Join-Path $llamaOut "llama-server.exe"))) {
    if (-not $LlamaDir) {
        $LlamaDir = Get-ChildItem (Join-Path $env:LOCALAPPDATA "llama.cpp") -Directory -ErrorAction SilentlyContinue |
            Where-Object { Test-Path (Join-Path $_.FullName "llama-server.exe") } |
            Sort-Object { [int]($_.Name -replace '\D', '') } -Descending |
            Select-Object -First 1 -ExpandProperty FullName
    }
    if ($LlamaDir) {
        New-Item -ItemType Directory -Force $llamaOut | Out-Null
        Get-ChildItem $LlamaDir -File | Where-Object {
            ($_.Extension -eq ".dll" -and ($_.Name -notlike "*-impl.dll" -or $_.Name -eq "llama-server-impl.dll")) -or
            $_.Name -eq "llama-server.exe" -or $_.Name -like "LICENSE*"
        } | Copy-Item -Destination $llamaOut
        Write-Host "Скопирован llama.cpp из $LlamaDir"
    } else {
        Write-Host "llama.cpp не найден — программа скачает его при первом запуске" -ForegroundColor Yellow
    }
} else { Write-Host "уже подготовлен" }

# ── XTTS v2 ────────────────────────────────────────────────────────────────
Step "XTTS v2"
$hfSrc = Join-Path $env:USERPROFILE ".cache\huggingface\hub\models--tts-hub--XTTS-v2"
$hfDst = Join-Path $Build "hf-cache\hub\models--tts-hub--XTTS-v2"
if ((Test-Path $hfSrc) -and -not (Test-Path (Join-Path $hfDst "snapshots"))) {
    New-Item -ItemType Directory -Force $hfDst | Out-Null
    foreach ($sub in "refs", "snapshots") {
        Copy-Item (Join-Path $hfSrc $sub) $hfDst -Recurse -Force
    }
    Write-Host "Скопирован кэш XTTS"
} elseif (-not (Test-Path $hfSrc)) {
    Write-Host "Кэша XTTS нет — голос скачается при первом синтезе" -ForegroundColor Yellow
} else { Write-Host "уже подготовлен" }

# ── Chroma ONNX ────────────────────────────────────────────────────────────
Step "Chroma all-MiniLM-L6-v2"
$chSrc = Join-Path $env:USERPROFILE ".cache\chroma\onnx_models"
$chDst = Join-Path $Build "chroma-cache\onnx_models"
if ((Test-Path $chSrc) -and -not (Test-Path $chDst)) {
    New-Item -ItemType Directory -Force (Split-Path $chDst) | Out-Null
    Copy-Item $chSrc $chDst -Recurse
    Write-Host "Скопирована ONNX-модель"
} else { Write-Host "пропуск" }

# ── Эмбеддинги для памяти и библиотеки ─────────────────────────────────────
Step "Эмбеддинги multilingual-e5-small"
$embedOut = Join-Path $Root "assets\embed\multilingual-e5-small"
$onnxFile = Join-Path $embedOut "onnx\model_quantized.onnx"
if (-not (Test-Path $onnxFile)) {
    New-Item -ItemType Directory -Force (Split-Path $onnxFile) | Out-Null
    $base = "https://huggingface.co/intfloat/multilingual-e5-small/resolve/main"
    foreach ($rel in "config.json", "tokenizer_config.json", "special_tokens_map.json",
                     "tokenizer.json", "onnx/model_quantized.onnx") {
        $dst = Join-Path $embedOut $rel
        if (-not (Test-Path $dst)) {
            Write-Host "скачиваю $rel"
            curl.exe -L --fail --retry 3 -o $dst "$base/$rel"
            if ($LASTEXITCODE -ne 0) { throw "не удалось скачать $rel" }
        }
    }
    Write-Host "Модель эмбеддингов скачана"
} else { Write-Host "уже подготовлена" }

# ── Tesseract: распознавание сканов ────────────────────────────────────────
# Библиотека книг смешанная (текст + сканы), поэтому OCR входит в сборку.
# Нужны только rus+eng, остальные языки не тянем.
Step "Tesseract (rus+eng)"
$tessOut = Join-Path $Build "tesseract"
$tessExe = Join-Path $tessOut "tesseract.exe"
if (-not (Test-Path (Join-Path $tessOut "tessdata\rus.traineddata"))) {
    $src = $env:TESSERACT_CMD
    if (-not $src -or -not (Test-Path $src)) { $src = "" }
    if (-not $src) {
        $cand = Join-Path $env:ProgramFiles "Tesseract-OCR\tesseract.exe"
        if (Test-Path $cand) { $src = $cand }
    }
    if (-not $src) {
        $cmd = Get-Command tesseract.exe -ErrorAction SilentlyContinue
        if ($cmd) { $src = $cmd.Source }
    }
    if (-not $src -and (Get-Command winget.exe -ErrorAction SilentlyContinue)) {
        Write-Host "Tesseract не найден — ставлю через winget"
        winget install --id UB-Mannheim.TesseractOCR --silent `
            --accept-package-agreements --accept-source-agreements --disable-interactivity | Out-Null
        $cand = Join-Path $env:ProgramFiles "Tesseract-OCR\tesseract.exe"
        if (Test-Path $cand) { $src = $cand }
    }
    if ($src) {
        $srcDir = Split-Path $src
        New-Item -ItemType Directory -Force (Join-Path $tessOut "tessdata") | Out-Null
        Copy-Item $src (Join-Path $tessOut "tesseract.exe") -Force
        # tesseract.exe тянет за собой ~60 DLL из своей папки
        Get-ChildItem $srcDir -Filter *.dll | Copy-Item -Destination $tessOut -Force
        Copy-Item (Join-Path $srcDir "tessdata\eng.traineddata") (Join-Path $tessOut "tessdata") -Force
        foreach ($extra in "configs", "tessconfigs") {
            $p = Join-Path $srcDir "tessdata\$extra"
            if (Test-Path $p) { Copy-Item $p (Join-Path $tessOut "tessdata") -Recurse -Force }
        }
        $pdfTtf = Join-Path $srcDir "tessdata\pdf.ttf"
        if (Test-Path $pdfTtf) { Copy-Item $pdfTtf (Join-Path $tessOut "tessdata") -Force }
        $rus = Join-Path $tessOut "tessdata\rus.traineddata"
        if (-not (Test-Path $rus)) {
            Write-Host "скачиваю русскую модель распознавания"
            curl.exe -L --fail --retry 3 -o $rus `
                "https://github.com/tesseract-ocr/tessdata/raw/main/rus.traineddata"
            if ($LASTEXITCODE -ne 0) { throw "не удалось скачать rus.traineddata" }
        }
        Write-Host "Tesseract подготовлен из $srcDir"
    } else {
        Write-Host "Tesseract не найден — сканы останутся без текста" -ForegroundColor Yellow
    }
} else { Write-Host "уже подготовлен" }

# ── Тесты ──────────────────────────────────────────────────────────────────
if (-not $SkipTests) {
    Step "Юнит-тесты"
    & $Py -m pytest (Join-Path $Root "tests\unit") -q
    if ($LASTEXITCODE -ne 0) { throw "Тесты не прошли — сборка остановлена" }
}

# ── PyInstaller ────────────────────────────────────────────────────────────
Step "PyInstaller"
& $Py -m PyInstaller (Join-Path $Root "sofia.spec") --noconfirm `
    --distpath (Join-Path $Root "dist") --workpath (Join-Path $Build "pyi")
if ($LASTEXITCODE -ne 0) { throw "PyInstaller завершился с ошибкой" }

$exe = Join-Path $Root "dist\Sofia\Sofia.exe"
$size = (Get-ChildItem (Join-Path $Root "dist\Sofia") -Recurse | Measure-Object Length -Sum).Sum / 1GB
Write-Host ("`nГотово: {0}  ({1:N1} ГБ)" -f $exe, $size) -ForegroundColor Green
