param(
    [string]$Python = '3.12',
    [string]$EnvironmentPath = (Join-Path $PSScriptRoot '..\.cache\knowledge-runtime')
)
$ErrorActionPreference = 'Stop'
$runtimeProject = Join-Path $PSScriptRoot 'knowledge-runtime'
$resolvedEnvironment = [System.IO.Path]::GetFullPath($EnvironmentPath)
$previousUvEnvironment = $env:UV_PROJECT_ENVIRONMENT
$previousTokenizerCache = $env:TIKTOKEN_CACHE_DIR
$env:UV_PROJECT_ENVIRONMENT = $resolvedEnvironment
try {
    & uv sync --project $runtimeProject --python $Python --frozen --no-dev
    if ($LASTEXITCODE -ne 0) { throw 'Knowledge runtime installation failed.' }
    $knowledgePython = Join-Path $resolvedEnvironment 'Scripts\python.exe'
    $env:TIKTOKEN_CACHE_DIR = Join-Path $resolvedEnvironment 'tokenizer-cache'
    # Tokenizer vocabulary is prepared explicitly during installation. Inference
    # never implicitly installs packages or downloads an embedding model.
    & $knowledgePython -c 'import tiktoken; tiktoken.encoding_for_model("gpt-4o-mini"); import importlib.metadata; print("LightRAG", importlib.metadata.version("lightrag-hku"))'
    if ($LASTEXITCODE -ne 0) { throw 'Knowledge runtime/tokenizer check failed.' }
    Write-Output "Knowledge Python: $knowledgePython"
    Write-Output 'Select this interpreter in SceneRecall knowledge settings, then configure and test the model connections.'
} finally {
    if ($null -eq $previousUvEnvironment) { Remove-Item Env:UV_PROJECT_ENVIRONMENT -ErrorAction SilentlyContinue }
    else { $env:UV_PROJECT_ENVIRONMENT = $previousUvEnvironment }
    if ($null -eq $previousTokenizerCache) { Remove-Item Env:TIKTOKEN_CACHE_DIR -ErrorAction SilentlyContinue }
    else { $env:TIKTOKEN_CACHE_DIR = $previousTokenizerCache }
}
