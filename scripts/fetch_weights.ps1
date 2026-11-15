# Download the Z-Image-Turbo weights this setup needs and verify each one against
# its exact byte count before accepting it.
#
#   powershell -ExecutionPolicy Bypass -File scripts\fetch_weights.ps1 -Target C:\...\ComfyUI\models
#   powershell -ExecutionPolicy Bypass -File scripts\fetch_weights.ps1 -Target C:\...\ComfyUI\models -Variant int8
#   powershell -ExecutionPolicy Bypass -File scripts\fetch_weights.ps1 -Target C:\...\ComfyUI\models -IncludeControlNet
#
# ModelScope 直链实测 9-11 MB/s；curl.exe 自带断点续传（-C -），断了重跑即可。
#
# 显存档位怎么选（8GB 显存笔记本）：
#   bf16   12.3GB DiT + 8.0GB TE = 20.3GB  -> 必须 --lowvram 大量卸载，最慢
#   int8    6.2GB DiT + 5.6GB TE = 11.8GB  -> 推荐，官方模板有对应组合
#   nvfp4   4.5GB DiT + 3.5GB TE =  8.0GB  -> 最省显存，官方模板暂无，DiT 用 UNETLoader 直接换名即可
param(
  [string]$Target = ".\models",
  [ValidateSet('bf16', 'int8', 'nvfp4')]
  [string]$Variant = 'bf16',
  [switch]$IncludeControlNet,
  [switch]$SkipExisting
)

$ErrorActionPreference = 'Stop'

$MS = 'https://www.modelscope.cn/api/v1/models/Comfy-Org/z_image_turbo/repo?Revision=master&FilePath='

$Variants = @{
  'bf16' = @{
    DiT = @{ Path = 'split_files/diffusion_models/z_image_turbo_bf16.safetensors';     Size = [long]12309866400; Dir = 'diffusion_models' }
    TE  = @{ Path = 'split_files/text_encoders/qwen_3_4b.safetensors';                  Size = [long]8044982048;  Dir = 'text_encoders' }
  }
  'int8' = @{
    DiT = @{ Path = 'split_files/diffusion_models/z_image_turbo_int8_convrot.safetensors'; Size = [long]6201001296; Dir = 'diffusion_models' }
    TE  = @{ Path = 'split_files/text_encoders/qwen_3_4b_fp8_mixed.safetensors';           Size = [long]5631994051; Dir = 'text_encoders' }
  }
  'nvfp4' = @{
    DiT = @{ Path = 'split_files/diffusion_models/z_image_turbo_nvfp4.safetensors';     Size = [long]4509509600; Dir = 'diffusion_models' }
    TE  = @{ Path = 'split_files/text_encoders/qwen_3_4b_fp4_mixed.safetensors';         Size = [long]3479416193; Dir = 'text_encoders' }
  }
}

$Files = @()
foreach ($k in @('DiT', 'TE')) {
  $e = $Variants[$Variant][$k]
  $Files += @{ Name = (Split-Path $e.Path -Leaf); Dir = $e.Dir; Size = $e.Size; Url = ($MS + $e.Path) }
}
$Files += @{
  Name = 'ae.safetensors'; Dir = 'vae'; Size = [long]335304388
  Url  = ($MS + 'split_files/vae/ae.safetensors')
}

if ($IncludeControlNet) {
  $Files += @{
    Name = 'Z-Image-Turbo-Fun-Controlnet-Union.safetensors'; Dir = 'model_patches'; Size = [long]3101572408
    Url  = 'https://huggingface.co/alibaba-pai/Z-Image-Turbo-Fun-Controlnet-Union/resolve/main/Z-Image-Turbo-Fun-Controlnet-Union.safetensors'
  }
}

function Get-Size([string]$p) {
  if (-not (Test-Path $p)) { return [long]0 }
  return ([System.IO.FileInfo]::new($p)).Length
}

Write-Host "variant: $Variant" -ForegroundColor Cyan

foreach ($f in $Files) {
  $dir = Join-Path $Target $f.Dir
  if (-not (Test-Path $dir)) { New-Item -ItemType Directory -Path $dir | Out-Null }
  $out = Join-Path $dir $f.Name

  $have = Get-Size $out
  if ($have -eq $f.Size) {
    Write-Host "[skip] $($f.Name) already complete ($have bytes)" -ForegroundColor DarkGray
    continue
  }
  if ($have -gt 0 -and $SkipExisting) {
    Write-Host "[skip] $($f.Name) present but incomplete ($have / $($f.Size))" -ForegroundColor Yellow
    continue
  }

  Write-Host "[get ] $($f.Name)  $($f.Size) bytes" -ForegroundColor Cyan
  & curl.exe -L -C - --retry 10 --retry-delay 5 --retry-all-errors -o $out $f.Url
  if ($LASTEXITCODE -ne 0) {
    Write-Host "[FAIL] curl exited $LASTEXITCODE for $($f.Name)" -ForegroundColor Red
    exit 1
  }

  $have = Get-Size $out
  if ($have -ne $f.Size) {
    Write-Host "[FAIL] $($f.Name): got $have bytes, expected $($f.Size)" -ForegroundColor Red
    Write-Host "       rerun this script; curl will resume from where it stopped." -ForegroundColor Yellow
    exit 1
  }
  Write-Host "[ok  ] $($f.Name) verified ($have bytes)" -ForegroundColor Green
}

Write-Host ''
Write-Host "All weights present and verified ($Variant)." -ForegroundColor Green
Write-Host 'Reference image for the ControlNet workflow:'
Write-Host '  https://huggingface.co/alibaba-pai/Z-Image-Turbo-Fun-Controlnet-Union/resolve/main/asset/canny.jpg'
