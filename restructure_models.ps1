# Restructure model files: move from cpu/gpu subdirectories to flat structure
$modelDir = "$PSScriptRoot\artifacts\models"

# Copy from cpu/ to root
Copy-Item "$modelDir\cpu\mnist_model.pt" "$modelDir\mnist_model.pt" -Force
Copy-Item "$modelDir\cpu\model.pth" "$modelDir\model.pth" -Force

# Remove old subdirectories
Remove-Item "$modelDir\cpu" -Recurse -Force
Remove-Item "$modelDir\gpu" -Recurse -Force

Write-Host "Model files restructured. Contents of $modelDir :"
Get-ChildItem "$modelDir"
