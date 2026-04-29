$ErrorActionPreference = "Stop"

$datasetRoot = "EuroSAT_RGB"
$outputRoot = "outputs"

python src\search.py `
  --dataset-root $datasetRoot `
  --output-root $outputRoot `
  --epochs 12 `
  --batch-size 64 `
  --lrs 0.01,0.005 `
  --weight-decays 0.0001,0.001 `
  --activations relu,tanh `
  --hidden-grid "256,128;512,256" `
  --lr-decay-gamma 0.5 `
  --lr-decay-step 6

Write-Host ""
Write-Host "请从 outputs\search\grid_search_results.json 中选取最佳 checkpoint，然后运行："
Write-Host "python src\test.py --dataset-root EuroSAT_RGB --checkpoint outputs\checkpoints\your_best_model.npz"
