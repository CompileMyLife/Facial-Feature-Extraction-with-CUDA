# run_with_timeout.ps1

# Start compute-sanitizer with your arguments, and get the process object.
$proc = Start-Process -FilePath "compute-sanitizer" `
                      -ArgumentList "--tool memcheck --print-limit=1 --launch-timeout=60 .\FacialFeatureExtractionWithCUDA.exe" `
                      -PassThru

# Wait for up to 45 seconds (45000 milliseconds) for the process to exit.
if (-not $proc.WaitForExit(45000)) {
    Write-Host "Process did not complete in 45 seconds; killing it."
    $proc.Kill()
} else {
    Write-Host "Process completed within 45 seconds."
}
