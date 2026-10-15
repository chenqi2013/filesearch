param([string]$InputPath, [string]$OutputPath)
[Console]::OutputEncoding = New-Object System.Text.UTF8Encoding($false)
$OutputEncoding = [Console]::OutputEncoding
$ErrorActionPreference = "Stop"
$application = $null
$presentation = $null
try {
    $application = New-Object -ComObject PowerPoint.Application
    $application.DisplayAlerts = 1
    $presentation = $application.Presentations.Open($InputPath, $true, $false, $false)
    $presentation.SaveAs($OutputPath, 24)
} catch {
    [Console]::Error.WriteLine("{0} (HRESULT: 0x{1})", $_.Exception.Message, $_.Exception.HResult.ToString("X8"))
    exit 1
} finally {
    if ($null -ne $presentation) { try { $presentation.Close() } catch {} }
    if ($null -ne $application) { try { $application.Quit() } catch {} }
}
