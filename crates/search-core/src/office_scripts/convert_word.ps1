param([string]$InputPath, [string]$OutputPath)
[Console]::OutputEncoding = New-Object System.Text.UTF8Encoding($false)
$OutputEncoding = [Console]::OutputEncoding
$ErrorActionPreference = "Stop"
$application = $null
$document = $null
try {
    $application = New-Object -ComObject Word.Application
    $application.Visible = $false
    $application.DisplayAlerts = 0
    $document = $application.Documents.Open($InputPath, $false, $true)
    $document.SaveAs2($OutputPath, 16)
} catch {
    [Console]::Error.WriteLine("{0} (HRESULT: 0x{1})", $_.Exception.Message, $_.Exception.HResult.ToString("X8"))
    exit 1
} finally {
    if ($null -ne $document) { $document.Close($false) }
    if ($null -ne $application) { $application.Quit() }
}
