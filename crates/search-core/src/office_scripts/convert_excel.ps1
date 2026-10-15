param([string]$InputPath, [string]$OutputPath)
[Console]::OutputEncoding = New-Object System.Text.UTF8Encoding($false)
$OutputEncoding = [Console]::OutputEncoding
$ErrorActionPreference = "Stop"
$application = $null
$workbook = $null
try {
    $application = New-Object -ComObject Excel.Application
    $application.Visible = $false
    $application.DisplayAlerts = $false
    $workbook = $application.Workbooks.Open($InputPath, 0, $true)
    $workbook.SaveAs($OutputPath, 51)
} catch {
    [Console]::Error.WriteLine("{0} (HRESULT: 0x{1})", $_.Exception.Message, $_.Exception.HResult.ToString("X8"))
    exit 1
} finally {
    if ($null -ne $workbook) { try { $workbook.Close($false) } catch {} }
    if ($null -ne $application) { try { $application.Quit() } catch {} }
}
