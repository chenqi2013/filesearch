!macro NSIS_HOOK_PREINSTALL
  nsExec::ExecToLog '"$SYSDIR\taskkill.exe" /F /T /IM filesearch-desktop.exe'
  nsExec::ExecToLog '"$SYSDIR\taskkill.exe" /F /T /IM search-core.exe'
  Sleep 500
!macroend
