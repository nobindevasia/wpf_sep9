param(
    [Parameter(Mandatory=$true)]
    [string]$Path,
    [switch]$NoXaml
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

function Remove-CSharpComments {
    param([string]$code)

    $sb = [System.Text.StringBuilder]::new()
    $i = 0
    $len = $code.Length

    $STATE_DEFAULT = 0
    $STATE_LINE = 1
    $STATE_BLOCK = 2
    $STATE_STRING = 3
    $STATE_VERBATIM = 4
    $STATE_CHAR = 5

    $state = $STATE_DEFAULT

    while ($i -lt $len) {
        $c = $code[$i]
        $n = if ($i + 1 -lt $len) { $code[$i+1] } else { [char]0 }
        $p = if ($i - 1 -ge 0) { $code[$i-1] } else { [char]0 }

        switch ($state) {
            $STATE_DEFAULT {
                # Detect comment starts
                if ($c -eq '/' -and $n -eq '/') {
                    $state = $STATE_LINE
                    $i += 2
                    continue
                }
                if ($c -eq '/' -and $n -eq '*') {
                    $state = $STATE_BLOCK
                    $i += 2
                    continue
                }

                # String/char starts (handle @$" and $@" and @" and $" )
                if ($c -eq '@' -and $n -eq '"') {
                    [void]$sb.Append($c)
                    [void]$sb.Append($n)
                    $i += 2
                    $state = $STATE_VERBATIM
                    continue
                }
                if ($c -eq '$' -and $n -eq '@' -and ($i + 2 -lt $len) -and $code[$i+2] -eq '"') {
                    [void]$sb.Append($c)
                    [void]$sb.Append($n)
                    [void]$sb.Append('"')
                    $i += 3
                    $state = $STATE_VERBATIM
                    continue
                }
                if ($c -eq '@' -and $n -eq '$' -and ($i + 2 -lt $len) -and $code[$i+2] -eq '"') {
                    [void]$sb.Append($c)
                    [void]$sb.Append($n)
                    [void]$sb.Append('"')
                    $i += 3
                    $state = $STATE_VERBATIM
                    continue
                }
                if ($c -eq '$' -and $n -eq '"') {
                    [void]$sb.Append($c)
                    [void]$sb.Append($n)
                    $i += 2
                    $state = $STATE_STRING
                    continue
                }
                if ($c -eq '"') {
                    [void]$sb.Append($c)
                    $i += 1
                    $state = $STATE_STRING
                    continue
                }
                if ($c -eq [char]39) {
                    [void]$sb.Append($c)
                    $i += 1
                    $state = $STATE_CHAR
                    continue
                }

                [void]$sb.Append($c)
                $i += 1
            }

            $STATE_LINE {
                # Skip until newline; keep the newline
                if ($c -eq "`n") {
                    [void]$sb.Append($c)
                    $state = $STATE_DEFAULT
                } elseif ($c -eq "`r" -and $n -eq "`n") {
                    [void]$sb.Append($c)
                } elseif ($c -eq "`r") {
                    [void]$sb.Append($c)
                    $state = $STATE_DEFAULT
                }
                $i += 1
            }

            $STATE_BLOCK {
                # Skip until */
                if ($c -eq '*' -and $n -eq '/') {
                    $i += 2
                    $state = $STATE_DEFAULT
                } else {
                    $i += 1
                }
            }

            $STATE_STRING {
                [void]$sb.Append($c)
                if ($c -eq '\') {
                    if ($i + 1 -lt $len) {
                        [void]$sb.Append($code[$i+1])
                        $i += 2
                        continue
                    }
                } elseif ($c -eq '"') {
                    $state = $STATE_DEFAULT
                }
                $i += 1
            }

            $STATE_VERBATIM {
                [void]$sb.Append($c)
                if ($c -eq '"') {
                    if ($n -eq '"') {
                        [void]$sb.Append($n)
                        $i += 2
                        continue
                    } else {
                        $state = $STATE_DEFAULT
                    }
                }
                $i += 1
            }

            $STATE_CHAR {
                [void]$sb.Append($c)
                if ($c -eq '\') {
                    if ($i + 1 -lt $len) {
                        [void]$sb.Append($code[$i+1])
                        $i += 2
                        continue
                    }
                } elseif ($c -eq [char]39) {
                    $state = $STATE_DEFAULT
                }
                $i += 1
            }
        }
    }

    return $sb.ToString()
}

function Remove-XamlComments {
    param([string]$code)
    # Remove XML comments <!-- ... --> across lines
    return [System.Text.RegularExpressions.Regex]::Replace($code, '<!--[\s\S]*?-->', '', 'Singleline')
}

function Process-File {
    param([string]$file)
    $ext = [System.IO.Path]::GetExtension($file).ToLowerInvariant()
    $content = [System.IO.File]::ReadAllText($file)
    $new = $null
    if ($ext -eq '.cs') {
        $new = Remove-CSharpComments -code $content
    } elseif ($ext -eq '.xaml') {
        $new = Remove-XamlComments -code $content
    } else {
        return $false
    }
    if ($new -ne $content) {
        [System.IO.File]::WriteAllText($file, $new)
        return $true
    }
    return $false
}

if (-not (Test-Path -Path $Path)) {
    throw "Path not found: $Path"
}

$changed = 0
$scanned = 0

# Build include patterns based on flags
$patterns = @('*.cs')
if (-not $NoXaml) { $patterns += '*.xaml' }

Get-ChildItem -Path $Path -Recurse -File -Include $patterns | Where-Object { $_.FullName -notmatch '\\obj\\|\\bin\\' } | ForEach-Object {
    $scanned++
    if (Process-File -file $_.FullName) { $changed++ }
}

Write-Output "Scanned: $scanned | Changed: $changed"
