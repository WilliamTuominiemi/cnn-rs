$location = Get-Location
$targetPath = Join-Path -Path $location -ChildPath "\images\raw_images\"
$folders = Get-ChildItem -Path $targetPath

foreach ($folder in $folders)
{
    $folderTargetPath = Join-Path -Path $targetPath -ChildPath $folder
    $files = Get-ChildItem -Path $folderTargetPath
    $i = 0
    foreach ($file in $files) {
        $newName = "$i.png"
        Rename-Item -Path $file.FullName -NewName $newName -ErrorAction SilentlyContinue
        $i++
    }
}