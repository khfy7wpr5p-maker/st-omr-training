# ST-OMR METER V5-0 PACKAGE_AB MATERIALIZER -- Windows 7 / PowerShell 2 compatible
#
# PURPOSE
#   Materialize the preregistered package_ab-only 500/500/500 selection into a
#   fresh staging directory after full preflight validation.
#
# SAFETY
#   - Never writes to TEST\METER_V2_1500.
#   - Refuses existing final or .partial output directories.
#   - Validates exact manifest and historical-blacklist SHA-256 before writing.
#   - Validates 500/class, 400/50/50, 1500 globally unique families, zero
#     historical overlap, and package_ab provenance before writing.
#   - Preflights every source PNG/semantic/agnostic/MEI before creating output.
#   - Copies only; source files are never moved, renamed, deleted, or edited.
#   - Verifies SHA-256 source == copied destination for every copied artifact.
#   - Writes bbox_status=NOT_ANNOTATED. No bbox, training, checkpoint, or model.

param(
    [string]$ManifestRoot = "",
    [string]$OutputRoot = "D:\veri eğitim seti\ST_OMR_WORKSPACE\V5_0_METER_V2_1500_PACKAGE_AB_STAGING"
)

$ErrorActionPreference = "Stop"

$EXPECTED_2_4_SHA = "d07ca3d0f7104ac1e5ed551886d80f5971da50b19ef65345c1fd6fa5ebbfb38e"
$EXPECTED_3_4_SHA = "5509bed3ba11dccbed7c277e90fb5e39e9ae6890bb7f460f0f24e41bb16bf2e8"
$EXPECTED_4_4_SHA = "cb8d036d1f0629eb6a14dbd57c887a5cec0d405d0e668ca403af8901080adc22"
$EXPECTED_BLACKLIST_SHA = "3231134495c3993b9d0d17355c8758bff2b879513289baad62d3dec03b641fc9"
$EXPECTED_TOTAL = 1500
$EXPECTED_PER_CLASS = 500
$EXPECTED_TRAIN = 400
$EXPECTED_VAL = 50
$EXPECTED_HOLDOUT = 50

function Get-Sha256File {
    param([string]$Path)
    $sha = New-Object System.Security.Cryptography.SHA256Managed
    $stream = [System.IO.File]::OpenRead($Path)
    try {
        $hashBytes = $sha.ComputeHash($stream)
        return ([BitConverter]::ToString($hashBytes)).Replace("-", "").ToLowerInvariant()
    }
    finally {
        $stream.Close()
        try { $sha.Clear() } catch {}
    }
}

function Assert-ExactSha {
    param([string]$Path, [string]$Expected, [string]$Label)
    if(-not (Test-Path -Path $Path -PathType Leaf)) {
        throw ($Label + " bulunamadi: " + $Path)
    }
    $observed = Get-Sha256File $Path
    if($observed -ne $Expected) {
        throw ($Label + " SHA256 FAIL expected=" + $Expected + " observed=" + $observed)
    }
    Write-Host ($Label + " SHA256 PASS " + $observed)
}

function Assert-PackageAbPath {
    param([string]$Path, [string]$Context)
    if([string]::IsNullOrEmpty($Path)) {
        throw ($Context + " bos source path")
    }
    $normalized = $Path.Replace("/", "\").ToLowerInvariant()
    if($normalized.IndexOf("\package_ab\") -lt 0) {
        throw ($Context + " package_ab provenance FAIL: " + $Path)
    }
}

function Assert-SafeOutputPath {
    param([string]$Path)
    if([string]::IsNullOrEmpty($Path)) { throw "OutputRoot bos olamaz" }
    $full = [System.IO.Path]::GetFullPath($Path)
    $lower = $full.Replace("/", "\").ToLowerInvariant()
    if($lower.IndexOf("\test\meter_v2_1500") -ge 0) {
        throw ("Mevcut TEST\\METER_V2_1500 icine yazmak yasak: " + $full)
    }
    return $full
}

if([string]::IsNullOrEmpty($ManifestRoot)) {
    $ManifestRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
}
$ManifestRoot = [System.IO.Path]::GetFullPath($ManifestRoot)
$OutputRoot = Assert-SafeOutputPath $OutputRoot
$PartialRoot = $OutputRoot + ".partial"

if(Test-Path -Path $OutputRoot) {
    throw ("Final staging zaten var; uzerine yazilmadi: " + $OutputRoot)
}
if(Test-Path -Path $PartialRoot) {
    throw ("Partial staging zaten var; sessiz rerun yasak: " + $PartialRoot)
}

$Manifest2 = Join-Path $ManifestRoot "2_4_SELECTION_MANIFEST.csv"
$Manifest3 = Join-Path $ManifestRoot "3_4_SELECTION_MANIFEST.csv"
$Manifest4 = Join-Path $ManifestRoot "4_4_SELECTION_MANIFEST.csv"
$BlacklistPath = Join-Path $ManifestRoot "METER_V1_HISTORICAL_FAMILIES.txt"

Assert-ExactSha $Manifest2 $EXPECTED_2_4_SHA "2/4 manifest"
Assert-ExactSha $Manifest3 $EXPECTED_3_4_SHA "3/4 manifest"
Assert-ExactSha $Manifest4 $EXPECTED_4_4_SHA "4/4 manifest"
Assert-ExactSha $BlacklistPath $EXPECTED_BLACKLIST_SHA "historical blacklist"

$blacklist = @{}
Get-Content -Path $BlacklistPath | ForEach-Object {
    $family = $_.Trim()
    if($family -ne "") { $blacklist[$family] = $true }
}
if($blacklist.Count -ne 325) {
    throw ("Historical blacklist cardinality FAIL: " + $blacklist.Count)
}

$specs = @(
    @{ Meter="2/4"; Path=$Manifest2 },
    @{ Meter="3/4"; Path=$Manifest3 },
    @{ Meter="4/4"; Path=$Manifest4 }
)

$allRows = @()
$globalFamilies = @{}
$globalFolders = @{}

foreach($spec in $specs) {
    $meter = $spec.Meter
    $rows = @(Import-Csv -Path $spec.Path)
    if($rows.Count -ne $EXPECTED_PER_CLASS) {
        throw ($meter + " count FAIL: " + $rows.Count)
    }

    if($rows.Count -gt 0) {
        $columns = @($rows[0].PSObject.Properties | ForEach-Object { $_.Name })
        $required = @("Split","Meter","FamilyId","SampleId","Folder","SourceImage","SourceSemantic","SourceAgnostic","SplitRank")
        foreach($column in $required) {
            if(-not ($columns -contains $column)) { throw ($meter + " schema missing " + $column) }
        }
        if($columns -contains "SelectionRank") { throw ($meter + " noncanonical SelectionRank forbidden") }
    }

    $trainCount = 0
    $valCount = 0
    $holdoutCount = 0
    $classFamilies = @{}

    foreach($row in $rows) {
        if($row.Meter -ne $meter) { throw ($meter + " label mismatch: " + $row.Meter) }
        if($row.Split -eq "train") { $trainCount++ }
        elseif($row.Split -eq "val") { $valCount++ }
        elseif($row.Split -eq "final_holdout") { $holdoutCount++ }
        else { throw ($meter + " invalid split: " + $row.Split) }

        if([string]::IsNullOrEmpty($row.FamilyId)) { throw ($meter + " empty FamilyId") }
        if([string]::IsNullOrEmpty($row.Folder)) { throw ($meter + " empty Folder") }
        if($classFamilies.ContainsKey($row.FamilyId)) { throw ($meter + " duplicate family: " + $row.FamilyId) }
        if($globalFamilies.ContainsKey($row.FamilyId)) { throw ("cross-class/split family leak: " + $row.FamilyId) }
        if($globalFolders.ContainsKey($row.Folder)) { throw ("duplicate folder: " + $row.Folder) }
        if($blacklist.ContainsKey($row.FamilyId)) { throw ("historical blacklist overlap: " + $row.FamilyId) }

        Assert-PackageAbPath $row.SourceImage ($row.FamilyId + " image")
        Assert-PackageAbPath $row.SourceSemantic ($row.FamilyId + " semantic")
        Assert-PackageAbPath $row.SourceAgnostic ($row.FamilyId + " agnostic")

        $sourceMei = [System.IO.Path]::ChangeExtension($row.SourceImage, ".mei")
        Assert-PackageAbPath $sourceMei ($row.FamilyId + " mei")

        foreach($source in @($row.SourceImage, $row.SourceSemantic, $row.SourceAgnostic, $sourceMei)) {
            if(-not (Test-Path -Path $source -PathType Leaf)) {
                throw ("Source file missing before write: " + $source)
            }
        }

        $classFamilies[$row.FamilyId] = $true
        $globalFamilies[$row.FamilyId] = $true
        $globalFolders[$row.Folder] = $true
        $row | Add-Member -MemberType NoteProperty -Name SourceMei -Value $sourceMei
        $allRows += $row
    }

    if($trainCount -ne $EXPECTED_TRAIN -or $valCount -ne $EXPECTED_VAL -or $holdoutCount -ne $EXPECTED_HOLDOUT) {
        throw ($meter + " split counts FAIL train=" + $trainCount + " val=" + $valCount + " holdout=" + $holdoutCount)
    }
    Write-Host ($meter + " PRECHECK PASS total=500 train=400 val=50 final_holdout=50")
}

if($allRows.Count -ne $EXPECTED_TOTAL) { throw ("Global row count FAIL: " + $allRows.Count) }
if($globalFamilies.Count -ne $EXPECTED_TOTAL) { throw ("Global unique-family FAIL: " + $globalFamilies.Count) }
if($globalFolders.Count -ne $EXPECTED_TOTAL) { throw ("Global unique-folder FAIL: " + $globalFolders.Count) }

Write-Host "PRECHECK=PASS rows=1500 unique_families=1500 package_ab_only=True blacklist_overlap=0"
Write-Host "No writes have occurred before this point."

# Only after complete preflight do we create a fresh partial staging tree.
New-Item -ItemType Directory -Path $PartialRoot | Out-Null

$hashRecords = @()
$copiedSamples = 0
$copiedArtifacts = 0

try {
    foreach($row in $allRows) {
        $meterDir = $row.Meter.Replace("/", "_")
        $destDir = Join-Path (Join-Path (Join-Path $PartialRoot $row.Split) $meterDir) $row.Folder
        New-Item -ItemType Directory -Path $destDir -Force | Out-Null

        $artifacts = @(
            @{ Kind="image"; Source=$row.SourceImage; Dest=(Join-Path $destDir "image.png") },
            @{ Kind="semantic"; Source=$row.SourceSemantic; Dest=(Join-Path $destDir "label.semantic") },
            @{ Kind="agnostic"; Source=$row.SourceAgnostic; Dest=(Join-Path $destDir "label.agnostic") },
            @{ Kind="mei"; Source=$row.SourceMei; Dest=(Join-Path $destDir "score.mei") }
        )

        foreach($artifact in $artifacts) {
            [System.IO.File]::Copy($artifact.Source, $artifact.Dest, $false)
            $sourceSha = Get-Sha256File $artifact.Source
            $destSha = Get-Sha256File $artifact.Dest
            if($sourceSha -ne $destSha) {
                throw ("Copy SHA mismatch family=" + $row.FamilyId + " kind=" + $artifact.Kind)
            }
            $hashRecords += New-Object PSObject -Property @{
                Split=$row.Split
                Meter=$row.Meter
                FamilyId=$row.FamilyId
                Folder=$row.Folder
                Artifact=$artifact.Kind
                SourcePath=$artifact.Source
                SourceSha256=$sourceSha
                DestinationPath=$artifact.Dest
                DestinationSha256=$destSha
            }
            $copiedArtifacts++
        }

        $metaLines = @(
            "meter=" + $row.Meter,
            "split=" + $row.Split,
            "family_id=" + $row.FamilyId,
            "sample_id=" + $row.SampleId,
            "source_image=" + $row.SourceImage,
            "source_semantic=" + $row.SourceSemantic,
            "source_agnostic=" + $row.SourceAgnostic,
            "source_mei=" + $row.SourceMei,
            "bbox_status=NOT_ANNOTATED"
        )
        $metaLines | Out-File -FilePath (Join-Path $destDir "meta.txt") -Encoding UTF8
        $copiedSamples++
        if(($copiedSamples % 100) -eq 0) {
            Write-Host ("Copied and hash-verified samples: " + $copiedSamples + "/1500")
        }
    }

    $hashCsv = Join-Path $PartialRoot "FILE_HASHES.csv"
    $hashRecords |
        Sort-Object Split,Meter,FamilyId,Artifact |
        Select-Object Split,Meter,FamilyId,Folder,Artifact,SourcePath,SourceSha256,DestinationPath,DestinationSha256 |
        Export-Csv -Path $hashCsv -NoTypeInformation -Encoding UTF8

    $receipt = @(
        "schema=st-omr-meter-v5-0-package-ab-materialization-v1",
        "status=MATERIALIZATION_COMPLETE",
        "source_domain=package_ab",
        "selected_samples=" + $copiedSamples,
        "copied_hash_verified_artifacts=" + $copiedArtifacts,
        "global_unique_families=" + $globalFamilies.Count,
        "historical_blacklist_overlap=0",
        "cross_class_family_overlap=0",
        "cross_split_family_overlap=0",
        "class_2_4=500",
        "class_3_4=500",
        "class_4_4=500",
        "per_class_train=400",
        "per_class_val=50",
        "per_class_final_holdout=50",
        "manifest_2_4_sha256=" + $EXPECTED_2_4_SHA,
        "manifest_3_4_sha256=" + $EXPECTED_3_4_SHA,
        "manifest_4_4_sha256=" + $EXPECTED_4_4_SHA,
        "historical_blacklist_sha256=" + $EXPECTED_BLACKLIST_SHA,
        "bbox_annotation_authorized=false",
        "training_authorized=false",
        "checkpoint_opened=false",
        "model_evaluated=false",
        "inference_count=0",
        "existing_meter_v2_1500_mutated=false"
    )
    $receipt | Out-File -FilePath (Join-Path $PartialRoot "MATERIALIZATION_RECEIPT.txt") -Encoding UTF8
    "COMPLETE" | Out-File -FilePath (Join-Path $PartialRoot "COMPLETE") -Encoding ASCII

    Move-Item -Path $PartialRoot -Destination $OutputRoot
}
catch {
    Write-Host "MATERIALIZATION FAILED. Partial staging intentionally preserved for forensic inspection." -ForegroundColor Red
    Write-Host ("PARTIAL=" + $PartialRoot) -ForegroundColor Red
    throw
}

Write-Host ""
Write-Host "=================================================="
Write-Host "METER V5-0 PACKAGE_AB MATERIALIZATION COMPLETE"
Write-Host "=================================================="
Write-Host "samples=1500 artifacts_hash_verified=6000"
Write-Host "source=package_ab only"
Write-Host "families=1500 unique; blacklist_overlap=0"
Write-Host "bbox_annotation_authorized=false"
Write-Host "training_authorized=false"
Write-Host ("OUTPUT=" + $OutputRoot)
