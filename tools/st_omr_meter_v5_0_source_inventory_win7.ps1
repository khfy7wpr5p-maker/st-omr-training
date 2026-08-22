# ST-OMR METER V5-0 SOURCE INVENTORY — READ ONLY
# Windows 7 / Windows PowerShell compatible.
#
# PURPOSE:
#   Measure whether package_aa/package_ab contain enough independent 2/4, 3/4,
#   and 4/4 candidates to repair the next dataset without source-domain shortcut.
#
# SAFETY:
#   - Reads MASTER_INDEX.tsv and only the referenced semantic/agnostic files.
#   - Does not scan the original PrIMuS tree.
#   - Does not copy, move, delete, link, crop, annotate, train, infer, or open models.
#   - Writes only a fresh isolated inventory directory.
#   - Existing dataset directories are never touched.

$ErrorActionPreference = "Stop"

$indexPath = "D:\veri eğitim seti\ST_OMR_PRIMUS_INDEX\MASTER_INDEX.tsv"
$outRoot = "D:\veri eğitim seti\ST_OMR_WORKSPACE\V5_0_SOURCE_INVENTORY"

$expectedConsumedSelectionSha = "4335a48a091912ba422c16d8fcbaaa7bbf5f7a0a43f088146a50a3e02e3ed7dc"
$expectedConsumedFamilyListSha = "9d969b6bf5749bae7003c45644c50be36495ccb9b10fe3e7569ace5d413adea3"

if(-not (Test-Path -LiteralPath $indexPath)) {
    throw "MASTER_INDEX bulunamadi: $indexPath"
}
if(Test-Path -LiteralPath $outRoot) {
    throw "V5-0 inventory klasoru zaten var; guvenlik icin uzerine yazilmadi: $outRoot"
}

# Exact family identities from the consumed one-shot V4-5 holdout.
# Source selection SHA is frozen above; this list is independently SHA-bound below.
$consumedBlock = @"
aa_000100726
aa_000102820
aa_000103040
aa_000103319
aa_000104444
aa_000105834
aa_000107292
aa_000107518
aa_000107568
aa_000107624
aa_000108391
aa_000109374
aa_000109947
aa_000110983
aa_000112419
aa_000113691
aa_000113692
aa_000113693
aa_000113876
aa_000114053
aa_000115543
aa_000115681
aa_000115783
aa_000115864
aa_000115979
aa_000116286
aa_000116587
aa_000117527
aa_000118024
aa_000119041
aa_000119725
aa_000121640
aa_000121664
aa_000124205
aa_000124454
aa_000124951
aa_000125443
aa_000125491
aa_000126896
aa_000127148
aa_000127207
aa_000127596
aa_000127709
aa_000128351
aa_000130430
aa_000130441
aa_000132301
aa_000132902
aa_000135324
aa_000135388
aa_000135517
aa_000135689
aa_000135935
aa_000136124
aa_000137747
aa_000138534
aa_000141240
aa_110002808
ab_120000111
ab_150201054
ab_150201207
ab_150201301
ab_150201401
ab_150201419
ab_150201426
ab_150201531
ab_150201708
ab_150201714
ab_150202242
ab_150202285
ab_150202679
ab_150202795
ab_150202961
ab_150203078
ab_150203118
ab_150203239
ab_150203472
ab_150203681
ab_150203693
ab_150203700
ab_150203713
ab_150203762
ab_150203997
ab_150204002
ab_150204045
ab_150204071
ab_150204104
ab_150204156
ab_150204261
ab_150204273
ab_150204717
ab_150204808
ab_150204837
ab_150204862
ab_150205041
ab_150205042
ab_150205456
ab_150206338
ab_150206391
ab_150206538
ab_150207139
ab_150230027
ab_150230081
ab_160000123
ab_160000140
ab_170000061
ab_190000631
ab_190005923
ab_190006225
ab_190012707
ab_190013491
ab_190013519
ab_190013640
ab_190020044
ab_190020798
ab_190022133
ab_200231864
ab_201001771
ab_201005342
ab_201007184
ab_201008702
ab_210000319
ab_210097092
ab_211000454
ab_211004617
ab_211010372
ab_211010388
ab_211010681
ab_212002996
ab_212003025
ab_220000459
ab_220000968
ab_220003007
ab_220003015
ab_220003032
ab_220003050
ab_220003053
ab_220003078
ab_220016267
ab_220017242
ab_220030962
ab_230000102
ab_230001004
ab_230001021
ab_230004467
ab_230004681
ab_230004963
ab_230005139
ab_230006147
ab_230006245
"@

function Get-StringSha256 {
    param([string]$Text)
    $sha = [System.Security.Cryptography.SHA256]::Create()
    try {
        $bytes = [System.Text.Encoding]::UTF8.GetBytes($Text)
        return ([BitConverter]::ToString($sha.ComputeHash($bytes))).Replace("-", "").ToLowerInvariant()
    }
    finally {
        $sha.Dispose()
    }
}

function Get-ContentKey {
    param([string]$SemanticPath,[string]$AgnosticPath)
    $sem = [System.IO.File]::ReadAllText($SemanticPath).Trim()
    $agn = [System.IO.File]::ReadAllText($AgnosticPath).Trim()
    return Get-StringSha256 ($sem + "`n---`n" + $agn)
}

function Test-SingleMeter {
    param([string]$SemanticPath,[string]$TargetLabel)
    $text = [System.IO.File]::ReadAllText($SemanticPath)
    $tokens = [regex]::Split($text.Trim(), "\s+")
    $meters = @($tokens | Where-Object { $_ -match "^timeSignature-" })
    if($meters.Count -eq 0) { return $false }
    foreach($m in $meters) {
        if($m -ne ("timeSignature-" + $TargetLabel)) { return $false }
    }
    return $true
}

function Test-AgnosticMeterPair {
    param([string]$AgnosticPath,[string]$TargetMeter)
    $parts = $TargetMeter.Split("/")
    if($parts.Count -ne 2) { return $false }
    $num = [regex]::Escape($parts[0])
    $den = [regex]::Escape($parts[1])
    $text = [System.IO.File]::ReadAllText($AgnosticPath)
    $pattern = "digit\." + $num + "-[^\s]+\s+digit\." + $den + "-[^\s]+"
    return [regex]::IsMatch($text, $pattern)
}

$consumedFamilies = @(
    $consumedBlock -split "\r?\n" |
    ForEach-Object { $_.Trim() } |
    Where-Object { $_ -ne "" } |
    Sort-Object -Unique
)
if($consumedFamilies.Count -ne 150) {
    throw "Consumed family cardinality FAIL: $($consumedFamilies.Count)"
}
$canonicalConsumed = (($consumedFamilies | Sort-Object) -join "`n") + "`n"
$observedConsumedFamilyListSha = Get-StringSha256 $canonicalConsumed
if($observedConsumedFamilyListSha -ne $expectedConsumedFamilyListSha) {
    throw "Consumed family SHA mismatch: $observedConsumedFamilyListSha"
}
$consumedSet = @{}
foreach($familyId in $consumedFamilies) { $consumedSet[$familyId] = $true }

$rows = Import-Csv -Delimiter "`t" -LiteralPath $indexPath
$configs = @()
$configs += New-Object PSObject -Property @{ Key="2_4"; Label="2/4"; Flag="Meter2_4" }
$configs += New-Object PSObject -Property @{ Key="3_4"; Label="3/4"; Flag="Meter3_4" }
$configs += New-Object PSObject -Property @{ Key="4_4"; Label="4/4"; Flag="Meter4_4" }

$candidates = @()
$excludedConsumed = 0
$missingReferencedFiles = 0

foreach($cfg in $configs) {
    $flagName = $cfg.Flag
    foreach($r in $rows) {
        if(($r.Package -ne "aa") -and ($r.Package -ne "ab")) { continue }
        if(($r.Complete -ne "1") -or ($r.ClefG2 -ne "1") -or ($r.$flagName -ne "1")) { continue }
        if((-not (Test-Path -LiteralPath $r.PNG)) -or
           (-not (Test-Path -LiteralPath $r.Semantic)) -or
           (-not (Test-Path -LiteralPath $r.Agnostic))) {
            $missingReferencedFiles++
            continue
        }
        if(-not (Test-SingleMeter $r.Semantic $cfg.Label)) { continue }
        if(-not (Test-AgnosticMeterPair $r.Agnostic $cfg.Label)) { continue }

        $familyId = $r.Package + "_" + $r.Family
        if($consumedSet.ContainsKey($familyId)) {
            $excludedConsumed++
            continue
        }

        $tier = "C"
        if($r.V1Strict -eq "1") { $tier = "A" }
        elseif($r.KeySignature -eq "0") { $tier = "B" }

        $contentKey = Get-ContentKey $r.Semantic $r.Agnostic
        $candidates += New-Object PSObject -Property @{
            Meter=$cfg.Label
            Package=("package_" + $r.Package)
            FamilyId=$familyId
            SampleId=$r.Sample
            Tier=$tier
            ContentKey=$contentKey
            SourceImage=$r.PNG
            SourceSemantic=$r.Semantic
            SourceAgnostic=$r.Agnostic
        }
    }
}

$summary = @()
foreach($cfg in $configs) {
    foreach($package in @("package_aa", "package_ab")) {
        $subset = @($candidates | Where-Object { ($_.Meter -eq $cfg.Label) -and ($_.Package -eq $package) })
        $familyCount = @($subset | Select-Object -ExpandProperty FamilyId | Sort-Object -Unique).Count
        $contentCount = @($subset | Select-Object -ExpandProperty ContentKey | Sort-Object -Unique).Count
        $capacity = [Math]::Min($familyCount, $contentCount)
        $summary += New-Object PSObject -Property @{
            Meter=$cfg.Label
            Package=$package
            RawCandidates=$subset.Count
            UniqueFamilies=$familyCount
            UniqueContent=$contentCount
            ConservativeCountCapacity=$capacity
        }
    }
}

$familyCollisions = @()
foreach($g in ($candidates | Group-Object FamilyId)) {
    $meters = @($g.Group | Select-Object -ExpandProperty Meter | Sort-Object -Unique)
    if($meters.Count -gt 1) {
        $familyCollisions += New-Object PSObject -Property @{
            FamilyId=$g.Name
            Meters=($meters -join "|")
            CandidateRows=$g.Count
        }
    }
}

$contentCollisions = @()
foreach($g in ($candidates | Group-Object ContentKey)) {
    $meters = @($g.Group | Select-Object -ExpandProperty Meter | Sort-Object -Unique)
    if($meters.Count -gt 1) {
        $contentCollisions += New-Object PSObject -Property @{
            ContentKey=$g.Name
            Meters=($meters -join "|")
            CandidateRows=$g.Count
        }
    }
}

$aaCaps = @($summary | Where-Object { $_.Package -eq "package_aa" } | Select-Object -ExpandProperty ConservativeCountCapacity)
$abCaps = @($summary | Where-Object { $_.Package -eq "package_ab" } | Select-Object -ExpandProperty ConservativeCountCapacity)
if(($aaCaps.Count -ne 3) -or ($abCaps.Count -ne 3)) { throw "Capacity summary cardinality FAIL" }
$minAA = ($aaCaps | Measure-Object -Minimum).Minimum
$minAB = ($abCaps | Measure-Object -Minimum).Minimum
$lowAA = [Math]::Max(0, 500 - $minAB)
$highAA = [Math]::Min(500, $minAA)
$countOnlyFeasible = ($lowAA -le $highAA)
$recommendedAA = -1
$recommendedAB = -1
if($countOnlyFeasible) {
    $recommendedAA = [Math]::Max($lowAA, [Math]::Min(250, $highAA))
    $recommendedAB = 500 - $recommendedAA
}

New-Item -ItemType Directory -Path $outRoot | Out-Null

$candidates |
    Sort-Object Meter,Package,FamilyId,ContentKey,SampleId |
    Select-Object Meter,Package,FamilyId,SampleId,Tier,ContentKey,SourceImage,SourceSemantic,SourceAgnostic |
    Export-Csv -Path (Join-Path $outRoot "candidate_inventory.csv") -NoTypeInformation -Encoding UTF8

$summary |
    Sort-Object Meter,Package |
    Select-Object Meter,Package,RawCandidates,UniqueFamilies,UniqueContent,ConservativeCountCapacity |
    Export-Csv -Path (Join-Path $outRoot "source_summary.csv") -NoTypeInformation -Encoding UTF8

$familyCollisions |
    Sort-Object FamilyId |
    Select-Object FamilyId,Meters,CandidateRows |
    Export-Csv -Path (Join-Path $outRoot "cross_meter_family_collisions.csv") -NoTypeInformation -Encoding UTF8

$contentCollisions |
    Sort-Object ContentKey |
    Select-Object ContentKey,Meters,CandidateRows |
    Export-Csv -Path (Join-Path $outRoot "cross_meter_content_collisions.csv") -NoTypeInformation -Encoding UTF8

$receipt = @(
    "schema=st-omr-meter-v5-0-source-inventory-v1",
    "status=INVENTORY_COMPLETE",
    "source_index=" + $indexPath,
    "output_root=" + $outRoot,
    "consumed_selection_sha256=" + $expectedConsumedSelectionSha,
    "consumed_family_count=" + $consumedFamilies.Count,
    "consumed_family_list_sha256=" + $observedConsumedFamilyListSha,
    "candidate_rows=" + $candidates.Count,
    "excluded_consumed_rows=" + $excludedConsumed,
    "missing_referenced_files=" + $missingReferencedFiles,
    "cross_meter_family_collision_count=" + $familyCollisions.Count,
    "cross_meter_content_collision_count=" + $contentCollisions.Count,
    "count_only_common_mix_feasible=" + $countOnlyFeasible,
    "count_only_aa_low=" + $lowAA,
    "count_only_aa_high=" + $highAA,
    "count_only_recommended_aa=" + $recommendedAA,
    "count_only_recommended_ab=" + $recommendedAB,
    "count_only_warning=NOT_REBUILD_AUTHORIZATION_GLOBAL_FAMILY_AND_CONTENT_DISJOINTNESS_STILL_REQUIRED",
    "training_authorized=false",
    "bbox_annotation_authorized=false",
    "model_evaluated=false",
    "checkpoint_opened=false",
    "inference_count=0",
    "dataset_mutated=false"
)
$receipt | Out-File -FilePath (Join-Path $outRoot "SAFETY_RECEIPT.txt") -Encoding UTF8

Write-Host ""
Write-Host "=============================================="
Write-Host "METER V5-0 SOURCE INVENTORY TAMAMLANDI"
Write-Host "=============================================="
foreach($s in ($summary | Sort-Object Meter,Package)) {
    Write-Host ($s.Meter + " " + $s.Package +
        " raw=" + $s.RawCandidates +
        " families=" + $s.UniqueFamilies +
        " content=" + $s.UniqueContent +
        " capacity=" + $s.ConservativeCountCapacity)
}
Write-Host ("Consumed exclusions: " + $excludedConsumed)
Write-Host ("Cross-meter family collisions: " + $familyCollisions.Count)
Write-Host ("Cross-meter content collisions: " + $contentCollisions.Count)
Write-Host ("COUNT-ONLY common mix feasible: " + $countOnlyFeasible)
Write-Host ("AA interval: " + $lowAA + ".." + $highAA)
Write-Host ("Recommended count-only AA/AB: " + $recommendedAA + "/" + $recommendedAB)
Write-Host "Bu sonuc dataset rebuild izni DEGILDIR. Global disjoint selector sonraki gate'tir."
Write-Host ("OUTPUT: " + $outRoot)
