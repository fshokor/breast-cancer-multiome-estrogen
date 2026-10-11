from pathlib import Path
import urllib.request,hashlib,json,concurrent.futures
ROOT=Path(__file__).resolve().parents[1]
DEST=ROOT/"data/raw/reference/nb07";DEST.mkdir(parents=True,exist_ok=True)
urls={
"hg38.2bit":"https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/hg38.2bit",
"hg38_md5sum.txt":"https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/md5sum.txt",
"JASPAR2024_CORE_vertebrates_non-redundant_pfms_jaspar.txt":"https://jaspar2024.elixir.no/download/data/2024/CORE/JASPAR2024_CORE_vertebrates_non-redundant_pfms_jaspar.txt",
"hg38-blacklist.v2.bed.gz":"https://raw.githubusercontent.com/Boyle-Lab/Blacklist/master/lists/hg38-blacklist.v2.bed.gz"}
def fetch(item):
 name,url=item;path=DEST/name
 if not path.exists():
  tmp=path.with_name(name+".partial")
  with urllib.request.urlopen(url,timeout=120) as response,tmp.open("wb") as f:
   while block:=response.read(1048576):f.write(block)
  tmp.replace(path)
 h=hashlib.sha256();md5=hashlib.md5()
 with path.open("rb") as f:
  while block:=f.read(1048576):h.update(block);md5.update(block)
 print(name,path.stat().st_size,h.hexdigest(),flush=True)
 return name,dict(url=url,path=str(path.relative_to(ROOT)),size=path.stat().st_size,sha256=h.hexdigest(),md5=md5.hexdigest())
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex: records=dict(ex.map(fetch,urls.items()))
checks=(DEST/"hg38_md5sum.txt").read_text().splitlines()
expected=[x.split()[0] for x in checks if x.split()[-1].lstrip("*")=="hg38.2bit"][0]
assert records["hg38.2bit"]["md5"]==expected,"UCSC genome checksum mismatch"
manifest={"retrieved_date":"2026-10-11","assembly":"hg38","motif_release":"JASPAR2024 CORE vertebrates non-redundant",
          "genome_official_MD5_verified":True,"resources":records}
(ROOT/"configs/motif_references.json").write_text(json.dumps(manifest,indent=2)+"\n")
print("All references verified.",flush=True)
