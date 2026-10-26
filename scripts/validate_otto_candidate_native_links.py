"""Candidate replacement closure; historical qualification remains unchanged.

System shared-cache libraries are recorded as OS dependencies. This verifies
Mach-O load commands, not every possible runtime dlopen or clean installation.
"""
import argparse
import hashlib
import importlib.metadata as metadata
import json
from pathlib import Path
import platform
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1]


def commands(path):
    lines=subprocess.check_output(['otool','-l',str(path)],text=True).splitlines()
    kind=None;links=[];rpaths=[]
    for line in lines:
        parts=line.strip().split(maxsplit=1)
        if not parts:continue
        if parts[0]=='cmd' and len(parts)==2:kind=parts[1]
        elif parts[0]=='name' and kind in ('LC_LOAD_DYLIB','LC_LOAD_WEAK_DYLIB','LC_REEXPORT_DYLIB','LC_LOAD_UPWARD_DYLIB'):
            links.append(parts[1].split(' (offset ')[0])
        elif parts[0]=='path' and kind=='LC_RPATH':rpaths.append(parts[1].split(' (offset ')[0])
    return links,rpaths


def resolve_link(link,loader,executable,rpaths):
    if link.startswith(('/usr/lib/','/System/Library/')):return None
    def expand(value):
        return value.replace('@loader_path',str(loader.parent)).replace('@executable_path',str(executable.parent))
    if link.startswith('@rpath/'):
        candidates=[Path(expand(root))/link[len('@rpath/'):] for root in rpaths]
    else:candidates=[Path(expand(link))]
    for candidate in candidates:
        if candidate.is_absolute() and candidate.is_file():return candidate.resolve()
    raise ValueError('Unresolved native link: '+link+' from '+loader.name)


def main():
    if platform.system()!='Darwin':raise ValueError('macOS-only link qualification')
    parser=argparse.ArgumentParser()
    for name in ('r-home','r-library','java-home','libfm'):parser.add_argument('--'+name,required=True,type=Path)
    cli=parser.parse_args()
    inventory=json.loads((ROOT/'docs/reviews/competition_otto_environment_inventory.json').read_text())
    roots=[]
    pyexe=Path(sys.executable).resolve();roots.append((pyexe,pyexe))
    for name,version in inventory['dependency_versions'].items():
        dist=metadata.distribution(name)
        if dist.version!=version:raise ValueError('Python dependency version changed')
        roots.extend((Path(dist.locate_file(f)).resolve(),pyexe) for f in dist.files or [] if f.suffix in ('.so','.dylib'))
    rexe=(cli.r_home/'bin/exec/R').resolve()
    if not rexe.is_file():rexe=(cli.r_home/'exec/R').resolve()
    roots.append((rexe,rexe))
    for directory in (cli.r_home/'lib',cli.r_library/'randomForest/libs',cli.r_library/'RSofia/libs',cli.r_library/'Rcpp/libs'):
        roots.extend((p.resolve(),rexe) for p in directory.rglob('*') if p.is_file() and p.suffix in ('.so','.dylib'))
    java=(cli.java_home/'bin/java').resolve();roots.append((java,java))
    roots.extend((p.resolve(),java) for p in (cli.java_home/'lib').rglob('*.dylib'))
    fm=cli.libfm.resolve();roots.append((fm,fm))
    records={};system=set();visited=set()
    def visit(path,executable,inherited=()):
        key=(str(path),str(executable),inherited)
        if key in visited:return
        visited.add(key)
        links,local=commands(path)
        expanded=tuple(p.replace('@loader_path',str(path.parent)).replace('@executable_path',str(executable.parent)) for p in local)
        search=tuple(dict.fromkeys(expanded+inherited))
        resolved=[]
        for link in links:
            target=resolve_link(link,path,executable,search)
            if target is None:system.add(link);resolved.append(dict(link=link,system=True))
            else:
                resolved.append(dict(link=link,target=str(target)))
                visit(target,executable,search)
        records[str(path)]=dict(sha256=hashlib.sha256(path.read_bytes()).hexdigest(),links=resolved)
    for path,executable in roots:
        _,rpaths=commands(executable)
        search=tuple(p.replace('@loader_path',str(executable.parent)).replace('@executable_path',str(executable.parent)) for p in rpaths)
        visit(path,executable,search)
    # These are software-installation locations only, never runtime input paths.
    report=dict(status='passed',qualification='candidate_links_only',approved=False,catalog_mutations=0,platform=platform.platform(),
        roots=len(set(str(p) for p,e in roots)),native_files=records,system_libraries=sorted(system),
        inventory_sha256=hashlib.sha256((ROOT/'docs/reviews/competition_otto_environment_inventory.json').read_bytes()).hexdigest(),
        verifier_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        historical_evidence_sha256=hashlib.sha256((ROOT/'docs/reviews/competition_otto_native_links.json').read_bytes()).hexdigest(),
        scope='Provisioned algorithm Python extensions, native R, Java and libFM link resolution; dynamic loading and clean installation not certified.')
    (ROOT/'docs/reviews/competition_otto_candidate_native_links.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(status='passed',roots=report['roots'],native_files=len(records),system_libraries=len(system))))


if __name__=='__main__':main()
