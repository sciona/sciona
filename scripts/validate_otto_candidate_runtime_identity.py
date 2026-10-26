"""Verify candidate runtime imports; preserve historical approval evidence."""
import argparse
import hashlib
import importlib
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from packaging.version import Version
ROOT=Path(__file__).resolve().parents[1]


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser()
    for name in ('r-home','r-library','java-home','lasagne','theano','libfm','rsofia'):parser.add_argument('--'+name,required=True,type=Path)
    cli=parser.parse_args();review=ROOT/'docs/reviews'
    inventory_path=review/'competition_otto_environment_inventory.json'
    inventory=json.loads(inventory_path.read_text());links_path=review/'competition_otto_candidate_native_links.json'
    links=json.loads(links_path.read_text())
    if links['status']!='passed' or links['inventory_sha256']!=sha(inventory_path):raise ValueError('Native link evidence differs')
    for file,record in links['native_files'].items():
        if sha(Path(file))!=record['sha256']:raise ValueError('Native binary changed')
    for file,digest in inventory['license_sha256'].items():
        if sha(ROOT/'docs/licenses'/file)!=digest:raise ValueError('Retained notice changed')
    for name,files in inventory['source_code_sha256'].items():
        root=getattr(cli,name)
        for file,digest in files.items():
            if sha(root/file)!=digest:raise ValueError('Provisioned source changed')
    aliases={'scikit-learn':'sklearn','charset-normalizer':'charset_normalizer'}
    imported={}
    for name,version in inventory['dependency_versions'].items():
        module=importlib.import_module(aliases.get(name,name))
        actual=getattr(module,'__version__',None)
        dist=metadata.distribution(name)
        if dist.version!=version or actual is None or Version(actual)!=Version(version):raise ValueError('Imported version differs: '+name)
        expected=Path(dist.locate_file(aliases.get(name,name))).resolve()
        location=Path(module.__file__).resolve()
        if not (location.is_relative_to(expected) or location==expected.with_suffix('.py')):raise ValueError('Imported location differs')
        imported[name]=actual
    code="""import json,sys
from pathlib import Path
import lasagne,theano
assert Path(lasagne.__file__).resolve().is_relative_to(Path(sys.argv[1]).resolve())
assert Path(theano.__file__).resolve().is_relative_to(Path(sys.argv[2]).resolve())
assert lasagne.__version__=='0.2.dev1' and theano.__version__=='1.0.5'
assert theano.config.device=='cpu' and theano.config.floatX=='float32' and theano.config.cxx==''
print(json.dumps(dict(lasagne=lasagne.__version__,theano=theano.__version__,device=theano.config.device)))
"""
    with tempfile.TemporaryDirectory(prefix='otto-runtime-check-') as temporary:
        env=dict(os.environ,PYTHONPATH=os.pathsep.join(str(p.resolve()) for p in (cli.lasagne,cli.theano)),
            OMP_NUM_THREADS='1',THEANO_FLAGS='device=cpu,floatX=float32,cxx=,blas.ldflags=,base_compiledir='+temporary)
        result=subprocess.run([sys.executable,'-c',code,str(cli.lasagne),str(cli.theano)],env=env,capture_output=True,text=True,timeout=60,check=True)
        neural=json.loads(result.stdout)
    if sha(cli.libfm/'bin/libFM')!=inventory['libfm_binary_sha256']:raise ValueError('Configured libFM binary differs')
    if str((cli.java_home/'bin/java').resolve()) not in links['native_files']:raise ValueError('Java outside reviewed native roots')
    r_code='''args <- commandArgs(TRUE)
stopifnot(normalizePath(R.home()) == normalizePath(args[2]))
root <- normalizePath(args[1]); .libPaths(c(root,.libPaths()))
stopifnot(as.character(getRversion()) == "4.6.1")
for (name in c("randomForest","RSofia","Rcpp")) {
 loadNamespace(name,lib.loc=root)
 stopifnot(normalizePath(find.package(name,lib.loc=root)) == file.path(root,name))
 cat(name,packageDescription(name,lib.loc=root)$Version,"\\n")
}
'''
    output=subprocess.run([str(cli.r_home/'bin/Rscript'),'--vanilla','-e',r_code,str(cli.r_library),str(cli.r_home)],capture_output=True,text=True,timeout=60,check=True).stdout
    packages={parts[0]:parts[1] for line in output.splitlines() if len(parts:=line.split())==2}
    if packages!={n:r['Version'] for n,r in inventory['r_packages'].items()}:raise ValueError('Imported R versions differ')
    java=subprocess.run([str(cli.java_home/'bin/java'),'-XX:ActiveProcessorCount=1','-XX:+UseSerialGC','-version'],capture_output=True,text=True,timeout=30,check=True)
    release=dict(line.split('=',1) for line in inventory['java_release'].splitlines() if '=' in line)
    if release['JAVA_VERSION'].strip('"') not in (java.stdout+java.stderr).splitlines()[0]:raise ValueError('Java version differs')
    report=dict(status='passed',qualification='candidate_imports_only',approved=False,catalog_mutations=0,python_import_versions=imported,
        neural_imports=neural,r_import_versions=packages,r_version='4.6.1',java_version=release['JAVA_VERSION'].strip('"'),
        verified_native_files=len(links['native_files']),verified_notices=len(inventory['license_sha256']),
        sha256={str(p.relative_to(ROOT)):sha(p) for p in [inventory_path,links_path,Path(__file__).resolve(),ROOT/'requirements/otto-execution.txt']},
        limits=['Provisioned CPU runtime identity and Mach-O links only; no clean-install or complete dynamic-loading claim.',
                'No new algorithm execution, updated native-license review or catalog approval is established by this candidate import check.'])
    (review/'competition_otto_candidate_runtime_identity.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(status='passed',python_imports=len(imported),native_files=len(links['native_files']),r_packages=len(packages),neural_imports=neural)))


if __name__=='__main__':main()
