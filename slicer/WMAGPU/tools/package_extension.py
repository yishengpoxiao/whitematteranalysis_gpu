"""Create an offline Slicer scripted extension archive without a Slicer SDK."""
import argparse
from datetime import date
from pathlib import Path
import subprocess
import zipfile

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]


def package(output, version):
    revision = subprocess.check_output(['git', '-C', str(REPO), 'rev-parse', 'HEAD'], text=True).strip()
    target = Path(output).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    modules = Path('WMAGPU') / 'lib' / ('Slicer-' + version) / 'qt-scripted-modules'
    metadata = ('scm git\nscmurl https://github.com/yishengpoxiao/whitematteranalysis_gpu\n'
                'scmrevision %s\ncategory Diffusion\n'
                'homepage https://github.com/yishengpoxiao/whitematteranalysis_gpu\n'
                'contributors WhiteMatterAnalysis GPU acceleration project\n'
                'description GPU/CPU tractography registration, atlas building, labeling and MRML export\n'
                'depends NA\nstatus Stable\nupdated %s\n') % (revision, date.today().isoformat())
    with zipfile.ZipFile(target, 'x', compression=zipfile.ZIP_DEFLATED) as archive:
        for source, destination in ((ROOT/'Modules', modules),
                                     (REPO/'whitematteranalysis', modules/'_vendor/whitematteranalysis')):
            for path in sorted(source.rglob('*')):
                if (not path.is_file() or '__pycache__' in path.parts
                        or path.suffix == '.pyc' or path.name == 'CMakeLists.txt'):
                    continue
                archive.write(path, str(destination/path.relative_to(source)))
        archive.writestr('WMAGPU/share/Slicer-%s/WMAGPU.s4ext' % version, metadata)
        archive.write(ROOT/'License.txt', 'WMAGPU/License.txt')
        archive.write(ROOT/'README.md', 'WMAGPU/README.md')
        archive.writestr('WMAGPU/BUILD.txt', 'Source revision: %s\nSlicer layout: %s\n' % (revision, version))
    print(target)
    return target


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--slicer-version', default='5.2', help='Major.minor; package layout must match Slicer')
    args = parser.parse_args()
    package(args.output, args.slicer_version)
