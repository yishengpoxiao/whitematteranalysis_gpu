"""Create an offline Slicer scripted extension archive without a Slicer SDK."""
import argparse
import base64
import io
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
                archive.write(path, (destination/path.relative_to(source)).as_posix())
        archive.writestr('WMAGPU/share/Slicer-%s/WMAGPU.s4ext' % version, metadata)
        archive.write(ROOT/'License.txt', 'WMAGPU/License.txt')
        archive.write(ROOT/'README.md', 'WMAGPU/README.md')
        archive.writestr('WMAGPU/BUILD.txt', 'Source revision: %s\nSlicer layout: %s\n' % (revision, version))
    print(target)
    return target


def package_universal(output):
    """Embed platform-independent sources in a Slicer-hosted installer."""
    revision = subprocess.check_output(['git', '-C', str(REPO), 'rev-parse', 'HEAD'], text=True).strip()
    payload = io.BytesIO()
    with zipfile.ZipFile(payload, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for source, destination in ((ROOT/'Modules', Path('Modules')),
                                   (REPO/'whitematteranalysis', Path('Modules/_vendor/whitematteranalysis'))):
            for path in sorted(source.rglob('*')):
                if (not path.is_file() or '__pycache__' in path.parts
                        or path.suffix == '.pyc' or path.name == 'CMakeLists.txt'):
                    continue
                archive.write(path, (destination/path.relative_to(source)).as_posix())
        archive.write(ROOT/'License.txt', 'License.txt')
        archive.write(ROOT/'README.md', 'README.md')
        archive.writestr('metadata.s4ext',
            'scm git\nscmurl https://github.com/yishengpoxiao/whitematteranalysis_gpu\n'
            'scmrevision %s\ncategory Diffusion\nhomepage https://github.com/yishengpoxiao/whitematteranalysis_gpu\n'
            'contributors WhiteMatterAnalysis GPU acceleration project\n'
            'description GPU and CPU tractography workflows\ndepends NA\nstatus Stable\n' % revision)
    script = (ROOT/'tools/installer_template.py').read_text().replace(
        '__WMA_PAYLOAD__', base64.b64encode(payload.getvalue()).decode('ascii')).replace('__WMA_REVISION__', revision)
    target = Path(output).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    instructions = (
        'WMAGPU universal installer\n\n'
        'Extract this ZIP. Open your target 3D Slicer, then View > Python Interactor.\n'
        'Run (replace PATH with the extracted installer path):\n\n'
        'p = r"PATH/InstallWMAGPU.py"\n'
        'exec(compile(open(p, encoding="utf-8").read(), p, "exec"), {"__name__": "__main__"})\n\n'
        'Click Install WMAGPU and required dependencies; restart after completion.\n'
        'CUDA PyTorch is optional. Windows/Linux NVIDIA CUDA and macOS/ARM CPU mode.\n'
        'This universal ZIP is NOT an Extensions Manager Install from file archive.\n'
        'It creates the matching extension archive inside the running Slicer.\n'
        'Remove any installed older WMAGPU release and restart before installation.\n')
    with zipfile.ZipFile(target, 'x', compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr('WMAGPU-Installer/InstallWMAGPU.py', script)
        archive.writestr('WMAGPU-Installer/INSTALL.txt', instructions)
    print(target)
    return target


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--universal', action='store_true', help='Build a version-adaptive installer ZIP to run inside Slicer')
    parser.add_argument('--slicer-version', default='5.2', help='Major.minor; package layout must match Slicer')
    args = parser.parse_args()
    if args.universal:
        package_universal(args.output)
    else:
        package(args.output, args.slicer_version)
