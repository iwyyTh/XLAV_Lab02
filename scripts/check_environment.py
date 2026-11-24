"""Import required libraries and check config/CPU/CUDA; persist actual results."""
import importlib
import importlib.metadata
import json
import os
import platform
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / '.cache/matplotlib'))


def main():
    report = dict(python=platform.python_version(), executable=sys.executable,
                  platform=platform.platform(), libraries={}, checks={})
    for package, module in {'torch':'torch', 'torchvision':'torchvision', 'numpy':'numpy',
                            'pandas':'pandas', 'matplotlib':'matplotlib', 'Pillow':'PIL',
                            'scikit-learn':'sklearn', 'tqdm':'tqdm', 'pytest':'pytest',
                            'jupyter':'jupyter_core', 'ipykernel':'ipykernel'}.items():
        try:
            importlib.import_module(module)
            report['libraries'][package] = importlib.metadata.version(package)
            report['checks'][package] = 'PASS'
        except Exception as exc:
            report['checks'][package] = f'FAIL: {exc}'
    try:
        import torch
        from source.config import CONFIG, CLASS_TO_LABEL
        assert torch.equal(torch.tensor([1., 2.], device='cpu') * 2, torch.tensor([2., 4.]))
        report['checks']['cpu_tensor'] = 'PASS'
        assert CONFIG.input_dim == 12288 and CLASS_TO_LABEL == {'horses':0, 'humans':1}
        assert CONFIG.num_workers == 0 and CONFIG.device in ('cpu','cuda')
        report['checks']['config'] = 'PASS'
        report['cuda_available'] = torch.cuda.is_available()
        report['torch_cuda_build'] = torch.version.cuda
        report['device'] = CONFIG.device
    except Exception as exc:
        report['checks']['tensor_and_config'] = f'FAIL: {exc}'
    out = ROOT / 'results/metrics/environment.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return 0 if all(x == 'PASS' for x in report['checks'].values()) else 1


if __name__ == '__main__':
    sys.exit(main())
