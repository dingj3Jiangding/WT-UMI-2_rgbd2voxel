"""Compare fixed and independently scanning head cameras while carrying a box."""
import argparse
from datetime import datetime
import json
from pathlib import Path
from wtumi_recon.view_demo import run_view_demo


def main():
    root=Path(__file__).resolve().parents[1]
    parser=argparse.ArgumentParser(description='Independent head orientation under payload occlusion')
    parser.add_argument('--config',type=Path,default=root/'configs/demos/stairs_views.yaml')
    parser.add_argument('--output',type=Path,help='New or empty output directory')
    args=parser.parse_args()
    out=args.output or Path('outputs')/f"view_demo_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
    try:
        result=run_view_demo(args.config,out,progress=lambda message:print(message,flush=True))
    except (ValueError,OSError,KeyError,ImportError,RuntimeError) as exc:
        parser.exit(2,f'View demo error: {exc}\n')
    print(json.dumps({'output':str(out.resolve()),'paths':result['paths'],'verification':result['verification']},indent=2))


if __name__=='__main__':
    main()
