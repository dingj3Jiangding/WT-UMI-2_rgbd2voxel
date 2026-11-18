"""Export recorded head RGB beside its synchronized cumulative voxel map."""

import argparse
import json

from wtumi_recon.scene_comparison import export_camera_voxel_comparison


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--result', required=True, help='Completed single-scene result directory')
    parser.add_argument('--max-voxels', type=int, default=50000, help='Maximum displayed surface cells per frame')
    parser.add_argument('--no-mp4', action='store_true', help='Export GIF and PNG frames without MP4')
    args = parser.parse_args()
    result = export_camera_voxel_comparison(args.result, max_voxels=args.max_voxels,
                                            export_mp4=not args.no_mp4)
    print(json.dumps({'scene_name': result['scene_name'], 'paths': result['paths'],
                      'frames': result['frame_count'], 'video': result['video']}, indent=2))


if __name__ == '__main__':
    main()
