#!/usr/bin/env python3
"""Export the reviewed sharing list without initializing Git or uploading files."""
import argparse
import hashlib
import json
import re
from pathlib import Path
import shutil


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--profile', choices=('full', 'public', 'student'), default='full',
                        help='full: historical sharing list; public: English project release; student: legacy alias')
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    output = args.output.resolve()
    if output.exists() or output == root or root in output.parents:
        parser.error('Output must be a new directory outside the source directory')
    list_name = 'RELEASE_FILES.txt' if args.profile == 'public' else ('STUDENT_RELEASE_FILES.txt' if args.profile == 'student' else 'SHARED_FILES.txt')
    names = [s.strip() for s in (root/list_name).read_text().splitlines() if s.strip() and not s.startswith('#')]
    sources = []
    for name in names:
        rel = Path(name)
        source = root/rel
        if args.profile in ('public', 'student'):
            publication = root/'publication_en'/rel
            if publication.is_file():
                source = publication
        if rel.is_absolute() or '..' in rel.parts or source.is_symlink() or not source.is_file():
            parser.error('Invalid sharing entry: '+name)
        if args.profile in ('public', 'student') and re.search(r'[\u3400-\u9fff]', source.read_text(encoding='utf-8')):
            parser.error('Public release must be English-only: '+name)
        sources.append((name, source))
    output.mkdir(parents=True)
    records = []
    for name, source in sources:
        target = output/name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        data = target.read_bytes()
        records.append({'path':name, 'bytes':len(data), 'sha256':hashlib.sha256(data).hexdigest()})
    (output/'EXPORT_MANIFEST.json').write_text(json.dumps({'files':records}, indent=2)+'\n')
    print(json.dumps({'output':str(output),'files':len(records),'bytes':sum(x['bytes'] for x in records)},indent=2))


if __name__ == '__main__':
    main()
