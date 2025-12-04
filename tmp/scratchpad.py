"""Debug why no map is generated for specific file."""

import sys
sys.path.insert(0, '/home/nuck/Repositories/RepoMapper')

from pathlib import Path
from grepmap import GrepMap

# Check if file exists
target = "build/lib/repomap_class.py"
abs_path = Path(target).resolve()

print(f"Target: {target}")
print(f"Absolute path: {abs_path}")
print(f"Exists: {abs_path.exists()}")
print(f"Is file: {abs_path.is_file()}")

if abs_path.exists():
    print(f"Size: {abs_path.stat().st_size} bytes")

    # Try to generate map
    print("\n" + "="*60)
    print("Testing GrepMap on this file...")
    print("="*60)

    mapper = GrepMap(
        map_tokens=2000,
        root=str(abs_path.parent),
        verbose=True
    )

    result, report = mapper.get_grep_map(
        chat_files=[],
        other_files=[str(abs_path)]
    )

    print(f"\nResult is None: {result is None}")
    print("\nFile Report:")
    print(f"  - Total files: {report.total_files_considered}")
    print(f"  - Definitions: {report.definition_matches}")
    print(f"  - References: {report.reference_matches}")
    print(f"  - Excluded: {len(report.excluded)}")

    if report.excluded:
        print("\nExcluded files:")
        for fname, reason in report.excluded.items():
            print(f"  - {fname}: {reason}")

    if result:
        print(f"\nMap length: {len(result)} chars")
        print("\nFirst 500 chars of map:")
        print(result[:500])
else:
    print("\n❌ File does not exist!")
    print("\nLet's check what files are in build/lib/:")
    build_lib = Path("build/lib")
    if build_lib.exists():
        print(f"\nFiles in {build_lib}:")
        for f in build_lib.iterdir():
            print(f"  - {f.name}")
    else:
        print(f"\n{build_lib} doesn't exist")
