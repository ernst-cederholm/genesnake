

from pathlib import Path
import subprocess


test_cases = [f'./{str(name)}' for name in Path('.').iterdir() if 'test' in str(name)]

for test_case in test_cases:

    print()
    print()
    print()
    print(f'test case: {test_case}')

    try:
        subprocess.run(f'ipython {test_case}')
    except Exception as e:
        print(e)



