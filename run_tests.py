

from pathlib import Path
import subprocess


test_cases = [name for name in Path('tests').iterdir() if 'test' in str(name)]

for test_case in test_cases:

    print()
    print()
    print()
    print(f'test case: {test_case}')

    try:
        subprocess.run(f'ipython {test_case}', shell = True)
    except Exception as e:
        print(e)



