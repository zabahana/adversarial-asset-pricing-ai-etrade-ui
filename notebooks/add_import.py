with open('protagonist_dqn.py', 'r') as f:
    content = f.read()

# Add pandas import after torch imports
if 'import pandas as pd' not in content:
    # Find the imports section and add pandas
    content = content.replace(
        'import numpy as np\nfrom collections import deque',
        'import numpy as np\nimport pandas as pd\nfrom collections import deque'
    )
    
    with open('protagonist_dqn.py', 'w') as f:
        f.write(content)
    
    print("✓ Added pandas import")
else:
    print("✓ Pandas already imported")
