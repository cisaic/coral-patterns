from setuptools import setup, find_packages

setup(name='coral_patterns',
      version='1.0.0',
      packages=find_packages(),
      authors=[
          {'name': 'Myriam Belkhatir', 'email': 'myriam.belkhatir@student.uva.nl'},
          {'name': 'Zoe Busche', 'email': 'zoe.busche@student.uva.nl'},
          {'name': 'Christina Isaicu', 'email': 'christina.isaicu@student.uva.nl'},
          {'name': 'Jensen Valkenhoff', 'email': 'jensen.valkenhoff@student.uva.nl'},
      ],
      license="MIT",
      description='Course project for Complex Systems Simulations @ UvA',
      python_requires='>=3.10',
      install_requires=[
            "numpy",
            "tqdm",
            "scipy",
            "pytest",
            "mock",
            "matplotlib",
            "seaborn",
            "plotly",
            "networkx",
            "powerlaw",
            "pandas"
      ],
      )
