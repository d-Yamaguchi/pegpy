from setuptools import setup
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent / 'tests'))

setup(
	name = 'pegpy',
    version = '0.1.1',
    url = 'https://github.com/KuramitsuLab/pegpy.git',
    license = 'KuramitsuLab',
    author = 'KuramitsuLab',
    description = 'Nez Parser for Python',
    install_requires = ['setuptools'],
	packages = ['pegpy', 'pegpy.gparser', 'pegpy.origami', 'pegpy.tbcnn'],
	package_data = {'pegpy': ['grammar/*.tpeg', 'grammar/*.gpeg', 'origami/*.origami']},
	entry_points = {
		'console_scripts': [
			'pegpy = pegpy.main:main'
		]
	},
	test_suite = 'test_all.suite'
)
