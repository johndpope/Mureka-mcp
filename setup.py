from setuptools import setup, find_packages

setup(
    name='mureka-mcp',
    version='0.0.11',
    packages=find_packages(exclude=["test_api.py"]),
    include_package_data=True,
    entry_points={
        'console_scripts': [
            'mureka-mcp = mureka_mcp.api:main',
        ],
    },
    author='wei.zhang',
    author_email='zhangwei@singularity-ai.com',
    description='The mcp server of Mureka.ai',
    keywords='aigc ai generate music song instrumental mureka mcp',
    url='https://github.com/SkyworkAI/Mureka-mcp',
    python_requires='>=3.10',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
)
