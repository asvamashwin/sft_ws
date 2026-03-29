import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'med_sentinel'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
        (os.path.join('share', package_name, 'proto'), glob('proto/*.proto')),
        (os.path.join('share', package_name, 'description', 'urdf'), glob('description/urdf/*.urdf')),
        (os.path.join('share', package_name, 'description', 'srdf'), glob('description/srdf/*.srdf')),
        (os.path.join('share', package_name, 'description', 'meshes', 'collision'), glob('description/meshes/collision/*.stl')),
        (os.path.join('share', package_name, 'description', 'meshes', 'visual'), glob('description/meshes/visual/*.dae')),
    ],
    install_requires=[
        'setuptools',
        'numpy',
        'pyyaml',
        'protobuf',
        'fastapi',
        'uvicorn[standard]',
        'websockets',
    ],
    zip_safe=True,
    maintainer='Ashwin',
    maintainer_email='asvamashwin@gmail.com',
    description='Med-Sentinel 360: Medical-grade digital twin in Isaac Sim',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'scene_builder = med_sentinel.scene_builder:main',
            'robot_controller = med_sentinel.robot_controller:main',
            'bridge_server = med_sentinel.bridge.server:run_server',
            'bridge_benchmark = med_sentinel.bridge.benchmark:main',
            'bridge_stress_test = med_sentinel.bridge.stress_test:main',
            'safety_test = med_sentinel.safety.safety_test_runner:main',
        ],
    },
)
