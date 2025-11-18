from setuptools import find_packages, setup

package_name = 'labassist_nodes'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='simonpontoppidan',
    maintainer_email='simonpontoppidan11@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
    'console_scripts': [
        'sim_actions = labassist_nodes.sim_actions:main',
        'fsm_validator = labassist_nodes.fsm_validator:main',
        'notifier_console = labassist_nodes.notifier_console:main',
        'camera_player = labassist_nodes.camera_player:main',
        'feature_resnet18 = labassist_nodes.feature_resnet18:main',
        'mstcn_infer = labassist_nodes.mstcn_infer:main',
    ],
},
)
