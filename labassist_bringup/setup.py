from setuptools import find_packages, setup

package_name = 'labassist_bringup'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/labassist_bringup/launch', 
        [
            'labassist_bringup/launch/demo_pipeline.launch.py',
            'labassist_bringup/launch/sim_from_video.launch.py'
        ])
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
        ],
    },
)
