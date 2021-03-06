from distutils.core import setup
setup(
  name = 'simple_image_classifier',         # How you named your package folder (MyLib)
  packages = ['simple_image_classifier'],   # Chose the same as "name"
  version = '0.6',      # Start with a small number and increase it with every change you make
  license='MIT',        # Chose a license from here: https://help.github.com/articles/licensing-a-repository
  description = 'A simple image classification library using keras, tensorflow and tensorflow hub',   # Give a short description about your library
  author = 'Vishnu N',                   # Type in your name
  author_email = 'vishnunkumar25@gmail.com',      # Type in your E-Mail
  url = 'https://github.com/Vishnunkumar/simple_image_classifier',   # Provide either the link to your github or to your website
  download_url ='https://github.com/Vishnunkumar/simple_image_classifier/archive/refs/tags/v_06.tar.gz',    # I explain this later on
  keywords = ['Image Classification', 'Keras', 'TensorFlow'],   # Keywords that define your package best
  install_requires=[            # I get to this in a second
          'pandas',
          'numpy',
          'opencv-python==4.1.2.30',
          'tensorflow==2.5.0',
          'tensorflow_hub',
          'sklearn',
  ],
  classifiers=[
    'Development Status :: 3 - Alpha',      # Chose either "3 - Alpha", "4 - Beta" or "5 - Production/Stable" as the current state of your package
    'Intended Audience :: Developers',      # Define that your audience are developers
    'Topic :: Software Development :: Build Tools',
    'License :: OSI Approved :: MIT License',   # Again, pick a license
    'Programming Language :: Python :: 3',      #Specify which pyhton versions that you want to support
    'Programming Language :: Python :: 3.4',
    'Programming Language :: Python :: 3.5',
    'Programming Language :: Python :: 3.6',
    'Programming Language :: Python :: 3.8'
  ],
)
