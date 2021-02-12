from setuptools import setup

with open("README.md","r") as fh:
    long_description = fh.read()  

setup(
    name = 'nxcurve',         # How you named your package folder (MyLib)
    version = '0.6',      # Start with a small number and increase it with every change you make
    description = 'draws RNS,QNX and BNX curves and their auc',   # Give a short description about your library
    py_modules = ['nxcurve'],
    package_dir={'':'src'},
    long_description=long_description,
  
    license='MIT',        # Chose a license from here: https://help.github.com/articles/licensing-a-repository
    author = 'Nicolas Marin',                   # Type in your name
    author_email = 'josue.marin1729@gmail.com',      # Type in your E-Mail
    url = 'https://github.com/tardigrade33/nxcurve',   # Provide either the link to your github or to your website
    #download_url = 'https://github.com/user/reponame/archive/v_01.tar.gz',    # I explain this later on
    keywords = ['RNX', 'qualitycurve', 'QNX', 'BNX'],   # Keywords that define your package best
 
    install_requires=[            # I get to this in a second
          'numpy',
          'matplotlib',
          'scikit-learn',
      ],
    classifiers=[
    'Development Status :: Beta 1',      # Chose either "3 - Alpha", "4 - Beta" or "5 - Production/Stable" as the current state of your package
    'Intended Audience :: Developers :: Data Scientist',      # Define that your audience are developers
    'Topic :: Dimentionality Reduction Quality',
    'License :: OSI Approved :: MIT License',   # Again, pick a license
    'Programming Language :: Python :: 3.9',      #Specify which pyhton versions that you want to support
    ],
)
