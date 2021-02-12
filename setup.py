from setuptools import setup

with open("README.md","r") as fh:
    long_description = fh.read()  

setup(
    name = 'nxcurve',         # How you named your package folder (MyLib)
    version = '0.6.1',      # Start with a small number and increase it with every change you make
    description = 'draws RNS,QNX and BNX curves and their auc',   # Give a short description about your library
    py_modules = ['nxcurve'],
    package_dir={'':'src'},
    long_description=long_description,
    long_description_content_type="text/markdown",
  
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
    'Development Status :: 4 - Beta ',     
    'License :: OSI Approved :: MIT License'  ,  
    'Programming Language :: Python :: 3.9',     
    ],
)
