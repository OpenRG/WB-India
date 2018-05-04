  # Fiscal Modeling and Overlapping Generations Model Training For the Department of Revenue

This public repository contains the training materials, tutorials, and code for the two week training delivered to the Government of India Department of Revenue in conjuction with the World Bank in New Delhi, India, April 30 to May 11, 2018 by [Richard Evans](https://sites.google.com/site/rickecon/) and [Jason DeBacker](http://jasondebacker.com/).

We will roughly follow schedule the schedule below for each of the 10 workdays during the training:

* 9am to noon: Lecture, theory, computational instruction
* noon to 1pm: Lunch
* 1pm to 4pm: Guided computational practice, implementation, problem sets

DeBacker and Evans (2018) have created a series of textbook chapters complete with theoretical exposition, computational description and tips, and problem sets. These chapters and their instruction are meant to progressively build upon each other. In the end, you will be building a computational implementation of an Overlapping Generations (OG) model for fiscal policy that is thousands of lines of code. We will train your research group to understand each section of code and to write the code in a way that is accessible, modular, scalable, and amenable to collaboration among a large group of researchers.

This will be an intensive two weeks. We are providing your researchers 6 areas of tutorials that they will benefit from reading before the training. We will, of course, teach these things as we go through the material. But we will be able to proceed at a faster pace if the attendees are already familiar with most of the concepts below.


## Daily Course Schedule ##

|  Date  | Day | Topic | Notes | Code |
|--------|-----|-------|----------|----------|
| Apr 30 |  M | Intro session, office hours, computational setup | [Slides](https://github.com/OpenRG/WB-India/blob/master/Slides/Day1slides.pdf) |  |
| May  1 |  T | 3-period-lived OG model                  | [Ch.  2](https://github.com/OpenRG/WB-India/blob/master/Chapters/OGtext_ch02.pdf) |  |
| May  2 |  W | S-period-lived OG model                  | [Ch.  3](https://github.com/OpenRG/WB-India/blob/master/Chapters/OGtext_ch03.pdf) |  |
| May  3 | Th | Endogenous labor supply                  | [Ch.  4](https://github.com/OpenRG/WB-India/blob/master/Chapters/OGtext_ch04.pdf) |  |
| May  4 |  F | Heterogeneous ability, wealth inequality | [Ch.  5](https://github.com/OpenRG/WB-India/blob/master/Chapters/OGtext_ch05.pdf) |  |
| May  7 |  M | Demographic dynamics                     | [Ch.  7](https://github.com/OpenRG/WB-India/blob/master/Chapters/OGtext_ch07.pdf) |  |
| May  8 |  T | Productivity growth and stationarization | [Ch.  7](https://github.com/OpenRG/WB-India/blob/master/Chapters/OGtext_ch07.pdf) |  |
| May  9 | W | Household and corporate taxation         | [Ch. 11](https://github.com/OpenRG/WB-India/blob/master/Chapters/OGtext_ch11.pdf) |  |
| May 10 |  Th | Unbalanced government budget constraint  | [Ch. 12](https://github.com/OpenRG/WB-India/blob/master/Chapters/OGtext_ch12.pdf) |  |
| May 11 |  F | Large-scale model training and practice |  |  | |


# Pre-course Tutorial Areas

1. [Instructions for installing the Anaconda distribution of Python](https://github.com/OpenRG/WB-India#1-instructions-for-installing-the-anaconda-distribution-of-python)
2. [Text editor suggestions](https://github.com/OpenRG/WB-India#2-text-editor-suggestions)
3. [Jupyter notebooks](https://github.com/OpenRG/WB-India#3-jupyter-notebooks)
4. [Python tutorials](https://github.com/OpenRG/WB-India#4-python-tutorials)
5. [Git and GitHub.com tutorial](https://github.com/OpenRG/WB-India#5-git-and-github-tutorial)
6. [PEP 8, docstring commenting, and module structure](https://github.com/OpenRG/WB-India#6-pep-8-docstring-commenting-and-module-structure)
7. [References](https://github.com/OpenRG/WB-India#7-references)


## 1. Instructions for installing the Anaconda distribution of Python

We will be using the [Python](https://www.python.org/) programming language and many of its powerful libraries for writing the code that will run most of the computational methods we will use during the Boot Camp. Using an open source language, such as Python, has the advantage of being free and accessible for anyone who wishes to learn these materials or contribute to these projects. Being open source also allows Python users to go into the source code of any function to modify it to suit one's needs.

We recommend that each participant download the Anaconda distribution of Python provided by [Anaconda, Inc.](https://www.anaconda.com/download/). We recommend the most recent stable version of Python, which is currently Python 3.6. This can be done from the [Anaconda download page](https://www.anaconda.com/download/) for Windows, Mac OSX, and Linux machines.


## 2. Text editor suggestions

In our recommended Python development workflow, you will write Python scripts and modules (`*.py` files) in a text editor. Then you will run those scripts from your terminal. You will want a capable text editor for developing your code. Many capable text editors exist, but we recommend two.

1. [Atom](https://atom.io)
2. [Sublime Text 3](https://www.sublimetext.com)

Atom and Vim are completely free. A trial version of Sublime Text 3 is available for free, but a licensed version is $80 (US dollars). In the following subsections, we give some of the details of each of the above three text editors.


### 2.1. Atom

[Atom](https://atom.io) is an open source text editor developed by people at GitHub.com. This editor has all the features of Sublime Text 3, but it also allows users full customizability. Further, it has been a while now that the users of Atom have surpassed the critical mass necessary to keep the editor progressing with the most cutting edge additions.

There are several packages you'll want to install with Atom.  Once Atom is installed, you can add packages by navigating File->Settings->Install (on Windows) or Atom->Preferences->Install (on Mac) and then typing in the name of the package you would like to install.  Windows users might also find [this](https://www.youtube.com/watch?v=nshxC0YO_X0) YouTube video or [this blog post](https://zenagiwa.wordpress.com/2015/02/15/installing-packages-for-atom-on-windows/) helpful in installing packages.

For work with Python, we recommend the following packages be installed:

* MagicPython
* linter
* linter-flake8 (need to install `flake8` in python)
* python-indent
* autocomplete-python
* tabs-to-spaces
* minimap
* open-recent

For development with GitHub we recommend:

* merge-conflict


### 2.2. Sublime Text 3

[Sublime Text 3](https://www.sublimetext.com) is the most widely used and versatile private software text editor. It has tremendous flexibility, as well as the polish of a piece of professional software. Sublime Text 3 will cost $80 for a license, although you can use a trial version indefinitely without charge while only having to suffer through frequent reminders to buy the full version.


## 3. Jupyter Notebooks

[Jupyter notebooks](http://jupyter.org/) are files that end with the `*.ipynb` suffix. These notebooks are opened in a browser environment and are an open source web application that combines instructional text with live executable and modifyable code for many different programming platforms (e.g., Python, R, Julia). Jupyter notebooks are an ideal tool for teaching programming as they provide the code for a user to execute and they also provide the context and explanation for the code. We have provided a number of Jupyter notebooks in the [Tutorials](https://github.com/OpenRG/WB-India/tree/master/Tutorials) folder of this repository.

These notebooks used to be Python-specific, and were therefore called iPython notebooks (hence the `*.ipynb` suffix). But Jupyter notebooks now support many programming languages, although the name still pays homage to Python with the vestigal "py" in "Jupyter". The notebooks execute code from the kernel of the specific programming language on your local machine.

Jupyter notebooks capability will be automatically installed with your download of the [Anaconda distribution](https://www.anaconda.com/download/) of Python. If you did not download the Anaconda distribution of Python, you can download Jupyter notebooks separately by following the instructions on the Jupyter [install page](http://jupyter.org/install).


### 3.1. Opening a Jupyter notebook

Once Jupyter is installed--whether through Anaconda or through the Jupyter website--you can open a Jupyter notebook by the following steps.

1. Navigate in your terminal to the folder in which the Jupyter notebook files reside. In the case of the Jupyter notebook tutorials in this repository, you would navigate to the `~/WB-India/Tutorials/` directory.
2. Type `jupyter notebook` at the terminal prompt.
3. A Jupyter notebook session will open in your browser, showing the available `*.ipynb` files in that directory.
  *  In some cases, you might receive a prompt in the terminal telling you to paste a url into your browser.
4. Double click on the Jupyter notebook you would like to open.

It is worth noting that you can also simply navigate to the URL of the Jupyter notebook file in the GitHub repository on the web (e.g., [https://github.com/OpenRG/WB-India/blob/master/Tutorials/PythonReadIn.ipynb](https://github.com/OpenRG/WB-India/blob/master/Tutorials/PythonReadIn.ipynb)). You can read the Jupyter notebook on GitHub.com, but you cannot execute any of the cells. You can only execute the cells in the Jupyter notebook when you follow the steps above and open the file from a Jupyter notebook session in your browser.


### 3.2. Using an open Jupyter notebook

Once you have opened a Jupyter notebook, you will find the notebook has two main types of cells: Markdown cells and Code cells. Markdown cells have formatted Jupyter notebook markdown text, and serve primarily to present context for the coding cells. A reference for the markdown options in Jupyter notebooks is found in the [Jupyter markdown documentation page](http://jupyter-notebook.readthedocs.io/en/latest/examples/Notebook/Working%20With%20Markdown%20Cells.html).

You can edit a Markdown cell in a Jupyter notebook by double clicking on the cell and then making your changes. Make sure the cell-type box in the middle of the top menu bar is set to `Markdown`. To implement your changes in the Markdown cell, type `Shift-Enter`.

A Code cell will have a `In [ ]:` immediately to the left of the cell for input. The code in that cell can be executed by typing `Shift-Enter`. For a Code cell, the  cell-type box in the middle of the top menu bar says `Code`.


### 3.3. Closing a Jupyter notebook

When you are done with a Jupyter notebook, you first save any changes that you want to remain with the notebook. Then you close the browser windows associated with that Jupyter notebook session. You should then close the local server instance that was opened to run the Jupyter notebook in your terminal window. On a Mac or Windows, this is done by going to your terminal window and typing `Cmd-C` or `Ctrl-C` and then selecting `y` for yes and hitting `Enter`.


## 4. Python tutorials

For this training, we have included in this repository six basic Python tutorials in the [`Tutorials`](https://github.com/OpenRG/WB-India/tree/master/Tutorials) directory.

1. [PythonReadIn.ipynb](https://github.com/OpenRG/WB-India/blob/master/Tutorials/PythonReadIn.ipynb). This Jupyter notebook provides instruction on basic Python I/O, reading data into Python, and saving data to disk.
2. [PythonNumpyPandas.ipynb](https://github.com/OpenRG/WB-India/blob/master/Tutorials/PythonNumpyPandas.ipynb). This Jupyter notebook provides instruction on working with data using `NumPy` as well as Python's powerful data library `pandas`.
3. [PythonDescribe.ipynb](https://github.com/OpenRG/WB-India/blob/master/Tutorials/PythonDescribe.ipynb). This Jupyter notebook provides instruction on describing, slicing, and manipulating data in Python.
4. [PythonFuncs.ipynb](https://github.com/OpenRG/WB-India/blob/master/Tutorials/PythonFuncs.ipynb). This Jupyter notebook provides instruction on working with and writing Python functions.
5. [PythonVisualize.ipynb](https://github.com/OpenRG/WB-India/blob/master/Tutorials/PythonVisualize.ipynb). This Jupyter notebook provides instruction on creating visualizations in Python.
6. [PythonRootMin.ipynb](https://github.com/OpenRG/WB-India/blob/master/Tutorials/PythonRootMin.ipynb). This Jupyter notebook provides instruction on implementing univariate and multivariate root finders and unconstrained and constrained minimizers using functions in the [`scipy.optimize`](https://docs.scipy.org/doc/scipy/reference/optimize.html) sub-library.

To further one's Python programming skills, a number of other great resources exist.

* The [official Python 3 tutorial site](https://docs.python.org/3/tutorial/)
* [QuantEcon.net](https://lectures.quantecon.org/py/) is a site run by [Thomas Sargent](http://www.tomsargent.com/) (NYU Stern) and [John Stachurski](http://johnstachurski.net/) (Australia National University). QuantEcon has a very large number of high-quality economics focused computational tutorials in Python. The first three sections provide a good introduction to Python programming.
* [Python computational labs](http://www.acme.byu.edu/2017-2018-materials/) of the Applied and Computational Mathematics Emphasis at Brigham Young University.
* [Code Academy's Python learning module](https://www.codecademy.com/learn/learn-python)

In addition, a number of excellent textbooks and reference manuals are very helpful and may be available in your local library. Or you may just want to have these in your own library. Lutz (2013) is a giant 1,500-page reference manual that has an expansive collection of materials targeted at beginners. Beazley (2009) is a more concise reference but is targeted at readers with some experience using Python. Despite its focus on a particular set of tools in the Python programming language, McKinney (2018) has a great introductory section that can serve as a good starting tutorial. Further, its focus on Python's data analysis capabilities is truly one of the important features of Python. Rounding out the list is Langtangen (2010). This book's focus on scientists and engineers makes it a unique reference for optimization, wrapping C and Fortran and other scientific computing topics using Python.


## 5. Git and GitHub tutorial

Git is a powerful version control software. Each participant should [install Git software](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git) on their computer. Each participant should also create their own personal account on [GitHub.com](https://github.com/) (select the "Sign Up" or "Sign Up for GitHub" option). The combination of Git software and the GitHub online platform represent an extremely powerful collaboration platform for computational model building.

We have included a tutorial on using [Git and GitHub.com](https://github.com/OpenRG/WB-India/blob/master/Tutorials/git_tutorial.pdf) in the [Tutorials](https://github.com/OpenRG/WB-India/tree/master/Tutorials) directory of this repository. Git is a powerful version control software that comes natively installed on many machines and is widely used. GitHub.com is the most widely used online platform for hosting open source projects and integrating with Git software. Git has a significant learning curve, but it is essential for large collaborations that involve software development.

A more comprehensive Git resource is [*Pro Git*](https://git-scm.com/book/en/v2), by Chacon and Straub (2014). This book is open access, and is available online at [https://git-scm.com/book/en/v2](https://git-scm.com/book/en/v2). But Evans likes having it in his library in hard copy. This book is the difinitive guide with everything Git, and it has as its primary application the interaction between Git and GitHub. However, the workflow described in the tutorial above was hard to find in this Git book.


## 6. PEP 8, docstring commenting, and module structure

Computer code executes some set of commands in an organized way. In every case, there are often many ways to execute a set of instructions--some ways more efficient than others. However, code has at least three functions.

1. Efficiently execute the task at hand.
2. Be accessible and usable to other programmers.
3. Be scalable and integrable with other projects and procedures.

Bill Gates is credited with the following plea for efficiency and parsimony in code writing.

> "Measuring programming progress by lines of code is like measuring aircraft building progress by weight."

Strong support for points (2) and (3) is Eagleson's Law.

> "Any code of your own that you haven't looked at for six or more months might as well have been written by someone else."

Because of the latter two characteristics, Python code has developed some conventions and best practices, some of which have been institutionalized in the [PEP 8--Style Guide for Python Code](https://www.python.org/dev/peps/pep-0008/) ("PEP" stands for Python Enhancement Proposals). Key examples PEP 8 Python coding conventions are the following.

* Indents should be 4 spaces (not tab)
* Limit all lines to a maximum of 79 characters long blocks of text being limited to 72 characters (Evans limits all his lines to 72 characters)
* Use a space after a comma
* Use a space before and after arithmetic operators

In the text editors Atom, Sublime Text 3, and Vim, you can install Linter packages that highlight areas of your code that break PEP 8 rules and tell you what the violation is.

There are fewer conventions in docstring structure, but we have developed some of our own that are outlined in the [PythonFuncs.ipynb](https://github.com/OpenRG/WB-India/blob/master/Tutorials/PythonFuncs.ipynb) Jupyter notebook. See especially Sections 3 and 4 of the notebook.


## 7. References

* Beazley, David M., *Python Essential Reference*, 4th edition, Addison-Wesley (2009).
* Chacon, Scott and Ben Straub, [*Pro Git: Everything You Need To Know About Git*](https://git-scm.com/book/en/v2), 2nd edition, Apress (2014).
* DeBacker, Jason and Richard W. Evans, *Overlapping Generations Models for Policy Analysis: Theory and Computation*, unpublished (2018)
* Langtangen, Hans Petter, *Python Scripting for Computational Science*, Texts in Computational Science and Engineering, 3rd edition, Springer (2010).
* Lutz, Mark, *Learning Python*, 5th edition, O'Reilly Media, Inc. (2013).
* McKinney, Wes, *Python for Data Analysis*, 2nd edition, O'Reilly Media, Inc. (2018).
