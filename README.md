lcota.github.io
===============

#### Data Analysis / Pandas Links  
[Modern Pandas - misc tips using pandas](http://tomaugspurger.github.io/modern-6-visualization.html)  


#### Coding Tools  
[Byte of ViM - ViM Tutorial](https://vim.swaroopch.com/)
[Pragmatic Programmer ViM](https://pragprog.com/book/dnvim2/practical-vim-second-edition)  
[nteract - literate programming for jupyter](https://github.com/nteract/nteract)  
[nteract blog post](https://medium.com/nteract/nteract-revolutionizing-the-notebook-experience-d106ca5d2c38#.hwbfoxdma  )

#### Scientific Writing Handouts
[Marie Davidian](http://www4.stat.ncsu.edu/~davidian/st810a/written_handout.pdf)  
[Rod Little](http://sitemaker.umich.edu/rlittle/files/styletips.pdf)  
[Paul Halmos](http://www.matem.unam.mx/ernesto/LIBROS/Halmos-How-To-Write%20Mathematics.pdf)  
[George Gopen and Judith Swan](http://engineering.missouri.edu/civil/files/science-of-writing.pdf)  

#### Cool Articles
[Green Tea Press - Free eBooks](http://greenteapress.com/wp/)
[Duolingo's Halflife Regression](http://making.duolingo.com/how-we-learn-how-you-learn)  


#### Application / GUI Frameworks  
[Kivy - app framework in python!](https://github.com/kivy-garden)  
[HyperGrid - HTML5 Realtime Grid](https://github.com/openfin/fin-hypergrid)
[Anaconda Mosaic](https://docs.continuum.io/anaconda/mosaic/)
[Anaconda Fusion](https://docs.continuum.io/anaconda/fusion/#how-to-get-fusion)
[PhosphorJS](https://phosphorjs.github.io/)  
[Photon - Electron Framework](https://github.com/connors/photon)  
[Electron Sample Applications](https://electron.atom.io/apps/)  


#### Visualization & Analysis
[Airbnb's Superset](https://github.com/airbnb/superset) | [Docs](http://airbnb.io/superset/)  
[GR Framework - Plotting Library](http://gr-framework.org/)  
[Misc Plotting Links](https://wiki.python.org/moin/NumericAndScientific/Plotting)  


##### Python Links  
[FRP in Python Article](https://jakubturek.com/functional-reactive-programming-in-python/)
[Python Design Patterns](https://www.toptal.com/python/python-design-patterns)  

#### R Links  
https://cran.r-project.org/web/packages/obAnalytics/vignettes/guide.html


#### Windows Utilities
[file search utils - slant](https://www.slant.co/topics/4033/~windows-tools-for-finding-files)  
[Locate32 - file search](http://locate32.cogit.net/)  
[CMD Prompt Tricks](http://www.articpost.com/unknown-command-prompt-tricks/)  
[FreeCommander - Explorer Replacement](http://freecommander.com/en/summary/)  
[Explorer Replacements](https://www.slant.co/topics/2404/~file-managers-for-windows?)  
[NexusFont, NexusImage](http://www.xiles.net/)  
[Power Utilities - Lifehacker](http://lifehacker.com/287966/power-replacements-for-built-in-windows-utilities)  
[Process Explorer](https://technet.microsoft.com/en-us/sysinternals/bb896653.aspx)  
[Everything File Search](http://www.voidtools.com/)  
[Mendeley PDF Article Management](https://www.mendeley.com/)  
[Copernic Desktop Search](http://www.copernic.com/en/products/desktop-search/home/index.html)  
[Window Tilers / Window Managers](https://www.slant.co/topics/1249/~window-managers-for-windows)  


#### Misc Useful Tools & Code Projects
[Docutils - Converts RST or MD docs man pages or html/latex](http://docutils.sourceforge.net/index.html)  
[Comparison between markup formats](http://hyperpolyglot.org/lightweight-markup)  
[Speeding up Python Code](https://www.ibm.com/developerworks/community/blogs/jfp/entry/Python_Meets_Julia_Micro_Performance?lang=en)  
[Documenting Python Code](https://docs.python.org/devguide/documenting.html)  
[Sphinx Docs Generator](http://www.sphinx-doc.org/en/stable/contents.html)  
[Cerebro - file manager replacement??](https://cerebroapp.com/)  
[Zazo - Electron Launcher](http://zazuapp.org/)  

[Brandon Rhodes](http://rhodesmill.org/brandon/)  

Thank you for providing the detailed feedback, I appreciate the time and effort made. I am a bit surprised about some of the comments, as they do not reflect our discussions and my notes. The need for written parameters on projects is seen in this situation. I need to reiterate that one of the most important accommodations requests was that of receiving written tasks, where possible, to reduce communication errors. While I can summarize what we speak about each time we have a call, what I notice happening is that the verbal recollection of those calls varies quite a lot from the notes taken during the call. One proposal I have to help alleviate this style of miscommunication is that I pro-actively summarize all of our calls and email those out with a meeting-minutes style summary.

I try to avoid this type of situation by taking detailed  notes using MS Office tools, as I rely on them to help me address aspects of my disabilities. While you once celebrated this tool, it now feels as if you are objecting to  my use of these tools.  These tools have become important for me as my ability to take detailed notes is one of the techniques I use to proactively reduce the impact of my disabilities. 

When we first spoke of the first part of the streaming algorithm projects on Tuesday, January 19th, it seemed we both had a mutual understanding of what was expected. It would take about a week to review the provided papers and summarize them with the primary focus being these points:
Identify any memory requirements that can be fixed and pre-allocated
Identify methods whose performance could be further enhanced with numpy and numba
Sketch out some notes about what would be needed to implement the algorithms proposed


We had a follow up conversation about this on Jan 21, which I requested to ensure we both had a mutual understanding of what was expected. We came out of this call agreeing that the approach I was taking was on point for the current phase of this project. During this call, you expressed clearly that no Python code was expected when we revisited my summaries the following week. On the 21st, we also discussed in considerable detail the pitfalls of T-Digest’s second algorithm, which required balanced search trees (AVL and Red-Black trees were proposed). You explicitly stated that while we can use C++ libraries to work with numba, and I suggested a few such implementations of these libraries, that our time, or my time, was more productively spent focusing on HistoryPCA due to its self-contained nature, as it does not require any additional data structures beyond it’s numerical algorithm. 

I was quite happy following our call on January 26th, where you expressed that I had gone into the algorithms more deeply than you had gone into them, or more deeply than you anticipated for this stage. During this call we also addressed the issues you raised in your comments. Those precise measures were very clearly presented during our screen share, which led to our decision to pursue an initial naive implementation of HistoryPCA and run some empirical tests comparing its performance with different parameters. 

Also along with our discussion on the 26th, I began the initial work of implementing the algorithms to empirically evaluate HistoryPCA’s performance with respect to it’s tuning parameters, results which I shared in the team chat. Your email this morning implied little work, or incorrect work, had been performed, while my contributions to the team chat clearly indicate that the work I had produced was in line with what we spoke about during our prior conversations. 

Let me know your thoughts on the above. 
