"""Convert the Springer (sn-jnl) main.tex into a REVTeX 4.2 (APS, two-column) version: main_revtex.tex.
Usage: python paper/to_revtex.py   (run after every change of main.tex)
"""
import os, re, hashlib
HERE = os.path.dirname(os.path.abspath(__file__))
s = open(os.path.join(HERE, "main.tex"), encoding="utf-8").read().replace("\r\n", "\n")

title = re.search(r"\\title\[[^\]]*\]\{(.*?)\}\n", s).group(1)
i = s.index("\\abstract{") + len("\\abstract{"); j = s.index("}\n\n\\keywords{", i); abstract = s[i:j]
kw = re.search(r"\\keywords\{(.*?)\}\n", s).group(1)
body = s[s.index("\\maketitle") + len("\\maketitle"):]

head = r"""%% REVTeX 4.2 (APS, two-column) version, generated from main.tex by to_revtex.py -- edit main.tex, not this file.
\documentclass[aps,prd,twocolumn,superscriptaddress,nofootinbib,floatfix]{revtex4-2}
\usepackage{graphicx}
\usepackage{amsmath,amssymb}
\usepackage{booktabs}
\usepackage{xcolor}
\usepackage{float}
\usepackage{adjustbox}
\usepackage{algorithm}
\usepackage{algpseudocode}
\usepackage[hidelinks]{hyperref}
% let floats fill pages: fewer half-empty pages and columns
\renewcommand{\topfraction}{0.9}\renewcommand{\bottomfraction}{0.8}\renewcommand{\textfraction}{0.07}
\renewcommand{\floatpagefraction}{0.85}\renewcommand{\dbltopfraction}{0.9}\renewcommand{\dblfloatpagefraction}{0.85}
\setcounter{topnumber}{3}\setcounter{dbltopnumber}{2}\setcounter{totalnumber}{5}

\begin{document}

\title{""" + title + r"""}

\author{Kamran Qadir Abbasi}
\email{kamranqadir@numl.edu.pk}
\affiliation{Department of Mathematics, Faculty of Engineering and Computing (FE\&C), National University of Modern Languages (NUML), Islamabad, Pakistan}
\author{Sohaib Sulman}
\thanks{Corresponding author}
\email{i248068@isb.nu.edu.pk}
\affiliation{Department of Data Science \& AI, National University of Computer and Emerging Sciences (FAST-NUCES), Islamabad, Pakistan}

\begin{abstract}
""" + abstract + r"""
\end{abstract}

\keywords{""" + kw + r"""}

\maketitle
"""
b = body
b = b.replace("\\begin{figure}[!htbp]", "\\begin{figure*}[!t]").replace("\\end{figure}", "\\end{figure*}")
# near-square figures go in one column, exactly where they are referenced
b = re.sub(r"\\begin\{figure\*\}\[!t\]\n\\centering\\includegraphics\[width=[^\]]*\]\{(figs/fig_classwise_heatmap\.png)\}(.*?)\\end\{figure\*\}",
           r"\\begin{figure*}[!t]\n\\centering\\includegraphics[width=0.62\\textwidth]{\1}\2\\end{figure*}", b, flags=re.S)
b = re.sub(r"\\input\{tables/(\w+)\}", r"\\input{tables/rt_\1}", b)
b = re.sub(r"\\backmatter\n\\bmhead\{Acknowledgements\}\n(.*?)\n\n", r"\\begin{acknowledgments}\n\1\n\\end{acknowledgments}\n\n", b, flags=re.S)
b = b.replace("\\begin{appendices}", "\\appendix").replace("\\end{appendices}", "")
b = b.replace("\\bibliography{references}", "\\bibliographystyle{apsrev4-2}\n\\bibliography{references}")
# two-column fixes: algorithm float placed with [H]; split the long Mondrian-ACI update over two lines
b = b.replace("\\begin{algorithm}[!htbp]", "\\begin{algorithm}[H]")
b = b.replace("\\big),\\qquad \\alpha_{t+1}^{(y)}=\\alpha_t^{(y)}\\ \\ \\text{for } y\\ne y_t.\n\\end{equation}",
              "\\big),\\\\\n\\alpha_{t+1}^{(y)}&=\\alpha_t^{(y)}\\quad \\text{for } y\\ne y_t.\n\\end{aligned}\\end{equation}")
b = b.replace("\\begin{equation}\n\\alpha_{t+1}^{(y_t)} = \\alpha_t^{(y_t)}", "\\begin{equation}\\begin{aligned}\n\\alpha_{t+1}^{(y_t)} &= \\alpha_t^{(y_t)}")
out = head + b
open(os.path.join(HERE, "main_revtex.tex"), "w", encoding="utf-8", newline="\n").write(out)
print("wrote main_revtex.tex", len(out), hashlib.sha256(out.encode("utf-8")).hexdigest()[:12])
