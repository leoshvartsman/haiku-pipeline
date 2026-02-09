Colab setup and reproducible build notes

Goal
- Ensure PDF and EPUB outputs produced in Google Colab match local builds by shipping templates, fonts and cover images and using Pandoc resource paths.

Quick steps (copy/paste into a Colab cell)

# 1) Clone your repo
!git clone https://github.com/you/your-repo.git
%cd your-repo

# 2) Install system requirements (Pandoc + XeLaTeX + fonts)
!sudo apt-get update
!sudo apt-get install -y pandoc texlive-xetex texlive-fonts-recommended texlive-latex-extra fonts-dejavu-core fonts-dejavu-extra
!fc-cache -fv

# 3) If you haven't committed font files, upload/copy them into book_formatter/templates/
# Example copying from a Drive mount:
# %cp /content/drive/MyDrive/fonts/*.ttf book_formatter/templates/

# 4) Run the formatter ensuring --templates points to the templates folder and cover path is correct
!python3 book_formatter/book_formatter.py input/yourfile.txt \
  --cover book_formatter/templates/cover.jpg \
  --templates book_formatter/templates \
  --out-dir out \
  --formats pdf,epub \
  --page-size letter \
  --font-size 12pt

Notes
- We added a Pandoc `--resource-path` inside `book_formatter.py` so Pandoc will search the current directory and the templates folder for CSS, images and fonts.
- To make EPUBs look identical across readers, include the desired TTF/OTF font files in `book_formatter/templates/` and add `@font-face` rules in `book_formatter/templates/ebook.css` (an example `@font-face` has been added).
- For PDF parity, Colab must have XeLaTeX and the fonts available. Installing system fonts (apt) and/or committing the fonts to the repo helps.

Recommended repo changes
- Commit `book_formatter/templates/` including:
  - `ebook.css` (already in repo)
  - any TTF/OTF font files you want to use
  - your cover image(s)

Verification
- After running, check `out/` for generated files and open them (Calibre viewer for EPUB; any PDF viewer for PDF) to confirm font sizes and cover presence.

If you'd like, I can prepare a tiny script to copy chosen font files from a fonts/ directory into the templates folder automatically during the Colab run.