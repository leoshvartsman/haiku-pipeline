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

# 3) Run the formatter ensuring --templates points to the templates folder and cover path is correct
!python3 book_formatter/book_formatter.py input/yourfile.txt \
  --cover book_formatter/templates/cover.jpg \
  --templates book_formatter/templates \
  --out-dir out \
  --formats pdf,epub \
  --page-size letter \
  --font-size 12pt

Notes
- EB Garamond (SIL Open Font License) is committed in `book_formatter/templates/fonts/` and is automatically embedded into EPUBs via `--epub-embed-font`.
- Both `ebook.css` and `ebook-poetry.css` include `@font-face` rules that reference these font files.
- Pandoc `--resource-path` is set inside `book_formatter.py` so Pandoc searches the current directory and the templates folder for CSS, images and fonts.
- For PDF parity, Colab must have XeLaTeX and the Palatino font available. Installing system fonts (apt) handles this.

Verification
- After running, check `out/` for generated files and open them (Calibre viewer for EPUB; any PDF viewer for PDF) to confirm fonts and cover presence.
