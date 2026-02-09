#!/usr/bin/env python3
"""
Automatic Book Converter - watches a folder and auto-converts new .txt files
"""
import time
import subprocess
from pathlib import Path
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import click


class TextFileHandler(FileSystemEventHandler):
    def __init__(self, output_dir, formats="pdf,epub"):
        self.output_dir = Path(output_dir)
        self.formats = formats
        self.processed = set()  # Track processed files to avoid duplicates

    def on_created(self, event):
        if event.is_directory:
            return

        file_path = Path(event.src_path)

        # Only process .txt files
        if file_path.suffix.lower() != '.txt':
            return

        # Avoid processing the same file multiple times
        if str(file_path) in self.processed:
            return

        click.echo(f"\n📝 New file detected: {file_path.name}")

        # Wait a moment to ensure file is fully written
        time.sleep(1)

        # Extract title from filename (remove .txt extension)
        title = file_path.stem.replace('_', ' ').replace('-', ' ')

        # Run book_formatter
        try:
            formatter_script = Path(__file__).parent / "book_formatter.py"
            cmd = [
                "python3",
                str(formatter_script),
                str(file_path),
                "--title", title,
                "--formats", self.formats,
                "--out-dir", str(self.output_dir)
            ]

            click.echo(f"🔄 Converting '{title}'...")
            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode == 0:
                click.echo(f"✅ Successfully converted: {title}")
                click.echo(f"   Output saved to: {self.output_dir}")
                self.processed.add(str(file_path))
            else:
                click.echo(f"❌ Error converting {file_path.name}:")
                click.echo(result.stderr)

        except Exception as e:
            click.echo(f"❌ Error: {e}")


@click.command()
@click.option("--input-dir", "input_dir", required=True, help="Directory to watch for new .txt files")
@click.option("--output-dir", "output_dir", required=True, help="Directory where PDFs/EPUBs will be saved")
@click.option("--formats", "formats", default="pdf,epub", help="Output formats (pdf,epub,mobi)")
def main(input_dir, output_dir, formats):
    """
    Watch a folder and automatically convert new .txt files to PDF/EPUB.

    Press Ctrl+C to stop monitoring.
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    # Create directories if they don't exist
    input_path.mkdir(parents=True, exist_ok=True)
    output_path.mkdir(parents=True, exist_ok=True)

    click.echo("=" * 60)
    click.echo("📚 Automatic Book Converter - Running")
    click.echo("=" * 60)
    click.echo(f"Watching: {input_path}")
    click.echo(f"Output:   {output_path}")
    click.echo(f"Formats:  {formats}")
    click.echo("\nWaiting for new .txt files... (Press Ctrl+C to stop)\n")

    event_handler = TextFileHandler(output_path, formats)
    observer = Observer()
    observer.schedule(event_handler, str(input_path), recursive=False)
    observer.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        click.echo("\n\n🛑 Stopping automatic converter...")
        observer.stop()
    observer.join()
    click.echo("✅ Converter stopped.")


if __name__ == "__main__":
    main()
