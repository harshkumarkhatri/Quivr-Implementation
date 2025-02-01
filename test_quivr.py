import os
import time
import json
import hashlib
import sqlite3
import asyncio
from pathlib import Path
import logging
from typing import List, Optional
import ssl
import nltk
from quivr_core import Brain
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.prompt import Prompt
from rich.spinner import Spinner
from rich.live import Live
from rich.theme import Theme

def ensure_nltk_resources():
    """Ensure required NLTK resources are downloaded."""
    try:
        try:
            _create_unverified_https_context = ssl._create_unverified_context
        except AttributeError:
            pass
        else:
            ssl._create_default_https_context = _create_unverified_https_context

        resources = ['punkt', 'averaged_perceptron_tagger', 'stopwords']

        for resource in resources:
            try:
                nltk.data.find(f'{resource}')
            except LookupError:
                nltk.download(resource)
        return True
    except Exception as e:
        print(f"Error downloading NLTK resources: {e}")
        return False


class BrainStorageDB:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.connection = sqlite3.connect(db_path)
        self._setup_database()

    def _setup_database(self):
        """Create tables if they don't exist."""
        with self.connection as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS brain_data (
                    id TEXT PRIMARY KEY,
                    name TEXT,
                    created_at REAL,
                    hash TEXT,
                    files TEXT,
                    embedding BLOB
                )
            """)

    def save_brain_data(self, brain_id: str, name: str, files: list, hash_val: str, embedding: bytes):
        """Save brain data into the database."""
        with self.connection as conn:
            conn.execute("""
                INSERT OR REPLACE INTO brain_data (id, name, created_at, hash, files, embedding)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (brain_id, name, time.time(), hash_val, json.dumps(files), embedding))

    def get_brain_data(self, brain_id: str):
        """Retrieve brain data by ID."""
        with self.connection as conn:
            result = conn.execute("""
                SELECT id, name, created_at, hash, files, embedding
                FROM brain_data
                WHERE id = ?
            """, (brain_id,)).fetchone()
        if result:
            return {
                "id": result[0],
                "name": result[1],
                "created_at": result[2],
                "hash": result[3],
                "files": json.loads(result[4]),
                "embedding": result[5]
            }
        return None

    def get_all_brains(self):
        """Retrieve information about all stored brains."""
        with self.connection as conn:
            results = conn.execute("""
                SELECT id, name, created_at, hash
                FROM brain_data
            """).fetchall()
        return [{"id": r[0], "name": r[1], "created_at": r[2], "hash": r[3]} for r in results]


class RepositoryProcessor:
    def __init__(self, brain_name: str = "repo_brain"):
        self.brain_name = brain_name
        self.brain = None
        self.storage = BrainStorageDB(Path.home() / ".quivr" / "brain_storage.db")
        self.supported_extensions = {'.pdf','.docx','.txt'}
        self.setup_logging()
        self.setup_nltk()

    def setup_nltk(self):
        if not ensure_nltk_resources():
            raise RuntimeError("Failed to initialize NLTK resources")

    def setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def get_all_files(self, repository_path: str) -> List[str]:
        """Get all valid files from the repository."""
        all_files = []
        try:
            policies_path = os.path.join(repository_path, "gtet_policies")
            if not os.path.exists(policies_path):
                self.logger.error(f"Policies directory not found: {policies_path}")
                return all_files

            for file in os.listdir(policies_path):
                if file.endswith(('.txt', '.pdf', '.docx')):
                    all_files.append(os.path.join(policies_path, file))

        except Exception as e:
            self.logger.error(f"Error scanning policies directory: {e}")
        return all_files

    def get_files_hash(self, files: List[str]) -> str:
        """Generate a hash for the given files' content."""
        content = ""
        for file_path in sorted(files):
            with open(file_path, 'rb') as f:
                content += f.read().decode(errors="ignore")
        return hashlib.md5(content.encode()).hexdigest()

    async def initialize_brain(self, repository_path: str) -> Optional[Brain]:
        try:
            valid_files = self.get_all_files(repository_path)
            if not valid_files:
                self.logger.error("No valid files found in repository")
                return None

            files_hash = self.get_files_hash(valid_files)
            brain_id = f"{self.brain_name}_{files_hash}"

            stored_data = self.storage.get_brain_data(brain_id)
            if stored_data:
                self.logger.info(f"[Storage Hit] Found stored brain data for {brain_id}")
                # Recreate the brain using stored files and settings
                self.brain = await Brain.afrom_files(name=self.brain_name, file_paths=stored_data["files"])
            else:
                self.logger.info(f"[Storage Miss] Creating new brain for {brain_id}")
                self.brain = await Brain.afrom_files(name=self.brain_name, file_paths=valid_files)

                # Serialize the files and hash, and store them in the database
                self.storage.save_brain_data(
                    brain_id=brain_id,
                    name=self.brain_name,
                    files=valid_files,
                    hash_val=files_hash,
                    embedding=None,  # Embedding storage may require different logic
                )

            # Configure the brain's settings
            self.brain.settings = {
                "model": "claude-3-sonnet-20240229",
                "temperature": 0.7,
                "max_tokens": 2000,
                "parser": "simple"
            }
            return self.brain

        except Exception as e:
            self.logger.error(f"Error initializing brain: {e}")
            return None



class ChatInterface:
    def __init__(self):
        self.console = Console(theme=Theme({
            "info": "cyan",
            "warning": "yellow",
            "error": "red",
            "success": "green"
        }))
        self.processor = None
        self.setup_interface()

    def setup_interface(self):
        self.console.print(Panel.fit(
            "[bold cyan]Welcome to Repository Assistant![/bold cyan]\n"
            "Type your questions and I'll help you find answers from the repository.\n"
            "Type 'exit' to quit, 'help' for commands.",
            border_style="cyan"
        ))

    def show_help(self):
        help_text = """
        Available Commands:
        - help: Show this help message
        - exit: Exit the chat
        - clear: Clear the screen
        - info: Show brain information
        - status: Show storage status
        - reload: Reload the repository
        """
        self.console.print(Panel(Markdown(help_text), title="Help", border_style="cyan"))

    def show_thinking_animation(self):
        return Spinner("dots", text="Thinking...", style="cyan")

    async def start_chat(self):
        try:
            self.processor = RepositoryProcessor()
            repo_path = os.getcwd()

            with Live(self.show_thinking_animation(), refresh_per_second=10) as live:
                live.update("[cyan]Initializing brain...[/cyan]")
                brain = await self.processor.initialize_brain(repo_path)
                if not brain:
                    self.console.print("[error]Failed to initialize repository processor[/error]")
                    return
                live.update("[success]Brain initialized successfully![/success]")

            while True:
                question = Prompt.ask("\n[bold cyan]You[/bold cyan]").strip().lower()
                if question == "exit":
                    self.console.print(Panel("Goodbye!", style="bold yellow"))
                    break
                elif question == "help":
                    self.show_help()
                elif question == "clear":
                    self.console.clear()
                elif question == "info" and self.processor.brain:
                    self.processor.brain.print_info()
                elif question == "status":
                    info = self.processor.storage.get_all_brains()
                    self.console.print(Panel(
                        "Stored Brains:\n" + "\n".join(
                            f"- ID: {brain['id']}, Name: {brain['name']}, Created At: {time.ctime(brain['created_at'])}, Hash: {brain['hash']}"
                            for brain in info
                        ),
                        title="Storage Status",
                        border_style="cyan"
                    ))
                elif question == "reload":
                    with Live(self.show_thinking_animation(), refresh_per_second=10) as live:
                        live.update("[cyan]Reloading brain...[/cyan]")
                        brain = await self.processor.initialize_brain(repo_path)
                        live.update("[success]Brain reloaded successfully![/success]" if brain else "[error]Failed to reload brain[/error]")
                else:
                    with Live(self.show_thinking_animation(), refresh_per_second=10) as live:
                        try:
                            start_time = time.time()
                            answer = await self.processor.brain.aask(question)
                            end_time = time.time()
                            self.console.print(f"\n[bold green]Assistant[/bold green]: {answer}", style="bright_white")
                            self.console.print(f"\n[dim]Processing time: {end_time - start_time:.2f} seconds[/dim]")
                        except Exception as e:
                            self.console.print(f"[error]Error during question processing: {e}[/error]")

        except Exception as e:
            self.console.print(f"[error]Fatal error occurred: {e}[/error]")


if __name__ == "__main__":
    asyncio.run(ChatInterface().start_chat())
