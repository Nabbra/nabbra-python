import os
import shutil
import functools
import subprocess

from typing import Dict, Callable
from typer import Typer, Exit, Option
from dotenv import load_dotenv
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm, Prompt

console = Console()

app = Typer(
    help="Nabbra Server CLI - Management of Nabbra core application.",
    no_args_is_help=True,
    add_completion=False,
)

env_example = Path(".env.example")
env_file = Path(".env")

required_env_vars=[
    "APP_PORT",
]


def require_env(f: Callable) -> Callable:
    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        if not check_required_env_vars():
            raise Exit(1)
        return f(*args, **kwargs)

    return wrapper


def check_required_env_vars() -> bool:
    if not env_file.exists():
        console.print("\n✗ No .env file found. Please run 'nabbra init' first.", style="red bold")
        return False

    load_dotenv()

    missing_vars = [var for var in required_env_vars if not os.getenv(var)]

    if missing_vars:
        console.print("\n✗ Missing required environment variables:", style="red bold")
        for var in missing_vars:
            console.print(f"  • {var}", style="red")
        console.print(
            "\nPlease run 'notooq init' to set these variables.", style="yellow"
        )
        return False

    return True


def format_env_contents(current_contents: list[str], updates: Dict[str, str]) -> list[str]:
    env_contents = current_contents.copy()

    for var_name, var_value in updates.items():
        var_found = False
        for i, line in enumerate(env_contents):
            if line.strip().startswith(f"{var_name}="):
                env_contents[i] = f'{var_name}="{var_value}"\n'
                var_found = True
                break

        if not var_found:
            env_contents.append(f'{var_name}="{var_value}"\n')

    return env_contents


def handle_env_init_mode(env_updates: Dict[str, str]):
    try:
        with open(env_example, "r") as f:
            current_contents = f.readlines()

        if env_file.exists():
            backup_file = env_file.with_suffix(".backup")
            shutil.copy2(env_file, backup_file)

            console.print(f"Created backup of existing .env at {backup_file}", style="yellow")

        updated_contents = format_env_contents(current_contents, env_updates)
        formatted_text = "".join(updated_contents)

        with open(env_file, "w") as f:
            f.writelines(updated_contents)

        console.print(
            "✓ Created new .env file",
            style="bold green",
        )
    except Exception as e:
        console.print(f"\nError handling .env file: {str(e)}", style="red")
        raise Exit(1)


@app.command()
def init():
    if env_file.exists():
        warning_panel = Panel(
            "[yellow]A project .env file already exists.\n"
            "Proceeding will create a fresh .env file.\n"
            "Your current .env will be backed up to .env.backup[/yellow]",
            title="[red bold]Warning",
            border_style="red",
        )
        console.print("\n", warning_panel, "\n")

        if not Confirm.ask("Would you like to proceed?"):
            raise Exit()

        handle_env_init_mode({
            "APP_PORT": "8888"
        })

        console.print("\nProject successfully initialized!", style="green bold")


@app.command()
@require_env
def run(
    host: str = Option(None, "--host", "-h", help="Bind socket to this host."),
    port: int = Option(None, "--port", "-p", help="Bind socket to this port."),
    reload: bool = Option(True, "--reload/--no-reload", help="Enable auto-reload."),
):
    """Run the FastAPI server using uvicorn."""
    load_dotenv(env_file)

    try:
        final_port = port or int(os.getenv("APP_PORT", "8000"))
        app_path = "app.server:app"

        command = [
            "uvicorn",
            app_path,
            "--port",
            str(final_port),
        ]

        if host:
            command.extend(["--host", host])

        if reload:
            command.append("--reload")

        console.print("\nStarting development server...", style="blue bold")
        console.print(f"Application: {app_path}", style="blue")
        if host:
            console.print(f"Host: {host}", style="blue")
        console.print(f"Port: {final_port}", style="blue")
        console.print(f"Reload: {'enabled' if reload else 'disabled'}", style="blue")
        console.print("\nPress CTRL+C to stop the server\n", style="yellow")

        # Run uvicorn
        subprocess.run(command)

    except KeyboardInterrupt:
        console.print("\nServer stopped", style="yellow")
    except Exception as e:
        console.print(f"\nError starting server: {str(e)}", style="red bold")
        raise Exit(1)


if __name__ == "__main__":
    app()