import os

import click
import uvicorn

from dotenv import load_dotenv

load_dotenv('./.env.example')

host = os.getenv("HOST", "")
port = int(os.getenv("APP_PORT", ""))

@click.command()
@click.option(
    "--env",
    type=click.Choice(["local", "dev", "test", "prod"], case_sensitive=False),
    default="local",
)
@click.option(
    "--debug",
    type=click.BOOL,
    is_flag=True,
    default=False,
)
def main(env: str, debug: bool):
    os.environ["ENV"] = env
    os.environ["DEBUG"] = str(debug)
    uvicorn.run(
      app="app.server:app",
      host=host,
      port=port,
      reload=True,
      workers=1,
    )

if __name__ == "__main__":
    main()
