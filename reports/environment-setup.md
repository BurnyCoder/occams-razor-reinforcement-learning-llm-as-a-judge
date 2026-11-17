# Environment setup record

2026-11-17. Initial implementation was pushed and its remote SHA verified before execution. The first setup invocation stopped at `uv: command not found` because the noninteractive WSL shell lacked the user-local binary directory. No project tests or model executions occurred. The setup script now explicitly adds that directory to PATH, matching the existing uv installation. Package symlinks use the approved secondary-drive cache to conserve system-drive space. This fix is published before retrying.
