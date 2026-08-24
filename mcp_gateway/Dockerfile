FROM python:3.12-slim

WORKDIR /app
COPY mcp_gateway/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt
COPY mcp_gateway /app/mcp_gateway

ENV PYTHONUNBUFFERED=1 \
    MCP_HOST=0.0.0.0 \
    MCP_PORT=8000

EXPOSE 8000
CMD ["python", "-m", "mcp_gateway.server"]
