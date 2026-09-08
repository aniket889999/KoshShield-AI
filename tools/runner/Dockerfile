FROM python:3.13.7-alpine3.22

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /opt/koshshield
COPY --chmod=0555 koshshield_tool_runner.py /opt/koshshield/tool_runner.py

USER 65534:65534
ENTRYPOINT ["python", "-I", "/opt/koshshield/tool_runner.py"]
