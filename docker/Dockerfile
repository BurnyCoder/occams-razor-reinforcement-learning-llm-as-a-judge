# No project directory or credentials enter this runtime; generated code arrives on stdin.
# Pin the resolved base-image digest in manifests/runtime.json before retained training.
FROM python:3.12-slim
RUN pip install --no-cache-dir numpy==2.3.5 sympy==1.14.0
USER 65534:65534
WORKDIR /tmp
