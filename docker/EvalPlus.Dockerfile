# Extend the existing CPU verifier image. Only the pinned evaluator, slim dependencies and runner enter it.
FROM occam-verifier:1
USER root
RUN pip install --no-cache-dir --no-deps evalplus==0.3.1 psutil==7.1.3
COPY evalplus_runner.py /opt/evalplus_runner.py
ENV OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
USER 65534:65534
