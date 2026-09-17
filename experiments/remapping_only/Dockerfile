ARG BASE_IMAGE
FROM ${BASE_IMAGE} AS original
RUN rm -rf /workspace/oneformer3d /workspace/configs /workspace/src /workspace/scripts
COPY main/oneformer3d/ /workspace/oneformer3d/
COPY main/configs/ /workspace/configs/
COPY tool/src/ /workspace/src/
COPY tool/scripts/ /workspace/scripts/
COPY tool/tools/test.py /workspace/tools/test.py
COPY benchmark_v6.py /workspace/configs/benchmark_v6.py
COPY v6_epoch_1500_fix.pth /workspace/work_dirs/forestmamba_chm_radius16_qp300_2many_v6/v6_epoch_1500_fix.pth
LABEL org.opencontainers.image.title="ForestMamba original v6 benchmark" org.3dtrees.model.source="24a68f9c2c0bc407508a7b6990817509674af74b" org.3dtrees.optimization="none"
WORKDIR /workspace
ENTRYPOINT ["python", "/workspace/src/run.py"]
CMD ["--help"]

FROM original AS remap
COPY remap/oneformer3d/oneformer3d.py remap/oneformer3d/remapping.py remap/oneformer3d/hybrid_nearest.py /workspace/oneformer3d/
LABEL org.opencontainers.image.title="ForestMamba remapping-only v6 benchmark" org.3dtrees.optimization="nearest-neighbor-remapping-only"
