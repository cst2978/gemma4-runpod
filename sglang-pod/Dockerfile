# Interactive RunPod POD image for serving Gemma 4 with SGLang.
#
# This is the POD (not serverless) variant: it runs sshd + an OpenAI-compatible
# `sglang.launch_server` directly (see start.sh), reachable on port 8000.
#
# WHY a rebuild: the previous image was based on the 2026-07-21 SGLang nightly,
# which mis-serves the gemma4 (multimodal) architecture — every request returns
# only <pad> tokens and the scheduler crashes. We move to a much newer nightly
# and stop force-enabling the gemma4 reasoning/tool-call parsers (suspected to
# interact badly with generation); they're now opt-in via env.
FROM lmsysorg/sglang:nightly-dev-cu12-20260803-12eadf86

# sshd for interactive access (RunPod injects the account key as $PUBLIC_KEY).
RUN apt-get update && apt-get install -y --no-install-recommends openssh-server \
    && rm -rf /var/lib/apt/lists/* && mkdir -p /run/sshd

# Weights cache: on the network volume when attached (/workspace), else ephemeral.
ENV HF_HOME=/workspace/hf \
    HUGGINGFACE_HUB_CACHE=/workspace/hf/hub \
    HF_HUB_ENABLE_HF_TRANSFER=1

COPY start.sh /start.sh
RUN chmod +x /start.sh

EXPOSE 8000 22
CMD ["/start.sh"]
