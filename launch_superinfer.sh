Script started on 2026-08-16 10:58:49+00:00 [TERM="xterm" TTY="/dev/pts/1" COLUMNS="158" LINES="24"]
[?2004h]0;richard@sopsinf033: /workspace/re-SuperInfer/vllm-superinfer-v4richard@sopsinf033:/workspace/re-SuperInfer/vllm-superinfer-v4$ cd : ..
[?2004l[?2004h]0;richard@sopsinf033: /workspace/re-SuperInferrichard@sopsinf033:/workspace/re-SuperInfer$ ls
[?2004l[0m[01;32m2601.20309v2.pdf[0m  [01;34mbenchmark_artifacts[0m  [01;34mfull_topo[0m                                    [01;34mvllm-full-superinfer[0m      [01;34mvllm-superinfer-v4[0m
[01;32mDeepSeek_V4.pdf[0m   [01;34mbenchmark_tools[0m      [01;34mmodels--deepseek-ai--DeepSeek-V4-Flash[0m       [01;34mvllm-modern[0m               [01;34mvllm-superinfer-v4.backup2[0m
[01;34mDeepSpec[0m          [01;34mcanada-quant[0m         [01;34mmodels--deepseek-ai--DeepSeek-V4-Flash-0731[0m  [01;34mvllm-modern-custom-mbind[0m
[01;34mSuperInfer[0m        [01;34mcanadataquant[0m        [01;32msuperinfer_deepseekv4_glossary.md[0m            [01;34mvllm-modern-dspark[0m
[01;34mSuperInfer-Old[0m    [01;34mds4[0m                  [01;32mtest_concurrent.py[0m                           [01;34mvllm-modern-full-topo[0m
[01;34m__pycache__[0m       [01;34mflash-memo-ds4[0m       [01;32mtest_concurrent_robust.py[0m                    [01;34mvllm-modern.backup2[0m
[?2004h]0;richard@sopsinf033: /workspace/re-SuperInferrichard@sopsinf033:/workspace/re-SuperInfer$ cllear
[?2004lbash: cllear: command not found
[?2004h]0;richard@sopsinf033: /workspace/re-SuperInferrichard@sopsinf033:/workspace/re-SuperInfer$ clear
[?2004l[H[2J[3J[?2004h]0;richard@sopsinf033: /workspace/re-SuperInferrichard@sopsinf033:/workspace/re-SuperInfer$ la  bash sc  vllm-modern      superinfer-v4.s  /scripts/launch_superinfer.sh .env.vanilla
[?2004lMissing .env.vanilla. Copy .env.example to .env first.
[?2004h]0;richard@sopsinf033: /workspace/re-SuperInferrichard@sopsinf033:/workspace/re-SuperInfer$ [Kbash vllm-superinfer-v4/scripts/launch_superinfer.sh .env.vanilla            basbash [1@.[1@e[1@nv[1@v[1@.[1@a[1Pv[1@v[1@a[1@n[1@i[1@l[1@l[1@a[1@ 
[?2004lbash: .env.vanilla: No such file or directory
[?2004h]0;richard@sopsinf033: /workspace/re-SuperInferrichard@sopsinf033:/workspace/re-SuperInfer$ [Kscript      vllm-superinfer-v4/launchbash .env.vanilla vllm-superinfer-v4/scripts/launch_superinfer.sh [1Pvllm-superinfer-v4/scripts/launch_superinfer.sh .env.vanilla.env.vanilla vllm-superinfer-v4/scripts/launch_superinfer.sh 
[?2004lbash: .env.vanilla: No such file or directory
[?2004h]0;richard@sopsinf033: /workspace/re-SuperInferrichard@sopsinf033:/workspace/re-SuperInfer$ [K[7mCUDA_VISIBLE_DEVICES=0,1 \[27m
[7mscripts/launch_superinfer.sh .env.superinfer[27m[A CUDA_VISIBLE_DEVICES=0,1 \
scripts/launch_superinfer.sh .env.superinfer
[?2004lbash: scripts/launch_superinfer.sh: No such file or directory
[?2004h]0;richard@sopsinf033: /workspace/re-SuperInferrichard@sopsinf033:/workspace/re-SuperInfer$ [Kcd .    cd vllm-superinfer-v4
[?2004l[?2004h]0;richard@sopsinf033: /workspace/re-SuperInfer/vllm-superinfer-v4richard@sopsinf033:/workspace/re-SuperInfer/vllm-superinfer-v4$ [7mCUDA_VISIBLE_DEVICES=0,1 \[27m
[7mscripts/launch_superinfer.sh .env.superinfer[27m[Avllm-superinfer-v4$ CUDA_VISIBLE_DEVICES=0,1 \
scripts/launch_superinfer.sh .env.superinfer
[?2004lStarting profile=superinfer port=8202
Model: /workspace/models/DeepSeek-V4-Flash-0731
Log: /workspace/re-SuperInfer/vllm-superinfer-v4/logs/server_superinfer_8202.log
Command: /workspace/re-SuperInfer/vllm-superinfer-v4/.venv/bin/python -m vllm.entrypoints.cli.main serve /workspace/models/DeepSeek-V4-Flash-0731 --host 0.0.0.0 --port 8202 --served-model-name deepseek-ai/DeepSeek-V4-Flash-0731 --trust-remote-code --tensor-parallel-size 2 --pipeline-parallel-size 1 --gpu-memory-utilization 0.92 --max-model-len 1048576 --max-num-seqs 32 --max-num-batched-tokens 16384 --block-size 256 --kv-cache-dtype fp8 --speculative-config \{\"method\":\"dspark\"\,\"num_speculative_tokens\":5\} --numa-bind --numa-bind-nodes 0 1 --enable-auto-tool-choice --tool-call-parser deepseek_v4 --reasoning-parser deepseek_v4 --enable-prefix-caching --swap-cpu-memory-gb 8 --proactive-swap-budget 2400 --vlt-alpha 3 --vlt-beta-bandwidth 1 --vlt-beta-future 1 --slo-ttft 5 --slo-tbt 0.1 --pin-memory-fix --gh200-topology-tuned --local-cpu-pool-fraction 0.75 --local-swap-bandwidth-bytes-per-s 966367641600 --remote-swap-bandwidth-bytes-per-s 300647710720
Server exited during startup. See /workspace/re-SuperInfer/vllm-superinfer-v4/logs/server_superinfer_8202.log
[?2004h]0;richard@sopsinf033: /workspace/re-SuperInfer/vllm-superinfer-v4richard@sopsinf033:/workspace/re-SuperInfer/vllm-superinfer-v4$ [KCUDA_VISIBLE_DEVICES=0,1 scripts/launch_superinfer.sh .env.superinfer
[?2004lStarting profile=superinfer port=8202
Model: /workspace/models/DeepSeek-V4-Flash-0731
Log: /workspace/re-SuperInfer/vllm-superinfer-v4/logs/server_superinfer_8202.log
Command: /workspace/re-SuperInfer/vllm-superinfer-v4/.venv/bin/python -m vllm.entrypoints.cli.main serve /workspace/models/DeepSeek-V4-Flash-0731 --host 0.0.0.0 --port 8202 --served-model-name deepseek-ai/DeepSeek-V4-Flash-0731 --trust-remote-code --tensor-parallel-size 2 --pipeline-parallel-size 1 --gpu-memory-utilization 0.5 --max-model-len 1048576 --max-num-seqs 32 --max-num-batched-tokens 16384 --block-size 256 --kv-cache-dtype fp8 --speculative-config \{\"method\":\"dspark\"\,\"num_speculative_tokens\":5\} --numa-bind --numa-bind-nodes 0 1 --enable-auto-tool-choice --tool-call-parser deepseek_v4 --reasoning-parser deepseek_v4 --enable-prefix-caching --swap-cpu-memory-gb 8 --proactive-swap-budget 2400 --vlt-alpha 3 --vlt-beta-bandwidth 1 --vlt-beta-future 1 --slo-ttft 5 --slo-tbt 0.1 --pin-memory-fix --gh200-topology-tuned --local-cpu-pool-fraction 0.75 --local-swap-bandwidth-bytes-per-s 966367641600 --remote-swap-bandwidth-bytes-per-s 300647710720
^C
[?2004h]0;richard@sopsinf033: /workspace/re-SuperInfer/vllm-superinfer-v4richard@sopsinf033:/workspace/re-SuperInfer/vllm-superinfer-v4$ [KCUDA_VISIBLE_DEVICES=0,1 scripts/launch_superinfer.sh .env.superinfercd vllm-superinfer-v4[KCUDA_VISIBLE_DEVICES=0,1 scripts/launch_superinfer.sh .env.superinfer[3Pbash .env.vanilla vllm-superinfer-v4/scripts/launch_superinfer.sh [1Pvllm-superinfer-v4/scripts/launch_superinfer.sh .env.vanillaclear[K[1@lear      clear
[?2004l[H[2J[3J[?2004h]0;richard@sopsinf033: /workspace/re-SuperInfer/vllm-superinfer-v4richard@sopsinf033:/workspace/re-SuperInfer/vllm-superinfer-v4$ [7m pkill -9 -f "VLLM::Worker_" || true && pkill -9 -f "multiprocessing.resource_tracker import m[27m[7ma[27m[7main" || true[27m[Ainf033:/workspace/re-SuperInfer/vllm-superinfer-v4$  pkill -9 -f "VLLM::Worker_" || true && pkill -9 -f "multiprocessing.resource_tracker import main" || true
[?2004l[?2004h]0;richard@sopsinf033: /workspace/re-SuperInfer/vllm-superinfer-v4richard@sopsinf033:/workspace/re-SuperInfer/vllm-superinfer-v4$ [Kclear[64@CUDA_VISIBLE_DEVICES=0,1 scripts/launch_superinfer.sh .env.superinfercd vllm-superinfer-v4[KCUDA_VISIBLE_DEVICES=0,1 scripts/launch_superinfer.sh .env.superinfer
[?2004lStarting profile=superinfer port=8202
Model: /workspace/models/DeepSeek-V4-Flash-0731
Log: /workspace/re-SuperInfer/vllm-superinfer-v4/logs/server_superinfer_8202.log
Command: /workspace/re-SuperInfer/vllm-superinfer-v4/.venv/bin/python -m vllm.entrypoints.cli.main serve /workspace/models/DeepSeek-V4-Flash-0731 --host 0.0.0.0 --port 8202 --served-model-name deepseek-ai/DeepSeek-V4-Flash-0731 --trust-remote-code --tensor-parallel-size 2 --pipeline-parallel-size 1 --gpu-memory-utilization 0.4 --max-model-len 1048576 --max-num-seqs 32 --max-num-batched-tokens 16384 --block-size 256 --kv-cache-dtype fp8 --speculative-config \{\"method\":\"dspark\"\,\"num_speculative_tokens\":5\} --numa-bind --numa-bind-nodes 0 1 --enable-auto-tool-choice --tool-call-parser deepseek_v4 --reasoning-parser deepseek_v4 --enable-prefix-caching --swap-cpu-memory-gb 8 --proactive-swap-budget 2400 --vlt-alpha 3 --vlt-beta-bandwidth 1 --vlt-beta-future 1 --slo-ttft 5 --slo-tbt 0.1 --pin-memory-fix --gh200-topology-tuned --local-cpu-pool-fraction 0.75 --local-swap-bandwidth-bytes-per-s 966367641600 --remote-swap-bandwidth-bytes-per-s 300647710720
^C
[?2004h]0;richard@sopsinf033: /workspace/re-SuperInfer/vllm-superinfer-v4richard@sopsinf033:/workspace/re-SuperInfer/vllm-superinfer-v4$ [Kclear
[?2004l[H[2J[3J[?2004h]0;richard@sopsinf033: /workspace/re-SuperInfer/vllm-superinfer-v4richard@sopsinf033:/workspace/re-SuperInfer/vllm-superinfer-v4$ [7mdocker exec richard-base-dev-sysnice pkill -9 -f 'vllm serve|VLLM::EngineCore|VLLM::Worker_'[27mdocker exec richard-base-dev-sysnice pkill -9 -f 'vllm serve|VLLM::EngineCore|VLLM::Worker_'
[?2004lbash: docker: command not found
[?2004h]0;richard@sopsinf033: /workspace/re-SuperInfer/vllm-superinfer-v4richard@sopsinf033:/workspace/re-SuperInfer/vllm-superinfer-v4$ [Kexit
[?2004lexit

Script done on 2026-08-16 11:10:41+00:00 [COMMAND_EXIT_CODE="127"]
