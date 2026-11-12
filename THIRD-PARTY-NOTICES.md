# Third-party licenses

| Path | Applies to | License |
|---|---|---|
| `patches/oai/0001`–`0004` | OpenAirInterface 5G RAN `09c9bb998fa2f30f2df452a0a9002787280389af` (and its FlexRIC submodule `ef6d722f22191eea74089966983da1f5ec1fedd4`, patch 0003) | The patched files are licensed by the OpenAirInterface Software Alliance; at this commit OAI is distributed under the **OAI Collaborative Standards Software License (CSSL) v1.0** (`SPDX: LicenseRef-CSSL-1.0`; OAI moved from the OAI Public License v1.1 to CSSL v1.0 on 2026-03-27, upstream commit 8107939f08). Our changes are contributed under the same license as the files they modify. https://openairinterface.org/oai-cssl/ |
| `patches/flexric/0001`–`0003`, `xapp/`, `images/ric/Dockerfile.flexric.ubuntu`, `images/ric/Dockerfile.dualbrain-layer` | FlexRIC `340c36bc8385dc0b9f5d8b2d51d16ff288acee79` (mosaic5g/flexric, branch dev) | **OAI Public License v1.1** (the license of FlexRIC at that commit). The congestion xApp is a FlexRIC example application and is distributed under the same license. |
| `images/gnb/Dockerfile`, `images/gnb/Dockerfile.rfsim-e2`, `images/gnb/build-rfsim-e2.sh` | build recipes on top of the official `oaisoftwarealliance/oai-gnb` images | recipes: MIT (ours); the images they produce contain OAI (CSSL v1.0) and UHD (GPL-3.0) |
| `testbed/helm-charts/oai-5g-core`, `oai-rfsim` | OAI CN5G / RAN container configuration | chart code: MIT (ours); configuration values derived from OAI examples |
| ONNX Runtime 1.30.0 (downloaded at image build time) | | MIT |

NeuralSmith is proprietary and not included (see `r1/mlengine/NEURALSMITH_CONTRACT.md`). Ollama model weights are
not redistributed (tags and digests only, `testbed/ollama_models.json`); each model has its own license.
