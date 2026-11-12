# `oai-5g-core` Helm chart

Installs one tenant's isolated OAI 5G Core: NRF, AMF, SMF, UPF, UDR, UDM, and
AUSF, plus fluent-bit sidecars on the AMF and UPF that stream NAS/GTP-U
telemetry to Kafka. This chart is **completely decoupled** from both the
`operator/` Go codebase and the sibling `oai-ran` chart — see
[`../../README.md`](../../README.md) for why.

This chart is installed **by the Harmony Operator**, not directly by end
users — the Operator loads it fresh from disk on every reconcile (see
`operator/internal/helmrelease/installer.go`) and renders per-tenant values
(tenant name, PLMN, image tags) from the fields of an `OaiTestbed` custom
resource.

## Manual install (development/testing only)

```bash
kubectl create namespace tenant-a-space
helm install tenant-a-core . \
  -n tenant-a-space \
  --set tenant.name=tenant-a \
  --set tenant.owner=saleh \
  --set plmn.mcc=001 \
  --set plmn.mnc=01
```

Release and resource names are all prefixed `<tenant.name>-core-*` so they
can coexist in the same namespace as the `oai-ran` chart's `<tenant.name>-ran-*`
resources without colliding.

## Replacing the placeholder images

`values.yaml` points `image.registry`/`image.repository` at placeholder
names. Before real deployment, either:

1. Build/pull the upstream OAI Core images
   ([`oai-cn5g-fed`](https://gitlab.eurecom.fr/oai/cn5g/oai-cn5g-fed)) and
   push them to your private registry, or
2. Vendor the upstream OAI Helm charts as a dependency of this chart instead
   of the hand-rolled per-NF templates here.

For the UPF specifically, set `upf.customImage.enabled=true` and point
`upf.customImage.repository`/`tag` at the image built from
[`../../docker/custom-upf`](../../docker/custom-upf) if you need the baked-in
GTP-U stats exporter.

## Hardware requirements enforced by this chart

* `upf` — scheduled only onto nodes labeled `network.sriov/intel-x710=true`
  and attached to the `intel-x710-dpdk` `NetworkAttachmentDefinition` for
  GTP-U data-plane traffic.

## Cross-chart wiring with `oai-ran`

The `oai-ran` chart's gNB looks up this chart's AMF Service using the naming
convention `<tenant.name>-core-amf` (see `oai-ran`'s `_helpers.tpl`). As long
as both charts are installed with the same `tenant.name`, no extra wiring is
required; override `ran.amfHost` in the `oai-ran` values if you need to point
at an externally-managed core instead.
