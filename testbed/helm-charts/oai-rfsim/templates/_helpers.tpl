{{/* experiment.slug (per-experiment RAN, docs/specs/experiment-isolation.md) → <tenant>-<slug>-rfsim; empty = legacy <tenant>-rfsim */}}
{{- define "oai-rfsim.fullname" -}}
{{- if .Values.experiment.slug -}}
{{- printf "%s-%s-rfsim" .Values.tenant.name .Values.experiment.slug | trunc 63 | trimSuffix "-" -}}
{{- else -}}
{{- printf "%s-rfsim" .Values.tenant.name | trunc 63 | trimSuffix "-" -}}
{{- end -}}
{{- end -}}

{{- define "oai-rfsim.labels" -}}
app.kubernetes.io/part-of: harmony-taas
app.kubernetes.io/managed-by: harmony-operator
harmony.telecom/tenant: {{ .Values.tenant.name }}
{{- with .Values.experiment.slug }}
harmony.telecom/experiment: {{ . | quote }}
{{- end }}
{{- end -}}

{{- define "oai-rfsim.amfHost" -}}
{{- if .Values.amfHost -}}
{{ .Values.amfHost }}
{{- else -}}
{{ printf "%s-core-amf" .Values.tenant.name }}
{{- end -}}
{{- end -}}

{{- define "oai-rfsim.gnbServiceName" -}}
{{ printf "%s-gnb" (include "oai-rfsim.fullname" .) }}
{{- end -}}

{{- define "oai-rfsim.ricHost" -}}
{{- if .Values.e2Agent.ricHost -}}
{{ .Values.e2Agent.ricHost }}
{{- else if .Values.experiment.slug -}}
{{ printf "%s-%s-ric" .Values.tenant.name .Values.experiment.slug }}
{{- else -}}
{{ printf "%s-ric" .Values.tenant.name }}
{{- end -}}
{{- end -}}
