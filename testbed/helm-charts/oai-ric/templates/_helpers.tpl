{{/* experiment.slug → <tenant>-<slug>-ric (per-experiment RIC); empty = legacy <tenant>-ric */}}
{{- define "oai-ric.fullname" -}}
{{- if .Values.experiment.slug -}}
{{- printf "%s-%s-ric" .Values.tenant.name .Values.experiment.slug | trunc 63 | trimSuffix "-" -}}
{{- else -}}
{{- printf "%s-ric" .Values.tenant.name | trunc 63 | trimSuffix "-" -}}
{{- end -}}
{{- end -}}

{{- define "oai-ric.labels" -}}
app.kubernetes.io/part-of: harmony-taas
app.kubernetes.io/managed-by: harmony-operator
harmony.telecom/tenant: {{ .Values.tenant.name }}
{{- with .Values.experiment.slug }}
harmony.telecom/experiment: {{ . | quote }}
{{- end }}
{{- end -}}
