{{/*
Common naming helpers, following the standard Helm chart conventions.
*/}}

{{- define "oai-5g-core.fullname" -}}
{{- printf "%s-core" .Values.tenant.name | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{- define "oai-5g-core.labels" -}}
app.kubernetes.io/part-of: harmony-taas
app.kubernetes.io/managed-by: harmony-operator
harmony.telecom/tenant: {{ .Values.tenant.name }}
{{- end -}}

{{- define "oai-5g-core.kafkaTopic" -}}
{{- if .Values.telemetry.kafka.coreTopic -}}
{{ .Values.telemetry.kafka.coreTopic }}
{{- else -}}
{{ printf "tenant-%s.telemetry.core" .Values.tenant.name }}
{{- end -}}
{{- end -}}

{{- define "oai-5g-core.upfImage" -}}
{{- if .Values.upf.customImage.enabled -}}
{{ printf "%s/%s:%s" .Values.image.registry .Values.upf.customImage.repository .Values.upf.customImage.tag }}
{{- else -}}
{{ printf "%s/%s-upf:%s" .Values.image.registry .Values.image.repository .Values.image.tag }}
{{- end -}}
{{- end -}}

{{- define "oai-5g-core.amfImage" -}}
{{- if .Values.amf.customImage.enabled -}}
{{ printf "%s/%s:%s" .Values.image.registry .Values.amf.customImage.repository .Values.amf.customImage.tag }}
{{- else -}}
{{ printf "%s/%s-amf:%s" .Values.image.registry .Values.image.repository .Values.image.tag }}
{{- end -}}
{{- end -}}

{{/*
Stamp every NRF-registered NF pod with one hash of the Core values, so any Core
change rolls ALL of them together. The NRF keeps registrations in memory: rolling
the NRF (and SMF) but not the UPF left the UPF unregistered, the SMF could not
discover it, and PDU sessions came up with no data path. MySQL is excluded.
*/}}
{{- define "oai-5g-core.rolloutAnnotation" -}}
harmony.telecom/core-config: {{ toJson (omit .Values "mysql") | sha256sum | trunc 16 | quote }}
{{- end -}}
