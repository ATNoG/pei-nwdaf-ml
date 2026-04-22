# Kubernetes Configuration

This document describes everything that must be configured on the Kubernetes cluster to run ML training jobs and expose their resource usage to Prometheus/Grafana.

---

## 1. Namespace

All training jobs run in the namespace configured via `TRAIN_KUBE_NAMESPACE` (default: `default`).  
Create a dedicated namespace if preferred:

```bash
kubectl create namespace nwdaf-ml
```

---

## 2. RBAC — Training Jobs

The ML service creates, inspects, and deletes `batch/v1 Job` objects. The service authenticates using the bearer token in `TRAIN_KUBE_TOKEN`.

### ServiceAccount

```yaml
apiVersion: v1
kind: ServiceAccount
metadata:
  name: nwdaf-ml-trainer
  namespace: nwdaf-ml
```

### Role

```yaml
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  name: nwdaf-ml-job-manager
  namespace: nwdaf-ml
rules:
  - apiGroups: ["batch"]
    resources: ["jobs"]
    verbs: ["create", "get", "list", "delete"]
  - apiGroups: [""]
    resources: ["pods"]
    verbs: ["get", "list"]
```

### RoleBinding

```yaml
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  name: nwdaf-ml-job-manager-binding
  namespace: nwdaf-ml
subjects:
  - kind: ServiceAccount
    name: nwdaf-ml-trainer
    namespace: nwdaf-ml
roleRef:
  kind: Role
  name: nwdaf-ml-job-manager
  apiGroup: rbac.authorization.k8s.io
```

### Generating the bearer token

```bash
kubectl create token nwdaf-ml-trainer --namespace nwdaf-ml --duration=8760h
```

Set the output as `TRAIN_KUBE_TOKEN` in the ML service environment.

---

## 3. RBAC — Resource Exposure (metrics-server)

The resource-usage endpoint (`GET /v1/resources/usage`) queries `metrics.k8s.io/v1beta1` (metrics-server) for live pod CPU/memory. The same service account needs access to this API.

### ClusterRole (metrics read)

```yaml
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRole
metadata:
  name: nwdaf-ml-metrics-reader
rules:
  - apiGroups: ["metrics.k8s.io"]
    resources: ["pods"]
    verbs: ["get", "list"]
```

### ClusterRoleBinding

```yaml
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRoleBinding
metadata:
  name: nwdaf-ml-metrics-reader-binding
subjects:
  - kind: ServiceAccount
    name: nwdaf-ml-trainer
    namespace: nwdaf-ml
roleRef:
  kind: ClusterRole
  name: nwdaf-ml-metrics-reader
  apiGroup: rbac.authorization.k8s.io
```

> **Requirement**: `metrics-server` must be installed in the cluster.  
> Quick install: `kubectl apply -f https://github.com/kubernetes-sigs/metrics-server/releases/latest/download/components.yaml`

---

## 4. Job naming and labels

Each training job is named `ml-train-{model_id}` and carries the following labels on both the Job and its Pod template:

```
app: ml-train
model-id: <model_id>
```

This makes it possible to filter all training resources by label in `kubectl` and in Prometheus/Grafana.

```bash
# All active training jobs
kubectl get jobs -l app=ml-train -n nwdaf-ml

# Job for a specific model
kubectl get job ml-train-<model_id> -n nwdaf-ml

# Pods for a specific model
kubectl get pods -l model-id=<model_id> -n nwdaf-ml
```

---

## 5. Exposing resource usage to Prometheus / Grafana

The cluster's built-in Prometheus integration (via **kube-state-metrics** and **metrics-server**) already exposes per-pod resource metrics. No custom exporter is needed.

### 5.1 Prerequisites

Install [kube-prometheus-stack](https://github.com/prometheus-community/helm-charts/tree/main/charts/kube-prometheus-stack), which bundles Prometheus, Grafana, kube-state-metrics, and node-exporter:

```bash
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo update
helm install kube-prometheus prometheus-community/kube-prometheus-stack \
  --namespace monitoring --create-namespace
```

kube-state-metrics exposes resource **requests/limits** per pod.  
metrics-server exposes live **cpu/memory usage** per pod (scraped via the `kubernetes-pods` scrape job).

### 5.2 PodMonitor for training pods

To ensure Prometheus scrapes pods in the training namespace, apply a `PodMonitor`:

```yaml
apiVersion: monitoring.coreos.com/v1
kind: PodMonitor
metadata:
  name: nwdaf-ml-training-pods
  namespace: monitoring
  labels:
    release: kube-prometheus   # must match the Prometheus Operator selector
spec:
  namespaceSelector:
    matchNames:
      - nwdaf-ml
  selector:
    matchLabels:
      app: ml-train
  podMetricsEndpoints:
    - port: metrics   # only needed if pods expose a /metrics endpoint
      path: /metrics
```

> The training worker pods do not expose a `/metrics` HTTP endpoint, so the `podMetricsEndpoints` block can be omitted. The PodMonitor is still useful for kube-state-metrics label enrichment.

### 5.3 Useful Prometheus queries

**Allocated CPU requests per model (from kube-state-metrics):**
```promql
kube_pod_container_resource_requests{
  namespace="nwdaf-ml",
  resource="cpu"
} * on(pod) group_left(label_model_id)
kube_pod_labels{namespace="nwdaf-ml", label_app="ml-train"}
```

**Allocated memory requests per model:**
```promql
kube_pod_container_resource_requests{
  namespace="nwdaf-ml",
  resource="memory"
} * on(pod) group_left(label_model_id)
kube_pod_labels{namespace="nwdaf-ml", label_app="ml-train"}
```

**Live CPU usage per model (from metrics-server via kubelet cAdvisor):**
```promql
rate(container_cpu_usage_seconds_total{
  namespace="nwdaf-ml",
  pod=~"ml-train-.*",
  container="worker"
}[2m])
```

**Live memory usage per model:**
```promql
container_memory_working_set_bytes{
  namespace="nwdaf-ml",
  pod=~"ml-train-.*",
  container="worker"
}
```

To join with `model-id` label, use:
```promql
container_memory_working_set_bytes{namespace="nwdaf-ml", pod=~"ml-train-.*"}
* on(pod) group_left(label_model_id)
kube_pod_labels{namespace="nwdaf-ml", label_app="ml-train"}
```

### 5.4 Grafana dashboard

1. Open Grafana → **Dashboards** → **New** → **New Dashboard**.
2. Add panels using the PromQL queries above.
3. Set the variable `$model_id` as a **Query** variable:
   ```promql
   label_values(kube_pod_labels{namespace="nwdaf-ml", label_app="ml-train"}, label_model_id)
   ```
4. Filter each panel by `label_model_id=~"$model_id"` to drill into a specific model.

Recommended panels:
| Panel | Metric |
|-------|--------|
| CPU usage (cores) | `container_cpu_usage_seconds_total` rate |
| Memory usage (bytes) | `container_memory_working_set_bytes` |
| CPU request vs usage | overlay of request and live usage |
| Memory request vs usage | overlay of request and live usage |
| Job duration | `kube_job_status_start_time` and `kube_job_status_completion_time` from kube-state-metrics |

### 5.5 kube-state-metrics job metrics

kube-state-metrics also exposes training job lifecycle metrics automatically:

```promql
# Jobs currently active (running)
kube_job_status_active{namespace="nwdaf-ml", job_name=~"ml-train-.*"}

# Jobs that completed successfully
kube_job_complete{namespace="nwdaf-ml", job_name=~"ml-train-.*"}

# Jobs that failed
kube_job_failed{namespace="nwdaf-ml", job_name=~"ml-train-.*"}

# Training duration in seconds (per model)
kube_job_status_completion_time{namespace="nwdaf-ml", job_name=~"ml-train-.*"}
- kube_job_status_start_time{namespace="nwdaf-ml", job_name=~"ml-train-.*"}
```

These require no additional configuration beyond kube-prometheus-stack being installed.
